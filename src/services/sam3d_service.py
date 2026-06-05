#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SAM3D service API

start: 
conda activate SAM3D && python src/services/sam3d_service.py --port 17860 --mode latent_custom --inference_steps 16 --preload

mode: full(Stage1+2) | stage1(skip 3DGS) | latent(skip ss_decoder) | full_custom | latent_custom
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

# Make the bundled SAM-3D package importable.
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sam3d_dir = project_root / "SAM3D"
os.chdir(sam3d_dir)
sys.path.insert(0, str(sam3d_dir))
sys.path.insert(0, str(sam3d_dir / "notebook"))


app = Flask(__name__)

inference_model = None

# Default inference mode — overridden by --mode at startup.
# mode: "full" | "stage1" | "latent" | "full_custom" | "latent_custom"
DEFAULT_MODE = "stage1"
DEFAULT_INFERENCE_STEPS = 25


def load_model():
    """Lazily load the SAM-3D inference model."""
    global inference_model
    if inference_model is None:
        print("Loading SAM-3D ...")
        from inference import Inference

        tag = "hf"
        config_path = str(sam3d_dir / f"checkpoints/{tag}/pipeline.yaml")
        inference_model = Inference(config_path, compile=False)
        print("SAM-3D loaded.")
    return inference_model


def decode_image(base64_str: str) -> np.ndarray:
    """Decode a base64 string into an HxWx3 NumPy image."""
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    return np.array(img)


def decode_mask(base64_str: str) -> np.ndarray:
    """Decode a base64 string into a boolean HxW mask."""
    mask_data = base64.b64decode(base64_str)
    mask = Image.open(io.BytesIO(mask_data))
    mask = np.array(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask > 0
    return mask


@app.route('/health', methods=['GET'])
def health_check():
    """Health-check endpoint."""
    default_mode = app.config.get("DEFAULT_MODE", "stage1")
    return jsonify({
        "status": "healthy",
        "model_loaded": inference_model is not None,
        "default_mode": default_mode,
    })


@app.route('/info', methods=['GET'])
def service_info():
    """Service-info endpoint."""
    default_mode = app.config.get("DEFAULT_MODE", "stage1")
    default_steps = app.config.get("DEFAULT_INFERENCE_STEPS", 25)
    mode_desc = {
        "full": "Stage 1 + Stage 2 (full 3DGS)",
        "stage1": "Stage 1 only (skip 3DGS)",
        "latent": "Latent only (DiT generation only)",
        "full_custom": f"Full pipeline with custom sampling steps ({default_steps})",
        "latent_custom": f"Latent only with custom sampling steps ({default_steps})",
    }
    return jsonify({
        "service": "SAM3D API",
        "default_mode": default_mode,
        "default_inference_steps": default_steps,
        "mode_description": mode_desc.get(default_mode, default_mode),
        "model_loaded": inference_model is not None,
        "available_modes": ["full", "stage1", "latent", "full_custom", "latent_custom"],
        "endpoints": {
            "/health": "Health check",
            "/info": "Service info",
            "/extract_tokens": "Return shape latent + layout tokens",
            "/extract_tokens_with_ply": "Return tokens plus a PLY file",
            "/extract_tokens_batch": "Batch token extraction",
        }
    })


def _run_latent_only_pipeline(model, image, mask, seed, detailed_timing=False, inference_steps=None):
    """Latent mode: DiT generation only (skip ss_decoder and coords extraction).

    Returns shape_latent (4096, 8) and layout (rotation, translation, scale).
    """
    import torch

    pipeline = model._pipeline
    timing = {} if detailed_timing else None

    total_start = time.time()

    # 1. Merge image with mask into an RGBA image
    if detailed_timing:
        t0 = time.time()
    rgba_image = model.merge_mask_to_rgba(image, mask)
    if detailed_timing:
        timing["1_merge_rgba"] = time.time() - t0

    # 2. Compute pointmap (depth estimation)
    if detailed_timing:
        t0 = time.time()
    pointmap_dict = pipeline.compute_pointmap(rgba_image)
    pointmap = pointmap_dict["pointmap"]
    if detailed_timing:
        timing["2_compute_pointmap"] = time.time() - t0

    # 3. Preprocess image
    if detailed_timing:
        t0 = time.time()
    ss_input_dict = pipeline.preprocess_image(
        rgba_image, pipeline.ss_preprocessor, pointmap=pointmap
    )
    if detailed_timing:
        timing["3_preprocess_image"] = time.time() - t0

    # 4. Seed
    torch.manual_seed(seed)

    # 5. Generator + sampling step override
    ss_generator = pipeline.models["ss_generator"]
    prev_inference_steps = ss_generator.inference_steps
    if inference_steps is not None:
        ss_generator.inference_steps = inference_steps
    actual_steps = ss_generator.inference_steps

    # 6. Latent shape spec
    bs = ss_input_dict["image"].shape[0]
    if pipeline.is_mm_dit():
        latent_shape_dict = {
            k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
            for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
        }
    else:
        latent_shape_dict = (bs,) + (4096, 8)

    # 7. Build condition embeddings
    if detailed_timing:
        t0 = time.time()
    condition_args, condition_kwargs = pipeline.get_condition_input(
        pipeline.condition_embedders["ss_condition_embedder"],
        ss_input_dict,
        pipeline.ss_condition_input_mapping,
    )
    if detailed_timing:
        timing["4_condition_embedding"] = time.time() - t0

    # 8. Run the DiT generator (cross-attention with the condition)
    if detailed_timing:
        t0 = time.time()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=pipeline.shape_model_dtype):
            return_dict = ss_generator(
                latent_shape_dict,
                ss_input_dict["image"].device,
                *condition_args,
                **condition_kwargs,
            )
            if not pipeline.is_mm_dit():
                return_dict = {"shape": return_dict}
    if detailed_timing:
        timing["5_ss_generator_dit"] = time.time() - t0

    # Latent-only mode stops here — skip ss_decoder and coords extraction.

    # 9. Extract shape token
    shape_latent = return_dict["shape"]
    shape_np = shape_latent.cpu().numpy()
    if shape_np.ndim == 3:
        shape_np = shape_np[0]  # (4096, 8)

    # 10. Decode layout (rotation, translation, scale)
    if detailed_timing:
        t0 = time.time()
    pointmap_scale = ss_input_dict.get("pointmap_scale", None)
    pointmap_shift = ss_input_dict.get("pointmap_shift", None)
    pose_dict = pipeline.pose_decoder(
        return_dict,
        scene_scale=pointmap_scale,
        scene_shift=pointmap_shift,
    )
    if detailed_timing:
        timing["6_pose_decoder"] = time.time() - t0

    # Restore the generator's previous inference steps
    ss_generator.inference_steps = prev_inference_steps
    
    total_time = time.time() - total_start
    
    return {
        "shape_latent": shape_np,
        "rotation": pose_dict.get("rotation"),
        "translation": pose_dict.get("translation"),
        "scale": pose_dict.get("scale"),
        "inference_time": total_time,
        "timing": timing,
        "mode": "latent_only",
        "inference_steps": actual_steps,
    }


def _run_stage1_pipeline(model, image, mask, seed, detailed_timing=False):
    """Stage-1 mode: DiT + ss_decoder + coords extraction (skip Stage-2 3DGS)."""
    import torch
    from sam3d_objects.pipeline.inference_utils import (
        prune_sparse_structure,
        downsample_sparse_structure,
    )

    pipeline = model._pipeline
    timing = {} if detailed_timing else None

    total_start = time.time()

    # 1. Merge image with mask into an RGBA image
    if detailed_timing:
        t0 = time.time()
    rgba_image = model.merge_mask_to_rgba(image, mask)
    if detailed_timing:
        timing["1_merge_rgba"] = time.time() - t0

    # 2. Compute pointmap (depth estimation)
    if detailed_timing:
        t0 = time.time()
    pointmap_dict = pipeline.compute_pointmap(rgba_image)
    pointmap = pointmap_dict["pointmap"]
    if detailed_timing:
        timing["2_compute_pointmap"] = time.time() - t0

    # 3. Preprocess image
    if detailed_timing:
        t0 = time.time()
    ss_input_dict = pipeline.preprocess_image(
        rgba_image, pipeline.ss_preprocessor, pointmap=pointmap
    )
    if detailed_timing:
        timing["3_preprocess_image"] = time.time() - t0

    # 4. Seed
    torch.manual_seed(seed)

    # 5. Generator + decoder
    ss_generator = pipeline.models["ss_generator"]
    ss_decoder = pipeline.models["ss_decoder"]

    # 6. Latent shape spec
    bs = ss_input_dict["image"].shape[0]
    if pipeline.is_mm_dit():
        latent_shape_dict = {
            k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
            for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
        }
    else:
        latent_shape_dict = (bs,) + (4096, 8)

    # 7. Build condition embeddings
    if detailed_timing:
        t0 = time.time()
    condition_args, condition_kwargs = pipeline.get_condition_input(
        pipeline.condition_embedders["ss_condition_embedder"],
        ss_input_dict,
        pipeline.ss_condition_input_mapping,
    )
    if detailed_timing:
        timing["4_condition_embedding"] = time.time() - t0

    # 8. Run the DiT generator
    if detailed_timing:
        t0 = time.time()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=pipeline.shape_model_dtype):
            return_dict = ss_generator(
                latent_shape_dict,
                ss_input_dict["image"].device,
                *condition_args,
                **condition_kwargs,
            )
            if not pipeline.is_mm_dit():
                return_dict = {"shape": return_dict}
    if detailed_timing:
        timing["5_ss_generator_dit"] = time.time() - t0

    # 9. ss_decoder: decode the shape latent into a voxel grid
    if detailed_timing:
        t0 = time.time()
    shape_latent = return_dict["shape"]
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=pipeline.shape_model_dtype):
            ss = ss_decoder(
                shape_latent.permute(0, 2, 1)
                .contiguous()
                .view(shape_latent.shape[0], 8, 16, 16, 16)
            )
    if detailed_timing:
        timing["6_ss_decoder"] = time.time() - t0

    # 10. Extract occupied voxel coordinates
    if detailed_timing:
        t0 = time.time()
    coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()
    coords_original = coords.clone()
    if detailed_timing:
        timing["7_coords_extraction"] = time.time() - t0

    # 11. Voxel downsampling
    if detailed_timing:
        t0 = time.time()
    if pipeline.downsample_ss_dist > 0:
        coords = prune_sparse_structure(
            coords,
            max_neighbor_axes_dist=pipeline.downsample_ss_dist,
        )
    coords, downsample_factor = downsample_sparse_structure(coords)
    if detailed_timing:
        timing["8_coords_downsample"] = time.time() - t0

    # 12. Persist the extracted coordinates back into the dict
    return_dict["coords"] = coords
    return_dict["coords_original"] = coords_original
    return_dict["downsample_factor"] = downsample_factor

    # 13. Decode layout
    if detailed_timing:
        t0 = time.time()
    pointmap_scale = ss_input_dict.get("pointmap_scale", None)
    pointmap_shift = ss_input_dict.get("pointmap_shift", None)
    pose_dict = pipeline.pose_decoder(
        return_dict,
        scene_scale=pointmap_scale,
        scene_shift=pointmap_shift,
    )
    if detailed_timing:
        timing["9_pose_decoder"] = time.time() - t0

    # 14. Apply downsample factor to scale
    if "scale" in pose_dict:
        pose_dict["scale"] = pose_dict["scale"] * downsample_factor

    # 15. Convert voxel indices to normalized coordinates
    voxel = coords[:, 1:].float() / 64 - 0.5

    total_time = time.time() - total_start

    shape_np = shape_latent.cpu().numpy()
    if shape_np.ndim == 3:
        shape_np = shape_np[0]
    
    return {
        "shape_latent": shape_np,
        "rotation": pose_dict.get("rotation"),
        "translation": pose_dict.get("translation"),
        "scale": pose_dict.get("scale"),
        "coords": coords,
        "coords_original": coords_original,
        "voxel": voxel,
        "downsample_factor": downsample_factor,
        "inference_time": total_time,
        "timing": timing,
        "mode": "stage1_only",
    }


def _run_full_pipeline(model, image, mask, seed, include_gaussian_data=False, detailed_timing=False, inference_steps=None):
    """Full mode: Stage 1 + Stage 2 (full 3DGS generation)."""
    timing = {} if detailed_timing else None

    total_start = time.time()

    if detailed_timing:
        t0 = time.time()
    output = model._pipeline.run(
        model.merge_mask_to_rgba(image, mask),
        None,
        seed,
        stage1_only=False,
        with_mesh_postprocess=False,
        with_texture_baking=False,
        with_layout_postprocess=True,
        use_vertex_color=True,
        stage1_inference_steps=inference_steps,
        pointmap=None,
    )
    if detailed_timing:
        timing["full_pipeline"] = time.time() - t0

    total_time = time.time() - total_start

    # Shape latent

    shape_latent = output.get("shape")
    shape_latent_data = None
    shape_latent_stats = None
    
    if shape_latent is not None:
        shape_np = shape_latent.cpu().numpy()
        if shape_np.ndim == 3:
            shape_np = shape_np[0]
        
        shape_latent_data = shape_np.tolist()
        shape_latent_stats = {
            "shape": list(shape_np.shape),
            "mean": shape_np.mean(axis=0).tolist(),
            "std": shape_np.std(axis=0).tolist(),
            "min": shape_np.min(axis=0).tolist(),
            "max": shape_np.max(axis=0).tolist(),
        }
    
    # Voxel coordinates
    voxel_coords = output.get("voxel")
    voxel_coords_data = None
    if voxel_coords is not None:
        voxel_np = voxel_coords.cpu().numpy() if hasattr(voxel_coords, 'cpu') else np.array(voxel_coords)
        voxel_coords_data = voxel_np.tolist()
    
    coords = output.get("coords")
    coords_data = None
    if coords is not None:
        coords_np = coords.cpu().numpy() if hasattr(coords, 'cpu') else np.array(coords)
        coords_data = {
            "shape": list(coords_np.shape),
            "num_voxels": len(coords_np),
        }
    
    # Layout token
    rotation = output.get("rotation")
    translation = output.get("translation")
    scale = output.get("scale")
    
    layout_token = {
        "rotation": rotation.cpu().numpy().tolist() if rotation is not None else None,
        "translation": translation.cpu().numpy().tolist() if translation is not None else None,
        "scale": scale.cpu().numpy().tolist() if scale is not None else None,
    }
    
    # 3DGS statistics
    gs = output.get("gs")
    gaussian_stats = None
    gs_data = None
    
    if gs is not None:
        gaussian_stats = {
            "num_gaussians": len(gs._xyz) if hasattr(gs, '_xyz') else None,
            "xyz_mean": gs._xyz.mean(dim=0).cpu().numpy().tolist() if hasattr(gs, '_xyz') else None,
            "xyz_std": gs._xyz.std(dim=0).cpu().numpy().tolist() if hasattr(gs, '_xyz') else None,
            "features_dc_shape": list(gs._features_dc.shape) if hasattr(gs, '_features_dc') else None,
        }
        
        if include_gaussian_data:
            gs_data = {
                "xyz": gs._xyz.cpu().numpy().tolist() if hasattr(gs, '_xyz') else None,
                "features_dc": gs._features_dc.cpu().numpy().tolist() if hasattr(gs, '_features_dc') else None,
                "scaling": gs._scaling.cpu().numpy().tolist() if hasattr(gs, '_scaling') else None,
                "rotation": gs._rotation.cpu().numpy().tolist() if hasattr(gs, '_rotation') else None,
                "opacity": gs._opacity.cpu().numpy().tolist() if hasattr(gs, '_opacity') else None,
            }
    
    return {
        "shape_latent_data": shape_latent_data,
        "shape_latent_stats": shape_latent_stats,
        "voxel_coords_data": voxel_coords_data,
        "coords_data": coords_data,
        "layout_token": layout_token,
        "rotation": rotation,
        "translation": translation,
        "scale": scale,
        "gaussian_stats": gaussian_stats,
        "gs_data": gs_data,
        "gs": gs,  # raw gs object — used by /extract_tokens_with_ply
        "inference_time": total_time,
        "timing": timing,
        "mode": "full",
        "inference_steps": inference_steps if inference_steps else 25,
    }


@app.route('/extract_tokens', methods=['POST'])
def extract_tokens():
    """Return shape_latent (4096, 8) + layout (rotation, translation, scale)."""
    try:
        data = request.json

        if 'image_base64' not in data or 'mask_base64' not in data:
            return jsonify({"error": "image_base64 and mask_base64 are required"}), 400

        image = decode_image(data['image_base64'])
        mask = decode_mask(data['mask_base64'])
        seed = data.get('seed', 42)
        include_gaussian_data = data.get('include_gaussian_data', False)
        detailed_timing = data.get('detailed_timing', False)

        default_mode = app.config.get("DEFAULT_MODE", "stage1")
        default_steps = app.config.get("DEFAULT_INFERENCE_STEPS", 25)

        inference_steps = data.get('inference_steps', None)

        use_custom_steps = False
        if 'mode' in data:
            mode = data['mode']
            if mode == "full":
                tokens_only = False
                latent_only = False
            elif mode == "stage1":
                tokens_only = True
                latent_only = False
            elif mode == "latent":
                tokens_only = True
                latent_only = True
            elif mode == "full_custom":
                tokens_only = False
                latent_only = False
                use_custom_steps = True
                if inference_steps is None:
                    inference_steps = default_steps
            elif mode == "latent_custom":
                tokens_only = True
                latent_only = True
                use_custom_steps = True
                if inference_steps is None:
                    inference_steps = default_steps
            else:
                return jsonify({
                    "error": f"Invalid mode: {mode}. "
                             "Expected one of: full / stage1 / latent / full_custom / latent_custom"
                }), 400
        elif 'tokens_only' in data or 'latent_only' in data:
            # Backward-compatible flag-based interface
            tokens_only = data.get('tokens_only', True)
            latent_only = data.get('latent_only', False)
        else:
            if default_mode == "full":
                tokens_only = False
                latent_only = False
            elif default_mode == "stage1":
                tokens_only = True
                latent_only = False
            elif default_mode == "latent":
                tokens_only = True
                latent_only = True
            elif default_mode == "full_custom":
                tokens_only = False
                latent_only = False
                use_custom_steps = True
                if inference_steps is None:
                    inference_steps = default_steps
            elif default_mode == "latent_custom":
                tokens_only = True
                latent_only = True
                use_custom_steps = True
                if inference_steps is None:
                    inference_steps = default_steps
            else:
                tokens_only = True
                latent_only = False
        
        model = load_model()

        if latent_only:
            # Latent only — fastest path; skips ss_decoder and coords
            output = _run_latent_only_pipeline(
                model, image, mask, seed, 
                detailed_timing=detailed_timing,
                inference_steps=inference_steps,
            )
            
            shape_np = output["shape_latent"]
            shape_latent_stats = {
                "shape": list(shape_np.shape),
                "mean": shape_np.mean(axis=0).tolist(),
                "std": shape_np.std(axis=0).tolist(),
                "min": shape_np.min(axis=0).tolist(),
                "max": shape_np.max(axis=0).tolist(),
            }
            
            rotation = output["rotation"]
            translation = output["translation"]
            scale = output["scale"]
            
            result = {
                "shape_latent": shape_np.tolist(),
                "shape_latent_stats": shape_latent_stats,
                "voxel_coords": None,
                "coords_info": None,
                "layout_token": {
                    "rotation": rotation.cpu().numpy().tolist() if rotation is not None else None,
                    "translation": translation.cpu().numpy().tolist() if translation is not None else None,
                    "scale": scale.cpu().numpy().tolist() if scale is not None else None,
                },
                "rotation": rotation.cpu().numpy().tolist() if rotation is not None else None,
                "translation": translation.cpu().numpy().tolist() if translation is not None else None,
                "scale": scale.cpu().numpy().tolist() if scale is not None else None,
                "gaussian_stats": None,
                "gaussian_data": None,
                "inference_time": output["inference_time"],
                "inference_steps": output.get("inference_steps", 25),
                "mode": "latent_only" if not use_custom_steps else "latent_custom",
                "tokens_only": True,
                "latent_only": True,
                "shape_token": shape_latent_stats,
            }
            
            if detailed_timing and output.get("timing"):
                result["timing"] = output["timing"]
            
        elif tokens_only:
            # Stage-1 — full Stage 1 (DiT + ss_decoder), skip Stage 2 (3DGS)
            output = _run_stage1_pipeline(
                model, image, mask, seed,
                detailed_timing=detailed_timing
            )
            
            shape_np = output["shape_latent"]
            shape_latent_stats = {
                "shape": list(shape_np.shape),
                "mean": shape_np.mean(axis=0).tolist(),
                "std": shape_np.std(axis=0).tolist(),
                "min": shape_np.min(axis=0).tolist(),
                "max": shape_np.max(axis=0).tolist(),
            }
            
            rotation = output["rotation"]
            translation = output["translation"]
            scale = output["scale"]
            
            coords = output.get("coords")
            coords_data = None
            if coords is not None:
                coords_np = coords.cpu().numpy() if hasattr(coords, 'cpu') else np.array(coords)
                coords_data = {
                    "shape": list(coords_np.shape),
                    "num_voxels": len(coords_np),
                }
            
            voxel = output.get("voxel")
            voxel_data = None
            if voxel is not None:
                voxel_np = voxel.cpu().numpy() if hasattr(voxel, 'cpu') else np.array(voxel)
                voxel_data = voxel_np.tolist()
            
            result = {
                "shape_latent": shape_np.tolist(),
                "shape_latent_stats": shape_latent_stats,
                "voxel_coords": voxel_data,
                "coords_info": coords_data,
                "layout_token": {
                    "rotation": rotation.cpu().numpy().tolist() if rotation is not None else None,
                    "translation": translation.cpu().numpy().tolist() if translation is not None else None,
                    "scale": scale.cpu().numpy().tolist() if scale is not None else None,
                },
                "rotation": rotation.cpu().numpy().tolist() if rotation is not None else None,
                "translation": translation.cpu().numpy().tolist() if translation is not None else None,
                "scale": scale.cpu().numpy().tolist() if scale is not None else None,
                "gaussian_stats": None,
                "gaussian_data": None,
                "inference_time": output["inference_time"],
                "mode": "stage1_only",
                "tokens_only": True,
                "latent_only": False,
                "shape_token": shape_latent_stats,
            }
            
            if detailed_timing and output.get("timing"):
                result["timing"] = output["timing"]
            
        else:
            # Full — Stage 1 + Stage 2 (3DGS), slowest path
            output = _run_full_pipeline(
                model, image, mask, seed,
                include_gaussian_data=include_gaussian_data,
                detailed_timing=detailed_timing,
                inference_steps=inference_steps,
            )
            
            rotation = output["rotation"]
            translation = output["translation"]
            scale = output["scale"]
            
            result = {
                "shape_latent": output["shape_latent_data"],
                "shape_latent_stats": output["shape_latent_stats"],
                "voxel_coords": output["voxel_coords_data"],
                "coords_info": output["coords_data"],
                "layout_token": output["layout_token"],
                "rotation": rotation.cpu().numpy().tolist() if rotation is not None else None,
                "translation": translation.cpu().numpy().tolist() if translation is not None else None,
                "scale": scale.cpu().numpy().tolist() if scale is not None else None,
                "gaussian_stats": output["gaussian_stats"],
                "gaussian_data": output["gs_data"] if include_gaussian_data else None,
                "inference_time": output["inference_time"],
                "inference_steps": output.get("inference_steps", 25),
                "mode": "full" if not use_custom_steps else "full_custom",
                "tokens_only": False,
                "latent_only": False,
                "shape_token": output["shape_latent_stats"],
            }
            
            if detailed_timing and output.get("timing"):
                result["timing"] = output["timing"]
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/extract_tokens_with_ply', methods=['POST'])
def extract_tokens_with_ply():
    """Run the full pipeline and return tokens plus a base64-encoded PLY.

    Only supported for ``full`` and ``full_custom`` modes.
    """
    import tempfile

    try:
        data = request.json

        if 'image_base64' not in data or 'mask_base64' not in data:
            return jsonify({"error": "image_base64 and mask_base64 are required"}), 400

        image = decode_image(data['image_base64'])
        mask = decode_mask(data['mask_base64'])
        seed = data.get('seed', 42)
        detailed_timing = data.get('detailed_timing', False)

        default_steps = app.config.get("DEFAULT_INFERENCE_STEPS", 25)
        inference_steps = data.get('inference_steps', default_steps)

        model = load_model()

        output = _run_full_pipeline(
            model, image, mask, seed,
            include_gaussian_data=False,
            detailed_timing=detailed_timing,
            inference_steps=inference_steps,
        )

        gs = output.get("gs")
        ply_base64 = None

        if gs is not None:
            with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
                tmp_path = tmp_file.name

            gs.save_ply(tmp_path)

            with open(tmp_path, 'rb') as f:
                ply_data = f.read()
            ply_base64 = base64.b64encode(ply_data).decode('utf-8')

            import os as os_module
            os_module.unlink(tmp_path)
        
        result = {
            "ply_base64": ply_base64,
            "shape_latent_stats": output["shape_latent_stats"],
            "gaussian_stats": output["gaussian_stats"],
            "inference_time": output["inference_time"],
            "inference_steps": output.get("inference_steps", 25),
            "mode": "full_custom" if inference_steps != 25 else "full",
        }
        
        if detailed_timing and output.get("timing"):
            result["timing"] = output["timing"]
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/extract_tokens_batch', methods=['POST'])
def extract_tokens_batch():
    """Batch token extraction. Request: ``items=[{image_base64, mask_base64, seed}, ...]``."""
    try:
        data = request.json
        items = data.get('items', [])

        if not items:
            return jsonify({"error": "items must be a non-empty list"}), 400
        
        model = load_model()
        results = []
        total_start = time.time()
        
        for item in items:
            try:
                image = decode_image(item['image_base64'])
                mask = decode_mask(item['mask_base64'])
                seed = item.get('seed', 42)
                
                start_time = time.time()
                output = model(image, mask, seed=seed)
                inference_time = time.time() - start_time
                
                gs = output.get("gs")
                shape_token = None
                if gs is not None:
                    shape_token = {
                        "xyz_mean": gs._xyz.mean(dim=0).cpu().numpy().tolist() if hasattr(gs, '_xyz') else None,
                        "xyz_std": gs._xyz.std(dim=0).cpu().numpy().tolist() if hasattr(gs, '_xyz') else None,
                        "num_gaussians": len(gs._xyz) if hasattr(gs, '_xyz') else None,
                    }
                
                rotation = output.get("rotation")
                translation = output.get("translation")
                scale = output.get("scale")
                
                layout_token = {
                    "rotation": rotation.cpu().numpy().tolist() if rotation is not None else None,
                    "translation": translation.cpu().numpy().tolist() if translation is not None else None,
                    "scale": scale.cpu().numpy().tolist() if scale is not None else None,
                }
                
                results.append({
                    "shape_token": shape_token,
                    "layout_token": layout_token,
                    "inference_time": inference_time,
                    "success": True,
                })
                
            except Exception as e:
                results.append({
                    "error": str(e),
                    "success": False,
                })
        
        total_time = time.time() - total_start
        
        return jsonify({
            "results": results,
            "total_time": total_time,
            "num_processed": len(results),
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


def parse_args():
    parser = argparse.ArgumentParser(description="SAM-3D HTTP service")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=17861)
    parser.add_argument("--preload", action="store_true",
                        help="Load the model at startup.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "stage1", "latent", "full_custom", "latent_custom"],
        default="stage1",
        help="Default inference mode.",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=25,
        help="DiT flow-matching sampling steps "
             "(only used by full_custom / latent_custom modes).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    DEFAULT_MODE = args.mode
    DEFAULT_INFERENCE_STEPS = args.inference_steps

    print("=" * 60)
    print("SAM-3D HTTP service")
    print("=" * 60)
    print(f"Address          : {args.host}:{args.port}")
    print(f"Default mode     : {args.mode}")
    print(f"Sampling steps   : {args.inference_steps}")

    mode_desc = {
        "full": "Stage 1 + Stage 2 (full 3DGS)",
        "stage1": "Stage 1 only (skip 3DGS)",
        "latent": "Latent only (DiT generation only)",
        "full_custom": f"Full pipeline with custom sampling ({args.inference_steps} steps)",
        "latent_custom": f"Latent only with custom sampling ({args.inference_steps} steps)",
    }
    print(f"  - {mode_desc.get(args.mode, args.mode)}")

    if args.preload:
        print("Preloading model ...")
        load_model()

    print("=" * 60)

    app.config["DEFAULT_MODE"] = args.mode
    app.config["DEFAULT_INFERENCE_STEPS"] = args.inference_steps

    app.run(host=args.host, port=args.port, threaded=True)

