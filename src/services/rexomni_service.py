#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RexOmni + SAM HTTP service.

Launch:
    conda activate rexomni
    python -m src.services.rexomni_service --port 7862 --preload \
        --model_path /path/to/rexomni-finetuned \
        --sam_checkpoint /path/to/sam_vit_h.pth

Endpoints:
    POST /detect              - object detection (category -> bbox)
    POST /pointing            - affordance pointing (instruction -> point + heatmap)
    POST /detect_and_point    - detection + pointing
    POST /affordance_pipeline - full pipeline (detect + SAM segment + point + heatmap)
"""

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from flask import Flask, jsonify, request

# Make the bundled Rex-Omni package importable.
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
rex_omni_dir = project_root / "Rex-Omni"
sys.path.insert(0, str(rex_omni_dir))

# Optional HuggingFace cache override (override REXOMNI_CACHE_DIR to disable
# the hard-coded path below).
cache_dir = Path(os.environ.get("REXOMNI_CACHE_DIR", "/mnt/data/RexOmni"))
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(cache_dir))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir))
os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))


app = Flask(__name__)

rex_model = None
sam_predictor = None

# Default RexOmni checkpoint. Set to a fine-tuned path to use a custom checkpoint;
# set to None to require the caller to pass --model_path explicitly.
REXOMNI_MODEL_PATH = "IDEA-Research/Rex-Omni"

DEFAULT_SAM_CHECKPOINT = "/mnt/data/sam_vit_h_4b8939.pth"
DEFAULT_SAM_MODEL_TYPE = "vit_h"


def load_rex_model(model_path: Optional[str] = None, backend: str = "transformers"):
    """Lazily load the RexOmni wrapper."""
    global rex_model
    if rex_model is None:
        resolved = model_path or REXOMNI_MODEL_PATH
        if resolved is None:
            raise ValueError(
                "REXOMNI_MODEL_PATH is None and no model_path was provided. "
                "Pass --model_path on the CLI or set REXOMNI_MODEL_PATH at the top of this file."
            )
        print(f"Loading RexOmni from {resolved} ...")
        from rex_omni import RexOmniWrapper

        rex_model = RexOmniWrapper(
            model_path=resolved,
            backend=backend,
            max_tokens=2048,
            temperature=0.0,
            top_p=0.05,
            top_k=1,
            repetition_penalty=1.05,
        )
        print("RexOmni loaded.")
    return rex_model


def load_sam_model(checkpoint: str = None, model_type: str = None):
    """Lazily load the SAM predictor."""
    global sam_predictor
    if sam_predictor is None:
        checkpoint = checkpoint or DEFAULT_SAM_CHECKPOINT
        model_type = model_type or DEFAULT_SAM_MODEL_TYPE

        print(f"Loading SAM ({model_type}) ...")
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError:
            raise ImportError(
                "segment-anything is not installed: "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device="cuda")
        sam_predictor = SamPredictor(sam)
        print("SAM loaded.")
    return sam_predictor


def decode_image(base64_str: str) -> Image.Image:
    """Decode a base64 string into a PIL.Image."""
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    return img.convert("RGB")


def encode_array(arr: np.ndarray) -> str:
    """Encode a NumPy array into a base64 string (npy format)."""
    buffer = io.BytesIO()
    np.save(buffer, arr)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_heatmap(
    points: List[Tuple[float, float]],
    h: int,
    w: int,
    sigma_ratio: float = 0.15,
    mask: np.ndarray = None,
) -> np.ndarray:
    """Build a Gaussian heatmap from detected points.

    Args:
        points: List of (x, y) pixel coordinates.
        h: Image height.
        w: Image width.
        sigma_ratio: Sigma relative to ``max(h, w)``.
        mask: Optional binary mask; the heatmap is constrained inside it.

    Returns:
        (H, W) float32 heatmap normalized to [0, 1].
    """
    heatmap = np.zeros((h, w), dtype=np.float32)

    if len(points) == 0:
        return heatmap

    sigma = max(h, w) * sigma_ratio
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    for point_x, point_y in points:
        gaussian = np.exp(
            -((x_coords - point_x) ** 2 + (y_coords - point_y) ** 2) / (2 * sigma**2)
        )
        heatmap += gaussian

    if mask is not None:
        heatmap = heatmap * mask.astype(np.float32)

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


def generate_masks_with_sam(
    image_array: np.ndarray, boxes: List[np.ndarray]
) -> Tuple[List[np.ndarray], List[float]]:
    """Run SAM with each bbox as a prompt and return per-bbox masks.

    Args:
        image_array: RGB image, (H, W, 3).
        boxes: List of bboxes, each formatted as [x1, y1, x2, y2].

    Returns:
        ``(masks, scores)``, with one (H, W) mask and one score per box.
    """
    predictor = load_sam_model()
    predictor.set_image(image_array)

    all_masks = []
    all_scores = []

    for box in boxes:
        box_array = np.array(box)
        masks, scores, _ = predictor.predict(box=box_array, multimask_output=False)
        all_masks.append(masks[0])
        all_scores.append(float(scores[0]))

    return all_masks, all_scores


def extract_boxes_from_predictions(
    predictions: Dict,
) -> Tuple[List[np.ndarray], List[str]]:
    """Extract bboxes (and their categories) from prediction dicts."""
    boxes = []
    categories = []
    for category, detections in predictions.items():
        for detection in detections:
            if detection.get("type") == "box":
                coords = detection.get("coords", [])
                if len(coords) == 4:
                    boxes.append(np.array(coords))
                    categories.append(category)
    return boxes, categories


def extract_points_from_predictions(predictions: Dict) -> List[Tuple[float, float]]:
    """Extract (x, y) points from prediction dicts."""
    points = []
    for category, detections in predictions.items():
        for detection in detections:
            if detection.get("type") == "point":
                coords = detection.get("coords", [])
                if len(coords) >= 2:
                    points.append((coords[0], coords[1]))
    return points


@app.route("/health", methods=["GET"])
def health_check():
    """Health-check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "rex_model_loaded": rex_model is not None,
            "sam_model_loaded": sam_predictor is not None,
        }
    )


@app.route("/detect", methods=["POST"])
def detect():
    """Object detection.

    Request JSON:
        image_base64: base64-encoded RGB image.
        categories:   list of category names.

    Response JSON:
        predictions:    detections grouped by category.
        bboxes:         flat list of all bounding boxes.
        inference_time: seconds.
    """
    try:
        data = request.json

        if "image_base64" not in data:
            return jsonify({"error": "image_base64 is required"}), 400

        image = decode_image(data["image_base64"])
        categories = data.get("categories", ["object"])

        model = load_rex_model()

        start_time = time.time()
        results = model.inference(images=image, task="detection", categories=categories)
        inference_time = time.time() - start_time

        result = results[0]
        predictions = result.get("extracted_predictions", {})

        bboxes = []
        for category, detections in predictions.items():
            for det in detections:
                if det.get("type") == "box":
                    bboxes.append(
                        {
                            "category": category,
                            "coords": list(det.get("coords", [])),
                        }
                    )

        return jsonify(
            {
                "predictions": predictions,
                "bboxes": bboxes,
                "raw_output": result.get("raw_output", ""),
                "inference_time": inference_time,
            }
        )

    except Exception as e:
        import traceback

        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/pointing", methods=["POST"])
def pointing():
    """Affordance pointing.

    Request JSON:
        image_base64: base64-encoded RGB image.
        instruction:  where-to affordance expression.
        sigma_ratio:  optional Gaussian sigma ratio (default 0.15).

    Response JSON:
        predictions, points (list of [x, y]),
        heatmap_base64 (npy bytes, (H, W) float32),
        heatmap_shape, inference_time.
    """
    try:
        data = request.json

        if "image_base64" not in data or "instruction" not in data:
            return jsonify({"error": "image_base64 and instruction are required"}), 400

        image = decode_image(data["image_base64"])
        instruction = data["instruction"]
        sigma_ratio = data.get("sigma_ratio", 0.15)

        model = load_rex_model()
        w, h = image.size

        start_time = time.time()
        results = model.inference(
            images=image, task="pointing", categories=[instruction]
        )
        inference_time = time.time() - start_time

        result = results[0]
        predictions = result.get("extracted_predictions", {})

        points = extract_points_from_predictions(predictions)

        heatmap = generate_heatmap(points=points, h=h, w=w, sigma_ratio=sigma_ratio)

        return jsonify(
            {
                "predictions": predictions,
                "points": [[p[0], p[1]] for p in points],
                "heatmap_base64": encode_array(heatmap),
                "heatmap_shape": list(heatmap.shape),
                "raw_output": result.get("raw_output", ""),
                "inference_time": inference_time,
            }
        )

    except Exception as e:
        import traceback

        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/detect_and_point", methods=["POST"])
def detect_and_point():
    """Detection + affordance pointing in one call (no SAM mask)."""
    try:
        data = request.json

        if "image_base64" not in data:
            return jsonify({"error": "image_base64 is required"}), 400

        image = decode_image(data["image_base64"])
        categories = data.get("categories", ["object"])
        instruction = data.get("instruction", "")
        sigma_ratio = data.get("sigma_ratio", 0.15)

        model = load_rex_model()
        w, h = image.size

        total_start = time.time()

        detection_results = {}
        bboxes = []
        detection_time = 0
        if categories:
            start_time = time.time()
            det_results = model.inference(
                images=image, task="detection", categories=categories
            )
            detection_time = time.time() - start_time

            detection_results = det_results[0].get("extracted_predictions", {})
            for category, detections in detection_results.items():
                for det in detections:
                    if det.get("type") == "box":
                        bboxes.append(
                            {
                                "category": category,
                                "coords": list(det.get("coords", [])),
                            }
                        )

        # Pointing
        pointing_results = {}
        points = []
        pointing_time = 0
        heatmap = np.zeros((h, w), dtype=np.float32)
        if instruction:
            start_time = time.time()
            point_results = model.inference(
                images=image, task="pointing", categories=[instruction]
            )
            pointing_time = time.time() - start_time

            pointing_results = point_results[0].get("extracted_predictions", {})
            points = extract_points_from_predictions(pointing_results)

            heatmap = generate_heatmap(points=points, h=h, w=w, sigma_ratio=sigma_ratio)

        total_time = time.time() - total_start

        return jsonify(
            {
                "detection_results": detection_results,
                "pointing_results": pointing_results,
                "bboxes": bboxes,
                "points": [[p[0], p[1]] for p in points],
                "heatmap_base64": encode_array(heatmap),
                "heatmap_shape": list(heatmap.shape),
                "detection_time": detection_time,
                "pointing_time": pointing_time,
                "total_inference_time": total_time,
            }
        )

    except Exception as e:
        import traceback

        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/affordance_pipeline", methods=["POST"])
def affordance_pipeline():
    """Full pipeline: detection + SAM segmentation + pointing + heatmap.

    Request JSON:
        image_base64:           base64-encoded RGB image.
        categories:             detection categories (bbox prompts).
        affordance_instruction: where-to affordance expression.
        heatmap_sigma_ratio:    optional sigma ratio (default 0.15).

    Response JSON:
        bboxes, masks_base64 (npy (N, H, W)), masks_shape, mask_scores,
        affordance_points, heatmap_base64, heatmap_shape, and per-stage timings.
    """
    try:
        data = request.json

        if "image_base64" not in data:
            return jsonify({"error": "image_base64 is required"}), 400

        image = decode_image(data["image_base64"])
        image_array = np.array(image)
        categories = data.get("categories", ["object"])
        affordance_instruction = data.get("affordance_instruction", "")
        heatmap_sigma_ratio = data.get("heatmap_sigma_ratio", 0.15)

        model = load_rex_model()
        h, w = image_array.shape[:2]

        total_start = time.time()

        # Step 1: object detection
        detection_time = 0
        bboxes = []
        box_categories = []
        if categories:
            start_time = time.time()
            det_results = model.inference(
                images=image, task="detection", categories=categories
            )
            detection_time = time.time() - start_time

            detection_results = det_results[0].get("extracted_predictions", {})
            boxes, box_categories = extract_boxes_from_predictions(detection_results)

            for box, cat in zip(boxes, box_categories):
                bboxes.append(
                    {
                        "category": cat,
                        "coords": box.tolist(),
                    }
                )

        # Step 2: SAM segmentation
        sam_time = 0
        masks = []
        mask_scores = []
        if bboxes:
            start_time = time.time()
            boxes_array = [np.array(b["coords"]) for b in bboxes]
            masks, mask_scores = generate_masks_with_sam(image_array, boxes_array)
            sam_time = time.time() - start_time

        # Step 3: affordance pointing
        pointing_time = 0
        affordance_points = []
        if affordance_instruction:
            start_time = time.time()
            point_results = model.inference(
                images=image, task="pointing", categories=[affordance_instruction]
            )
            pointing_time = time.time() - start_time

            pointing_results = point_results[0].get("extracted_predictions", {})
            affordance_points = extract_points_from_predictions(pointing_results)

        # Step 4: build heatmap (union of SAM masks if available)
        combined_mask = None
        if masks:
            combined_mask = np.zeros((h, w), dtype=bool)
            for mask in masks:
                combined_mask = combined_mask | mask

        heatmap = generate_heatmap(
            points=affordance_points,
            h=h,
            w=w,
            sigma_ratio=heatmap_sigma_ratio,
            mask=combined_mask,
        )

        total_time = time.time() - total_start

        masks_output = None
        masks_shape = [0, h, w]
        if masks:
            masks_array = np.stack(masks, axis=0)  # (N, H, W)
            masks_output = encode_array(masks_array)
            masks_shape = list(masks_array.shape)

        return jsonify(
            {
                "bboxes": bboxes,
                "masks_base64": masks_output,
                "masks_shape": masks_shape,
                "mask_scores": mask_scores,
                "affordance_points": [[p[0], p[1]] for p in affordance_points],
                "heatmap_base64": encode_array(heatmap),
                "heatmap_shape": list(heatmap.shape),
                "detection_time": detection_time,
                "sam_time": sam_time,
                "pointing_time": pointing_time,
                "total_inference_time": total_time,
            }
        )

    except Exception as e:
        import traceback

        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


def parse_args():
    parser = argparse.ArgumentParser(description="RexOmni + SAM HTTP service")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument(
        "--model_path",
        type=str,
        default=REXOMNI_MODEL_PATH,
        help="RexOmni checkpoint path; defaults to REXOMNI_MODEL_PATH at the top of this file.",
    )
    parser.add_argument(
        "--backend", type=str, default="transformers", choices=["transformers", "vllm"]
    )
    parser.add_argument("--sam_checkpoint", type=str, default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default=DEFAULT_SAM_MODEL_TYPE,
        choices=["vit_h", "vit_l", "vit_b"],
    )
    parser.add_argument(
        "--preload", action="store_true", help="Load both models at startup."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("RexOmni + SAM HTTP service")
    print("=" * 60)
    print(f"Address     : {args.host}:{args.port}")
    print(f"RexOmni path: {args.model_path}")
    print(f"Backend     : {args.backend}")
    print(f"SAM ckpt    : {args.sam_checkpoint}")
    print(f"SAM type    : {args.sam_model_type}")

    if args.preload:
        print("\nPreloading models ...")
        load_rex_model(args.model_path, args.backend)
        load_sam_model(args.sam_checkpoint, args.sam_model_type)

    print("=" * 60)

    app.run(host=args.host, port=args.port, threaded=True)
