#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""End-to-end affordance annotation pipeline.

Orchestrates the four steps described in the paper:
    Step 1 — Rule-based keyframe detection (`keyframe_detection.py`).
    Step 2a — Instruction decomposition via a text LLM
              (`instruction_decomposition.py`).
    Step 2b — Per-keyframe affordance annotation via a VLM
              (`keyframe_affordance_annotation.py`).
    Step 3 — Visual grounding & affordance generation
             (`RexOmni_SAM_Affordance_pipeline.py`), optionally with SAM-3D
             token extraction served at SAM3D_SERVICE_URL.

Stage-I scope: PRISM / AGD20K / RefSpatial / VQA (no Step 1 needed; Steps 2-3
are run on each annotated frame).
Stage-II/III scope: A1 / LIBERO / CALVIN trajectories (run all four steps).

This module is intentionally dataset-agnostic: callers iterate their own
frames / episodes and invoke the helpers below.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .keyframe_detection import KeyframeDetector
from .instruction_decomposition import decompose_instruction
from .keyframe_affordance_annotation import annotate_keyframe, AnnotationResult
from .RexOmni_SAM_Affordance_pipeline import (
    AffordancePipeline,
    AffordancePipelineConfig,
    AffordancePipelineOutput,
)

# Stage I VQA-style sources (see data/VQA/README.md). Used to validate the
# `dataset` argument of `annotate_vqa_sample`.
STAGE1_VQA_SOURCES: Tuple[str, ...] = ("prism", "agd20k", "refspatial", "vqa")

# When non-empty, point-in-bbox QC will reject frames whose affordance point
# falls outside the predicted bbox by more than this many pixels.
POINT_IN_BBOX_TOLERANCE_PX: int = 0

# Optional SAM-3D HTTP service URL. When set, Step 3 also extracts shape +
# layout tokens. Available for both Stage I (set explicitly) and Stage II/III.
SAM3D_SERVICE_URL: Optional[str] = None


@dataclass
class StepOneResult:
    """Output of Step 1: keyframe detection over one trajectory."""
    indices: List[int]
    reasons: List[str]


@dataclass
class StepTwoResult:
    """Output of Steps 2a + 2b for one keyframe."""
    keyframe_index: int
    role: str
    sub_instruction: str
    detection_category: str
    affordance_instruction: str


@dataclass
class FrameAnnotation:
    """End-to-end annotation for one frame.

    Attributes:
        bbox: ``[x_min, y_min, x_max, y_max]`` predicted by Step 3.
        point: ``(x, y)`` interaction point predicted by Step 3.
        heatmap: ``(H, W, 1)`` mask-bounded Gaussian heatmap.
        mask: SAM mask aligned with ``bbox``, or None.
        sam3d: ``{"shape_latent": (4096, 8), "layout": {...}}`` when SAM-3D was
            queried; None otherwise.
        passes_qc: True iff the frame passes the point-in-bbox check.
    """
    bbox: Optional[List[float]]
    point: Optional[Tuple[float, float]]
    heatmap: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    sam3d: Optional[Dict[str, Any]]
    passes_qc: bool


def _point_in_bbox(point: Tuple[float, float], bbox: List[float]) -> bool:
    """Step 4 QC: True if ``point`` lies within ``bbox`` (with tolerance)."""
    if bbox is None or point is None:
        return False
    x, y = point
    x1, y1, x2, y2 = bbox
    tol = float(POINT_IN_BBOX_TOLERANCE_PX)
    return (x1 - tol) <= x <= (x2 + tol) and (y1 - tol) <= y <= (y2 + tol)


def _query_sam3d(
    image: Union[str, Image.Image, np.ndarray],
    mask: np.ndarray,
) -> Dict[str, Any]:
    """Call the SAM-3D HTTP service for shape + layout tokens.

    Args:
        image: RGB image (path, PIL.Image, or HxWx3 array).
        mask: Boolean / uint8 SAM mask aligned with ``image``.

    Returns:
        ``{"shape_latent": np.ndarray (4096, 8), "layout": {"rotation":[..],
        "scale":[..], "translation":[..]}, "raw": <full response>}``.
    """
    if not SAM3D_SERVICE_URL:
        raise RuntimeError(
            "SAM3D_SERVICE_URL is not set; cannot extract SAM-3D tokens."
        )

    import base64
    import io
    import requests

    if isinstance(image, str):
        with open(image, "rb") as f:
            img_bytes = f.read()
    elif isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        img_bytes = buf.getvalue()
    elif isinstance(image, np.ndarray):
        buf = io.BytesIO()
        Image.fromarray(image).convert("RGB").save(buf, format="PNG")
        img_bytes = buf.getvalue()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    mask_buf = io.BytesIO()
    np.save(mask_buf, mask.astype(np.uint8))

    payload = {
        "image_base64": base64.b64encode(img_bytes).decode("utf-8"),
        "mask_base64": base64.b64encode(mask_buf.getvalue()).decode("utf-8"),
    }
    resp = requests.post(
        SAM3D_SERVICE_URL.rstrip("/") + "/extract_tokens",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    return {
        "shape_latent": np.asarray(data.get("shape_latent"), dtype=np.float32),
        "layout": {
            "rotation": data.get("rotation"),
            "scale": data.get("scale"),
            "translation": data.get("translation"),
        },
        "raw": data,
    }


# ============================================================
# Step 1
# ============================================================


def detect_keyframes(
    joint_positions: np.ndarray,
    gripper_openness: np.ndarray,
    fps: float,
    gripper_action: Optional[np.ndarray] = None,
    detector: Optional[KeyframeDetector] = None,
) -> StepOneResult:
    """Run Step 1 — rule-based keyframe detection over one trajectory."""
    detector = detector or KeyframeDetector()
    indices, reasons = detector.detect(
        joint_positions=joint_positions,
        gripper_openness=gripper_openness,
        fps=fps,
        gripper_action=gripper_action,
    )
    return StepOneResult(indices=indices, reasons=reasons)


# ============================================================
# Steps 2a + 2b
# ============================================================


def _to_role(reason: str) -> str:
    """Pick the primary role label out of a possibly-merged reason string."""
    return reason.split(" | ")[0]


def annotate_steps_2a_2b(
    images: List[Union[str, Image.Image, np.ndarray]],
    original_instruction: str,
    step1: StepOneResult,
) -> List[StepTwoResult]:
    """Run Step 2a + Step 2b for each detected keyframe of one trajectory.

    Args:
        images: One image per keyframe in ``step1`` order.
        original_instruction: Long-horizon task instruction.
        step1: Step 1 result (keyframe indices + reasons).

    Returns:
        One ``StepTwoResult`` per keyframe, aligned to ``step1.indices``.
    """
    if len(images) != len(step1.indices):
        raise ValueError(
            f"len(images) ({len(images)}) != len(step1.indices) ({len(step1.indices)})"
        )

    keyframe_sequence: List[Tuple[int, str]] = [
        (idx, _to_role(reason))
        for idx, reason in zip(step1.indices, step1.reasons)
    ]

    sub_instructions = decompose_instruction(
        original_instruction=original_instruction,
        keyframe_sequence=keyframe_sequence,
    )

    by_index = {item["keyframe_index"]: item["sub_instruction"] for item in sub_instructions}

    results: List[StepTwoResult] = []
    for image, (idx, role) in zip(images, keyframe_sequence):
        sub = by_index.get(idx, "")
        ann: AnnotationResult = annotate_keyframe(
            image=image,
            original_instruction=original_instruction,
            keyframe_sequence=keyframe_sequence,
            current_index=idx,
            current_role=role,
            sub_instruction=sub,
        )
        results.append(StepTwoResult(
            keyframe_index=idx,
            role=role,
            sub_instruction=sub,
            detection_category=ann.detection_category,
            affordance_instruction=ann.affordance_instruction,
        ))
    return results


# ============================================================
# Step 3
# ============================================================


def annotate_frame(
    image: Union[str, Image.Image, np.ndarray],
    detection_category: str,
    affordance_instruction: str,
    affordance_pipeline: AffordancePipeline,
    enable_sam3d: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    output_prefix: Optional[str] = None,
) -> FrameAnnotation:
    """Run Step 3 (and optional SAM-3D) on a single frame.

    Args:
        image: Source frame.
        detection_category: Concise noun phrase from Step 2b.
        affordance_instruction: Where-to expression from Step 2b.
        affordance_pipeline: Pre-built ``AffordancePipeline``.
        enable_sam3d: If True, also query SAM-3D for shape + layout tokens.
        output_dir / output_prefix: Forwarded to the affordance pipeline.

    Returns:
        FrameAnnotation with bbox, point, heatmap, mask, optional SAM-3D and
        a Step-4 point-in-bbox flag.
    """
    pipeline_output: AffordancePipelineOutput = affordance_pipeline.process(
        image=image,
        affordance_instructions=[affordance_instruction],
        detection_categories=[detection_category],
        output_dir=output_dir,
        output_prefix=output_prefix,
        save_visualization=False,
    )

    bbox = list(map(float, pipeline_output.boxes[0])) if pipeline_output.boxes else None
    point = (
        (float(pipeline_output.affordance_points[0][0]),
         float(pipeline_output.affordance_points[0][1]))
        if pipeline_output.affordance_points else None
    )
    mask = pipeline_output.masks[0] if pipeline_output.masks else None
    heatmap = pipeline_output.heatmap

    sam3d = None
    if enable_sam3d and mask is not None:
        sam3d = _query_sam3d(image, mask)

    passes_qc = bbox is not None and point is not None and _point_in_bbox(point, bbox)

    return FrameAnnotation(
        bbox=bbox,
        point=point,
        heatmap=heatmap,
        mask=mask,
        sam3d=sam3d,
        passes_qc=passes_qc,
    )


# ============================================================
# Stage I — VQA-style sample annotation
# ============================================================
#
# All four Stage-I sources (PRISM, AGD20K, RefSpatial, VQA) ship with their own native labels — heatmaps, segmentation maps, bounding boxes, or interaction points.


def annotate_vqa_sample(
    image: Union[str, Image.Image, np.ndarray],
    detection_category: str,
    affordance_instruction: str,
    affordance_pipeline: AffordancePipeline,
    dataset: str = "vqa",
    enable_sam3d: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    output_prefix: Optional[str] = None,
) -> FrameAnnotation:
    """Annotate one Stage-I VQA-style sample.

    Stage-I sources (PRISM, AGD20K, RefSpatial, VQA) already provide a textual
    affordance prompt per sample, so Step 1 / Step 2a are skipped and only
    Step 3 (+ optional SAM-3D) is invoked.

    Args:
        image: Source image.
        detection_category: Concise noun phrase that names the target part.
        affordance_instruction: Where-to interaction expression.
        affordance_pipeline: Pre-built ``AffordancePipeline``.
        dataset: One of ``STAGE1_VQA_SOURCES``.
        enable_sam3d: If True, additionally extract shape + layout tokens.
        output_dir / output_prefix: Forwarded to the affordance pipeline.
    """
    if dataset.lower() not in STAGE1_VQA_SOURCES:
        raise ValueError(
            f"Unknown Stage-I VQA source: {dataset!r}. "
            f"Expected one of {STAGE1_VQA_SOURCES}."
        )
    return annotate_frame(
        image=image,
        detection_category=detection_category,
        affordance_instruction=affordance_instruction,
        affordance_pipeline=affordance_pipeline,
        enable_sam3d=enable_sam3d,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )
