"""
Custom collate function for AffordanceVLA mixed-stage batches.

Handles the unification of Stage I (VQA) and Stage II (VLA) samples where
Stage-II-only fields (wrist_image, action, shape_token, layout_token) may be
None for Stage I samples.

Rules:
  - Tensor fields present in ALL samples: torch.stack normally
  - Tensor fields with some None: stack non-None values, produce a boolean mask
  - str fields (instruction): kept as list of strings
  - LayoutToken dicts: stack each sub-field separately
  - Metadata (stage, dataset_name): kept as lists
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional

import torch

from src.datasets.base_dataset import AffordanceSample


def affordance_collate_fn(
    batch: List[AffordanceSample],
) -> Dict[str, Any]:
    """Collate a list of AffordanceSample dicts into a batched dict.

    Returns a dict with:
      - image:          (B, 3, H, W) float32
      - instruction:    list[str] of length B
      - bbox:           (B, 4) float32
      - heatmap:        (B, 1, H, W) float32
      - wrist_image:    (B, 3, H, W) float32 | None  (None if all samples lack it)
      - wrist_image_mask: (B,) bool — True where wrist_image is valid
      - action:         (B, D_action) float32 | None
      - action_mask:    (B,) bool — True where action is valid
      - shape_token:    (B, 4096, 8) float32 | None
      - shape_token_mask: (B,) bool — True where shape_token is valid
      - layout_token:   dict with rotation(B,4), scale(B,3), translation(B,3) | None
      - layout_token_mask: (B,) bool — True where layout_token is valid
      - stage:          list[str] of length B
      - dataset_name:   list[str] of length B
    """
    B = len(batch)

    # ── Check if images have uniform size ──
    images = [s["image"] for s in batch]
    heatmaps = [s["heatmap"] for s in batch]
    shapes = {img.shape for img in images}
    mixed_sizes = len(shapes) > 1

    if mixed_sizes:
        # Mixed-dataset batch: images have different resolutions.
        # Resize image/heatmap to max size in batch for VLM stacking,
        # and preserve originals as raw_image (list) for Which2Act cropping.
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)

        from src.datasets.base_dataset import resize_sample

        resized_images = []
        resized_heatmaps = []
        resized_bboxes = []
        raw_image_list = []
        raw_bbox_list = []

        for s in batch:
            img, hm, bb = s["image"], s["heatmap"], s["bbox"]
            # Preserve original for Which2Act cropping
            raw_image_list.append(
                s.get("raw_image") if s.get("raw_image") is not None else img
            )
            raw_bbox_list.append(
                s.get("raw_bbox") if s.get("raw_bbox") is not None else bb
            )
            # Resize to uniform size for stacking
            img_r, hm_r, bb_r = resize_sample(img, hm, bb, max_h, max_w)
            resized_images.append(img_r)
            resized_heatmaps.append(hm_r)
            resized_bboxes.append(bb_r)

        collated: Dict[str, Any] = {
            "image": torch.stack(resized_images),
            "bbox": torch.stack(resized_bboxes),
            "heatmap": torch.stack(resized_heatmaps),
        }
        collated["raw_image"] = raw_image_list
        collated["raw_bbox"] = torch.stack(raw_bbox_list)
    else:
        # Uniform-size batch: stack directly (fast path).
        collated: Dict[str, Any] = {
            "image": torch.stack(images),
            "bbox": torch.stack([s["bbox"] for s in batch]),
            "heatmap": torch.stack(heatmaps),
        }

        # ── Original-resolution image & bbox for Which2Act cropping ──
        raw_images = [s.get("raw_image") for s in batch]
        has_raw = any(v is not None for v in raw_images)
        if has_raw:
            collated["raw_image"] = [
                v if v is not None else s["image"]
                for v, s in zip(raw_images, batch)
            ]
            collated["raw_bbox"] = torch.stack([
                s.get("raw_bbox") if s.get("raw_bbox") is not None else s["bbox"]
                for s in batch
            ])
        else:
            collated["raw_image"] = None
            collated["raw_bbox"] = None

    # ── String / metadata fields: collect as lists ──
    collated["instruction"] = [s["instruction"] for s in batch]
    collated["stage"] = [s["stage"] for s in batch]
    collated["dataset_name"] = [s["dataset_name"] for s in batch]

    # ── Optional tensor fields (may be None for Stage I) ──
    collated.update(_collate_optional_tensor(batch, "wrist_image"))
    collated.update(_collate_optional_tensor(batch, "state"))
    collated.update(_collate_optional_tensor(batch, "action"))
    collated.update(_collate_optional_tensor(batch, "action_is_pad"))
    collated.update(_collate_optional_tensor(batch, "shape_token"))

    # ── Optional layout_token (dict of tensors) ──
    collated.update(_collate_optional_layout(batch))

    return collated


def _collate_optional_tensor(
    batch: List[AffordanceSample],
    key: str,
) -> Dict[str, Any]:
    """Collate an optional tensor field that may be None in some samples.

    Returns:
        {key: stacked_tensor | None, key_mask: bool_tensor}
    """
    values = [s.get(key) for s in batch]
    mask = torch.tensor([v is not None for v in values], dtype=torch.bool)

    if mask.any():
        # Get a reference shape from the first non-None value
        ref = next(v for v in values if v is not None)
        # Replace None with zero tensors of matching shape
        filled = [
            v if v is not None
            else torch.zeros_like(ref)
            for v in values
        ]
        return {
            key: torch.stack(filled),
            f"{key}_mask": mask,
        }
    else:
        return {
            key: None,
            f"{key}_mask": mask,
        }


def _collate_optional_layout(
    batch: List[AffordanceSample],
) -> Dict[str, Any]:
    """Collate the optional layout_token field (dict of tensors).

    Returns:
        {layout_token: {rotation: (B,4), scale: (B,3), translation: (B,3)} | None,
         layout_token_mask: (B,) bool}
    """
    values = [s.get("layout_token") for s in batch]
    mask = torch.tensor([v is not None for v in values], dtype=torch.bool)

    if mask.any():
        # Get reference shapes from first non-None layout
        ref = next(v for v in values if v is not None)

        rotations = []
        scales = []
        translations = []

        for v in values:
            if v is not None:
                rotations.append(v["rotation"])
                scales.append(v["scale"])
                translations.append(v["translation"])
            else:
                rotations.append(torch.zeros_like(ref["rotation"]))
                scales.append(torch.zeros_like(ref["scale"]))
                translations.append(torch.zeros_like(ref["translation"]))

        return {
            "layout_token": {
                "rotation": torch.stack(rotations),
                "scale": torch.stack(scales),
                "translation": torch.stack(translations),
            },
            "layout_token_mask": mask,
        }
    else:
        return {
            "layout_token": None,
            "layout_token_mask": mask,
        }
