"""
Base dataset classes and unified sample format for AffordanceVLA.

Defines:
  - AffordanceSample: unified return format for all datasets
  - BaseAffordanceDataset: abstract base class for Stage I / Stage II datasets

Stage I (VQA):  image, instruction, bbox, heatmap
Stage II (VLA): image, instruction, bbox, heatmap + wrist_image, action, shape_token, layout_token
"""

from __future__ import annotations

import abc
from typing import Tuple, TypedDict, Optional

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset


# ═══════════════════════════════════════════════════════════════════════
# Unified Sample Format
# ═══════════════════════════════════════════════════════════════════════


class LayoutToken(TypedDict):
    """SAM-3D layout token with rotation, scale, translation."""
    rotation: torch.Tensor      # (4,) quaternion
    scale: torch.Tensor         # (3,) xyz scale
    translation: torch.Tensor   # (3,) xyz position


class AffordanceSample(TypedDict, total=False):
    """Unified sample format returned by all AffordanceVLA datasets.

    Common fields (all stages):
        image:        (3, H, W) float32 [0, 1] — RGB observation
        instruction:  str — natural language task description
        bbox:         (4,) float32 — [x_min, y_min, x_max, y_max]
        heatmap:      (1, H, W) float32 [0, 1] — affordance heatmap

    Stage II only (None for Stage I):
        wrist_image:  (3, H, W) float32 [0, 1] — wrist/hand camera
        state:        (D_state,) float32 — robot proprioceptive state
        action:       (chunk_size, D_action) float32 — action chunk
        action_is_pad:(chunk_size,) bool — True for padded action steps
        shape_token:  (4096, 8) float32 — SAM-3D shape latent
        layout_token: LayoutToken — SAM-3D layout (rotation, scale, translation)

    Metadata:
        stage:        "stage1" | "stage2"
        dataset_name: "prism" | "a1" | "agd20k" | "refspatial" | "libero" | "calvin"
    """
    # Common (may be resized to fixed target_size for collation)
    image: torch.Tensor
    instruction: str
    bbox: torch.Tensor
    heatmap: torch.Tensor
    # Original-resolution image and bbox for Which2Act cropping.
    # Set when target_size resize is applied; None otherwise (use image/bbox).
    raw_image: Optional[torch.Tensor]
    raw_bbox: Optional[torch.Tensor]
    # Stage II only
    wrist_image: Optional[torch.Tensor]
    state: Optional[torch.Tensor]
    action: Optional[torch.Tensor]
    action_is_pad: Optional[torch.Tensor]
    shape_token: Optional[torch.Tensor]
    layout_token: Optional[LayoutToken]
    # Metadata
    stage: str
    dataset_name: str


# ═══════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════


def image_to_tensor(image_np: np.ndarray) -> torch.Tensor:
    """Convert HWC uint8 numpy image to CHW float32 tensor in [0, 1].

    Args:
        image_np: (H, W, 3) uint8 numpy array.

    Returns:
        (3, H, W) float32 tensor in [0, 1].
    """
    if not image_np.flags.writeable:
        image_np = np.array(image_np)
    if image_np.dtype == np.uint8:
        tensor = torch.from_numpy(image_np).float() / 255.0
    else:
        tensor = torch.from_numpy(image_np.astype(np.float32))
    # HWC → CHW
    if tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(2, 0, 1)
    return tensor


def heatmap_to_tensor(heatmap_np: np.ndarray) -> torch.Tensor:
    """Convert numpy heatmap to (1, H, W) float32 tensor.

    Handles both (H, W, 1) and (H, W) input shapes.

    Args:
        heatmap_np: (H, W, 1) or (H, W) float32 numpy array in [0, 1].

    Returns:
        (1, H, W) float32 tensor.
    """
    tensor = torch.from_numpy(heatmap_np.astype(np.float32))
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)            # (H, W) → (1, H, W)
    elif tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor.permute(2, 0, 1)        # (H, W, 1) → (1, H, W)
    return tensor


def resize_sample(
    image: torch.Tensor,
    heatmap: torch.Tensor,
    bbox: torch.Tensor,
    target_h: int,
    target_w: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resize image, heatmap and scale bbox to a fixed target size.

    Used by datasets with variable-resolution images (AGD20K, RefSpatial)
    to ensure all samples within one dataset share the same spatial size,
    which is required by torch.stack in the collate function.

    Args:
        image:    (3, H_orig, W_orig) float32 [0, 1].
        heatmap:  (1, H_orig, W_orig) float32 [0, 1].
        bbox:     (4,) float32 [x_min, y_min, x_max, y_max] in pixels.
        target_h: Target height.
        target_w: Target width.

    Returns:
        (image, heatmap, bbox) resized to (target_h, target_w)
        with bbox coordinates scaled accordingly.
    """
    _, H_orig, W_orig = image.shape

    if H_orig == target_h and W_orig == target_w:
        return image, heatmap, bbox

    scale_x = target_w / W_orig
    scale_y = target_h / H_orig

    # Resize image: (3, H, W) → (3, H_target, W_target)
    image = F.interpolate(
        image.unsqueeze(0),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # Resize heatmap: (1, H, W) → (1, H_target, W_target)
    heatmap = F.interpolate(
        heatmap.unsqueeze(0),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # Scale bbox pixel coordinates
    bbox = bbox.clone()
    bbox[0] *= scale_x   # x_min
    bbox[1] *= scale_y   # y_min
    bbox[2] *= scale_x   # x_max
    bbox[3] *= scale_y   # y_max

    return image, heatmap, bbox


def build_stage1_sample(
    image: torch.Tensor,
    instruction: str,
    bbox: torch.Tensor,
    heatmap: torch.Tensor,
    dataset_name: str,
    raw_image: Optional[torch.Tensor] = None,
    raw_bbox: Optional[torch.Tensor] = None,
) -> AffordanceSample:
    """Construct a Stage I (VQA) sample with Stage-II fields set to None.

    Args:
        raw_image: Original-resolution image before DataLoader resize.
            Used by Which2Act for high-quality bbox cropping. None when
            no resize is applied (image itself is original resolution).
        raw_bbox: Original-resolution bbox matching raw_image.
    """
    return AffordanceSample(
        image=image,
        instruction=instruction,
        bbox=bbox,
        heatmap=heatmap,
        raw_image=raw_image,
        raw_bbox=raw_bbox,
        wrist_image=None,
        state=None,
        action=None,
        action_is_pad=None,
        shape_token=None,
        layout_token=None,
        stage="stage1",
        dataset_name=dataset_name,
    )


def build_stage2_sample(
    image: torch.Tensor,
    instruction: str,
    bbox: torch.Tensor,
    heatmap: torch.Tensor,
    wrist_image: torch.Tensor,
    state: torch.Tensor,
    action: torch.Tensor,
    action_is_pad: torch.Tensor,
    shape_token: Optional[torch.Tensor],
    layout_token: Optional[LayoutToken],
    dataset_name: str,
) -> AffordanceSample:
    """Construct a Stage II (VLA) sample with all fields populated."""
    return AffordanceSample(
        image=image,
        instruction=instruction,
        bbox=bbox,
        heatmap=heatmap,
        raw_image=None,
        raw_bbox=None,
        wrist_image=wrist_image,
        state=state,
        action=action,
        action_is_pad=action_is_pad,
        shape_token=shape_token,
        layout_token=layout_token,
        stage="stage2",
        dataset_name=dataset_name,
    )


# ═══════════════════════════════════════════════════════════════════════
# Base Dataset Class
# ═══════════════════════════════════════════════════════════════════════


class BaseAffordanceDataset(Dataset, abc.ABC):
    """Abstract base class for all AffordanceVLA datasets.

    Subclasses must implement:
        - stage (class attribute): "stage1" or "stage2"
        - dataset_name (class attribute): e.g. "prism", "a1"
        - __len__()
        - __getitem__(idx) -> AffordanceSample
    """

    stage: str = ""             # "stage1" or "stage2"
    dataset_name: str = ""      # "prism", "a1", etc.

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> AffordanceSample:
        ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"stage={self.stage!r}, "
            f"dataset={self.dataset_name!r}, "
            f"len={len(self)})"
        )
