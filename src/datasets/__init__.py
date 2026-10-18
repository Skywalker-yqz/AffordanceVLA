"""
AffordanceVLA dataset package.

Stage II datasets support unified action processing via three parameters:
  - CONTROL_FRAME:  resample to target frequency (FRAME_NUM Hz)
  - FRAME_NUM:      target frequency in Hz
  - CONTROL_SPEED:  compute unified 7D delta task-space actions + normalize

The unified sample schema (``AffordanceSample``) and reusable building blocks
live in ``base_dataset`` and are bundled here as the public data contract.

Dataset adapters follow the ``AffordanceSample`` contract and can be selected
through ``data.datasets[].factory`` and ``kwargs`` in the training configuration.
"""

from .base_dataset import (
    BaseAffordanceDataset,
    AffordanceSample,
    LayoutToken,
    image_to_tensor,
    heatmap_to_tensor,
    resize_sample,
    build_stage1_sample,
    build_stage2_sample,
)
from .action_utils import (
    POS_LIMIT,
    ORI_LIMIT,
    UNIFIED_ACTION_DIM,
    DATASET_FREQ,
    compute_delta_actions,
    normalize_delta_actions,
    denormalize_delta_actions,
)
from .collate import affordance_collate_fn
from .multi_loader import MultiDatasetLoader
