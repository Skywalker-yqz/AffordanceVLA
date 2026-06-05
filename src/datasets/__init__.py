"""
AffordanceVLA dataset package.

Stage II datasets support unified action processing via three parameters:
  - CONTROL_FRAME:  resample to target frequency (FRAME_NUM Hz)
  - FRAME_NUM:      target frequency in Hz
  - CONTROL_SPEED:  compute unified 7D delta task-space actions + normalize

The unified sample schema (``AffordanceSample``) and reusable building blocks
live in ``base_dataset`` and are bundled here as the public data contract.

Dataset-specific loaders (PRISM / AGD20K / RefSpatial / VQA / A1 / LIBERO /
CALVIN) are intentionally NOT bundled: each source's on-disk format depends on
the release / split you download. Implement them against the ``AffordanceSample``
contract (see ``base_dataset``) — vibe coding is recommended (see the top-level
README). ``src/train.py`` imports them lazily and tolerates their absence.
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
