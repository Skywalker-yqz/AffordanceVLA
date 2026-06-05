"""
Unified action processing utilities for AffordanceVLA Stage II datasets.

Provides frame-rate resampling, delta task-space computation, and
normalization for A1, LIBERO, and CALVIN datasets.

Unified delta action format (7D):
    [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper_openness]
    - Δxyz:     position delta in meters
    - Δrxyz:    orientation delta as axis-angle (radians)
    - gripper:  openness in [0, 1]  (0 = closed, 1 = open)

Normalization (when CONTROL_SPEED=True):
    pos_norm  = clip(Δpos, -pos_limit, pos_limit) / pos_limit   → [-1, 1]
    ori_norm  = clip(Δori, -ori_limit, ori_limit) / ori_limit   → [-1, 1]
    grip      = gripper_openness                                  → [0, 1]

`pos_limit` and `ori_limit` are not hard-coded here. They must be supplied
by the YAML config and propagated by each dataset.

Gripper unification (→ 0 = closed, 1 = open):
    A1:     already [0, 1]                      → use directly
    LIBERO: {+1 = close, -1 = open}             → (-cmd + 1) / 2
    CALVIN: {-1 = close, +1 = open}             → (cmd + 1) / 2

Frame-rate resampling:
    A1:     30 Hz → FRAME_NUM Hz  (step = 30 / FRAME_NUM)
    LIBERO: 20 Hz → FRAME_NUM Hz  (step = 20 / FRAME_NUM)
    CALVIN: 30 Hz → FRAME_NUM Hz  (step = 30 / FRAME_NUM)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

POS_LIMIT: Optional[float] = None
ORI_LIMIT: Optional[float] = None
PANDA_GRIPPER_MAX_WIDTH: float = 0.08  # meters (Panda gripper hardware spec)

DATASET_FREQ = {
    "a1": 30,
    "libero": 20,
    "calvin": 30,
}

# Action dimension when using unified delta format
UNIFIED_ACTION_DIM: int = 7


# ═══════════════════════════════════════════════════════════════════════
# Frame-Rate Resampling
# ═══════════════════════════════════════════════════════════════════════


def get_resampled_indices(
    orig_freq: int,
    target_freq: int,
    num_frames: int,
) -> np.ndarray:
    """Compute which original frame indices to keep after resampling.

    Uses uniform subsampling: picks the nearest original frame for each
    target time step.

    Example:
        30 Hz → 10 Hz, 300 frames → indices [0, 3, 6, 9, …, 297]
        20 Hz → 10 Hz, 100 frames → indices [0, 2, 4, 6, …, 98]

    Args:
        orig_freq:   Original frame rate in Hz (e.g. 30, 20).
        target_freq: Target frame rate in Hz (e.g. 10).
        num_frames:  Total number of frames in the original sequence.

    Returns:
        1-D int array of selected original frame indices.
    """
    if target_freq >= orig_freq:
        # No subsampling needed (or would require interpolation — skip)
        return np.arange(num_frames)

    duration = (num_frames - 1) / orig_freq  # seconds
    target_times = np.arange(0, duration + 1e-9, 1.0 / target_freq)
    # Map each target time to the nearest original frame index
    indices = np.round(target_times * orig_freq).astype(int)
    indices = np.clip(indices, 0, num_frames - 1)
    # Remove duplicates while preserving order
    _, unique_idx = np.unique(indices, return_index=True)
    indices = indices[np.sort(unique_idx)]
    return indices


# ═══════════════════════════════════════════════════════════════════════
# Orientation Helpers
# ═══════════════════════════════════════════════════════════════════════


def euler_to_quat_xyzw(euler_angles: np.ndarray) -> np.ndarray:
    """Convert euler angles (xyz convention) to quaternion (x,y,z,w).

    Args:
        euler_angles: (..., 3) array of euler angles in radians.

    Returns:
        (..., 4) quaternion array in scipy (x,y,z,w) convention.
    """
    return Rotation.from_euler("xyz", euler_angles).as_quat()


def axisangle_to_quat_xyzw(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle to quaternion (x,y,z,w).

    Args:
        axis_angle: (..., 3) axis-angle vector (direction = axis, norm = angle).

    Returns:
        (..., 4) quaternion in scipy (x,y,z,w) convention.
    """
    return Rotation.from_rotvec(axis_angle).as_quat()


def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion from (w,x,y,z) to scipy (x,y,z,w) convention.

    Args:
        q_wxyz: (..., 4) quaternion array in (w, x, y, z) order.

    Returns:
        (..., 4) quaternion array in (x, y, z, w) order.
    """
    return q_wxyz[..., [1, 2, 3, 0]]


# ═══════════════════════════════════════════════════════════════════════
# Delta Action Computation
# ═══════════════════════════════════════════════════════════════════════


def compute_delta_actions(
    positions: np.ndarray,
    quat_xyzw: np.ndarray,
    gripper_openness: np.ndarray,
) -> np.ndarray:
    """Compute unified 7-D delta actions from a trajectory.

    For N frames, produces (N-1) delta actions:
        delta[t] describes the transition from frame t to frame t+1.

    Delta orientation is computed via quaternion:
        R_delta = R[t]^{-1} * R[t+1]  →  axis-angle  (avoids RPY wrapping!)

    Args:
        positions:        (N, 3) float64 — end-effector xyz in meters.
        quat_xyzw:        (N, 4) float64 — orientation quaternion (scipy convention).
        gripper_openness:  (N,)  float64 — [0, 1], 0 = closed, 1 = open.

    Returns:
        (N-1, 7) float64 array:
            [:, 0:3] = Δposition  (meters)
            [:, 3:6] = Δorientation axis-angle  (radians)
            [:, 6]   = gripper openness of the *next* frame
    """
    N = len(positions)
    assert len(quat_xyzw) == N and len(gripper_openness) == N

    # Position delta — straightforward subtraction
    delta_pos = np.diff(positions, axis=0)  # (N-1, 3)

    # Orientation delta — via quaternion to avoid Euler/RPY wrapping
    rots = Rotation.from_quat(quat_xyzw)
    delta_rotvec = np.zeros((N - 1, 3), dtype=np.float64)
    for i in range(N - 1):
        delta_rot = rots[i].inv() * rots[i + 1]
        delta_rotvec[i] = delta_rot.as_rotvec()

    # Gripper — take the value at the next frame (target state)
    grip_next = gripper_openness[1:]  # (N-1,)

    return np.column_stack([delta_pos, delta_rotvec, grip_next])  # (N-1, 7)


# ═══════════════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════════════


def _require_limit(name: str, value: Optional[float]) -> float:
    if value is None:
        raise ValueError(
            f"{name} is not set. Pass it explicitly (e.g. via YAML) — "
            f"`{name}` has no default in action_utils."
        )
    return float(value)


def normalize_delta_actions(
    delta_actions: np.ndarray,
    pos_limit: Optional[float] = None,
    ori_limit: Optional[float] = None,
) -> np.ndarray:
    """Normalize delta actions to [-1, 1] (position/orientation) and [0, 1] (gripper).

    Args:
        delta_actions: (..., 7) array — [Δpos(3), Δori(3), gripper(1)].
        pos_limit: Clip & scale limit for position (meters). Required.
        ori_limit: Clip & scale limit for orientation (radians). Required.

    Returns:
        (..., 7) array with pos/ori in [-1, 1], gripper in [0, 1].
    """
    pos_limit = _require_limit("pos_limit", pos_limit)
    ori_limit = _require_limit("ori_limit", ori_limit)

    out = delta_actions.copy()
    out[..., 0:3] = np.clip(out[..., 0:3], -pos_limit, pos_limit) / pos_limit
    out[..., 3:6] = np.clip(out[..., 3:6], -ori_limit, ori_limit) / ori_limit
    out[..., 6] = np.clip(out[..., 6], 0.0, 1.0)
    return out


def denormalize_delta_actions(
    norm_actions: np.ndarray,
    pos_limit: Optional[float] = None,
    ori_limit: Optional[float] = None,
) -> np.ndarray:
    """Reverse normalization: [-1, 1] → physical units.

    Args:
        norm_actions: (..., 7) normalized actions.
        pos_limit: position limit used during normalization (meters). Required.
        ori_limit: orientation limit used during normalization (radians). Required.

    Returns:
        (..., 7) delta actions in physical units (meters, radians).
    """
    pos_limit = _require_limit("pos_limit", pos_limit)
    ori_limit = _require_limit("ori_limit", ori_limit)

    out = norm_actions.copy()
    out[..., 0:3] *= pos_limit
    out[..., 3:6] *= ori_limit
    return out


# ═══════════════════════════════════════════════════════════════════════
# High-Level: Extract an Action Chunk
# ═══════════════════════════════════════════════════════════════════════


def extract_delta_action_chunk(
    positions: np.ndarray,
    quat_xyzw: np.ndarray,
    gripper_openness: np.ndarray,
    start_idx: int,
    chunk_size: int,
    normalize: bool = True,
    pos_limit: Optional[float] = None,
    ori_limit: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a delta action chunk for training.

    Given a trajectory (all frames of an episode, possibly resampled),
    extract a chunk of `chunk_size` delta actions starting at `start_idx`.
    Pad with zeros (and mark padding) if fewer than chunk_size steps remain.

    Args:
        positions:         (N, 3) end-effector positions (meters).
        quat_xyzw:         (N, 4) orientation quaternions (scipy x,y,z,w).
        gripper_openness:  (N,) gripper openness [0, 1].
        start_idx:         Frame index to start the chunk.
        chunk_size:        Number of delta steps to return.
        normalize:         Whether to normalize to [-1, 1].
        pos_limit:         Clip & scale limit for position (meters).
        ori_limit:         Clip & scale limit for orientation (radians).

    Returns:
        (action_chunk, action_is_pad):
            action_chunk:  (chunk_size, 7) float32 tensor.
            action_is_pad: (chunk_size,)   bool tensor (True = padded).
    """
    N = len(positions)

    # We need frames [start_idx, start_idx + chunk_size] to compute
    # chunk_size deltas (each delta needs frame t and t+1).
    end_idx = min(start_idx + chunk_size + 1, N)
    segment_pos = positions[start_idx:end_idx]
    segment_quat = quat_xyzw[start_idx:end_idx]
    segment_grip = gripper_openness[start_idx:end_idx]

    # Compute deltas for the available segment
    if len(segment_pos) >= 2:
        deltas = compute_delta_actions(segment_pos, segment_quat, segment_grip)
        if normalize:
            deltas = normalize_delta_actions(deltas, pos_limit=pos_limit, ori_limit=ori_limit)
    else:
        deltas = np.zeros((0, UNIFIED_ACTION_DIM), dtype=np.float64)

    num_valid = len(deltas)

    # Pad to chunk_size
    action_chunk = np.zeros((chunk_size, UNIFIED_ACTION_DIM), dtype=np.float32)
    action_chunk[:num_valid] = deltas[:chunk_size].astype(np.float32)

    # Pad remaining with last valid action (or zeros)
    if num_valid > 0 and num_valid < chunk_size:
        action_chunk[num_valid:] = action_chunk[num_valid - 1]

    # Padding mask
    action_is_pad = np.zeros(chunk_size, dtype=bool)
    action_is_pad[num_valid:] = True

    return (
        torch.from_numpy(action_chunk),
        torch.from_numpy(action_is_pad),
    )


# ═══════════════════════════════════════════════════════════════════════
# Gripper Unification Helpers
# ═══════════════════════════════════════════════════════════════════════


def unify_gripper_a1(gripper_position: np.ndarray) -> np.ndarray:
    """A1 gripper: already [0, 1] with 0=closed, 1=open."""
    return np.clip(gripper_position.astype(np.float64), 0.0, 1.0)


def unify_gripper_libero(
    gripper_states: np.ndarray,
) -> np.ndarray:
    """LIBERO gripper: two finger positions (N, 2) in meters → [0, 1].

    gripper_width = left_finger - right_finger, range [0, 0.08] m.
    """
    gripper_width = gripper_states[:, 0] - gripper_states[:, 1]
    openness = gripper_width / PANDA_GRIPPER_MAX_WIDTH
    return np.clip(openness.astype(np.float64), 0.0, 1.0)


def unify_gripper_libero_action(gripper_cmd: np.ndarray) -> np.ndarray:
    """LIBERO gripper action command: +1=close, -1=open → [0, 1].

    Maps: +1 → 0 (closed), -1 → 1 (open).
    """
    return np.clip((-gripper_cmd.astype(np.float64) + 1.0) / 2.0, 0.0, 1.0)


def unify_gripper_calvin(
    gripper_width: np.ndarray,
) -> np.ndarray:
    """CALVIN gripper: width in meters [0, 0.08] → [0, 1]."""
    openness = gripper_width / PANDA_GRIPPER_MAX_WIDTH
    return np.clip(openness.astype(np.float64), 0.0, 1.0)


def unify_gripper_calvin_action(gripper_cmd: np.ndarray) -> np.ndarray:
    """CALVIN gripper action command: +1=open, -1=close → [0, 1].

    Maps: +1 → 1 (open), -1 → 0 (closed).
    """
    return np.clip((gripper_cmd.astype(np.float64) + 1.0) / 2.0, 0.0, 1.0)
