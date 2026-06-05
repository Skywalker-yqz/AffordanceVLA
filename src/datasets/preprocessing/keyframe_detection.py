#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rule-based keyframe detection from robot state (joint positions, gripper).

Detects six complementary keyframe types over a trajectory:
  - Start:       first frame (always enabled)
  - Pre-Action:  N frames before each gripper state change
  - Gripper:     frame where the gripper open/close state flips
  - Stop:        frames where joint velocities fall below a threshold
  - Apex:        local minima of joint velocity norm (turning points)
  - End:         M frames before the trajectory end
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# JSON encoder
# ============================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that serializes NumPy scalars and arrays."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================
# Data structures
# ============================================================

@dataclass
class KeyframeInfo:
    """One keyframe entry."""
    frame_idx: int
    reason: str


@dataclass
class EpisodeKeyframeResult:
    """Keyframe detection result for one trajectory (episode)."""
    episode_id: str
    task_name: str
    instruction: str
    total_frames: int
    frame_range: Tuple[int, int]
    keyframes: List[Dict[str, Any]]  # List of {"frame_idx": int, "reason": str}
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_keyframes(self) -> int:
        return len(self.keyframes)
    
    @property
    def keyframe_indices(self) -> List[int]:
        return [kf["frame_idx"] for kf in self.keyframes]
    
    @property
    def keyframe_reasons(self) -> List[str]:
        return [kf["reason"] for kf in self.keyframes]


# ============================================================
# Detector
# ============================================================

class KeyframeDetector:
    """Rule-based keyframe detector over a robot trajectory.

    Detection types (all enabled by default):
      - Start:       first frame
      - Pre-Action:  N frames before each gripper state change
      - Gripper:     frame where the gripper state flips
      - Stop:        joint velocities below threshold (mid-trajectory pauses)
      - Apex:        local minima of joint velocity norm (turning points)
      - End:         M frames before the trajectory end
    """
    
    def __init__(
        self,
        gripper_threshold: float = 0.5,
        velocity_threshold: float = 0.01,
        stopped_buffer_size: int = 4,
        pre_action_offset: int = 30,
        end_frame_offset: int = 25,
        merge_threshold: int = 10,
        enable_pre_action: bool = True,
        enable_gripper: bool = True,
        enable_stop: bool = True,
        enable_apex: bool = True,
        enable_end: bool = True,
    ):
        """Initialize the detector.

        Args:
            gripper_threshold:    Open/close threshold on gripper openness in [0, 1].
            velocity_threshold:   Joint velocity magnitude treated as ~zero.
            stopped_buffer_size:  Frames waited after a Stop event before a new one.
            pre_action_offset:    N — Pre-Action looks N frames before each gripper flip.
            end_frame_offset:     M — End is placed M frames before the last frame.
            merge_threshold:      Keyframes closer than this distance are merged.
            enable_pre_action / gripper / stop / apex / end:
                Toggle individual detection rules. All default to True.
        """
        self.gripper_threshold = gripper_threshold
        self.velocity_threshold = velocity_threshold
        self.stopped_buffer_size = stopped_buffer_size
        self.pre_action_offset = pre_action_offset
        self.end_frame_offset = end_frame_offset
        self.merge_threshold = merge_threshold
        self.enable_pre_action = enable_pre_action
        self.enable_gripper = enable_gripper
        self.enable_stop = enable_stop
        self.enable_apex = enable_apex
        self.enable_end = enable_end
    
    def get_config(self) -> Dict[str, Any]:
        """Return the detector configuration as a dict."""
        return {
            "gripper_threshold": self.gripper_threshold,
            "velocity_threshold": self.velocity_threshold,
            "stopped_buffer_size": self.stopped_buffer_size,
            "pre_action_offset": self.pre_action_offset,
            "end_frame_offset": self.end_frame_offset,
            "merge_threshold": self.merge_threshold,
            "enable_pre_action": self.enable_pre_action,
            "enable_gripper": self.enable_gripper,
            "enable_stop": self.enable_stop,
            "enable_apex": self.enable_apex,
            "enable_end": self.enable_end,
        }
    
    def _compute_joint_velocity(
        self,
        joint_positions: np.ndarray,
        dt: float = 1.0 / 30.0,
    ) -> np.ndarray:
        """Compute joint velocities from positions via finite differences.

        Args:
            joint_positions: (N, J) joint position trajectory.
            dt: Time step (seconds per frame).

        Returns:
            (N, J) joint velocity sequence.
        """
        velocities = np.zeros_like(joint_positions)
        if len(joint_positions) < 2:
            return velocities

        if len(joint_positions) > 2:
            velocities[1:-1] = (joint_positions[2:] - joint_positions[:-2]) / (2 * dt)

        velocities[0] = (joint_positions[1] - joint_positions[0]) / dt
        velocities[-1] = (joint_positions[-1] - joint_positions[-2]) / dt

        return velocities

    def _is_gripper_open(self, gripper_openness: float) -> bool:
        """Return True if the gripper is open (above threshold)."""
        return gripper_openness > self.gripper_threshold

    def _is_stopped(
        self,
        i: int,
        joint_velocities: np.ndarray,
        gripper_states: np.ndarray,
        stopped_buffer: int,
        total_frames: int,
    ) -> bool:
        """Check whether the robot is stopped at frame i.

        Conditions:
          1. Stop buffer has elapsed.
          2. Joint velocities are within velocity_threshold of zero.
          3. Not in the last two frames.
          4. Gripper state is consistent across neighbors.
        """
        next_is_not_final = i < (total_frames - 2)

        gripper_state_no_change = (
            i < (total_frames - 2) and i >= 2 and
            gripper_states[i] == gripper_states[i + 1] and
            gripper_states[i] == gripper_states[max(0, i - 1)] and
            gripper_states[max(0, i - 2)] == gripper_states[max(0, i - 1)]
        )

        small_delta = np.allclose(joint_velocities[i], 0, atol=self.velocity_threshold)

        stopped = (
            stopped_buffer <= 0 and
            small_delta and
            next_is_not_final and
            gripper_state_no_change
        )

        return stopped

    def _is_apex_or_turn(
        self,
        i: int,
        joint_velocities: np.ndarray,
        total_frames: int,
    ) -> bool:
        """Heuristic: detect trajectory apex / turning point.

        Triggers when the joint-velocity norm is a local minimum that is
        slow but not fully static (avoiding overlap with Stop).
        """
        if i < 2 or i > total_frames - 3:
            return False

        vel_norm = np.linalg.norm(joint_velocities, axis=1)
        curr_vel = vel_norm[i]

        is_slow_but_moving = self.velocity_threshold < curr_vel < (self.velocity_threshold * 5)
        is_local_min = curr_vel < vel_norm[i - 1] and curr_vel < vel_norm[i + 1]

        return is_slow_but_moving and is_local_min
    
    def detect(
        self,
        joint_positions: np.ndarray,
        gripper_openness: np.ndarray,
        fps: float = 30.0,
        gripper_action: Optional[np.ndarray] = None,
    ) -> Tuple[List[int], List[str]]:
        """Detect keyframes from a robot trajectory.

        Args:
            joint_positions:  (N, J) joint position sequence.
            gripper_openness: (N,) gripper openness in [0, 1].
            fps:              Frame rate (Hz).
            gripper_action:   (N,) optional gripper command, +1=open / -1=close.
                              When given, gripper transitions are computed from the
                              command; otherwise from `gripper_openness` + threshold.

        Returns:
            (keyframe_indices, keyframe_reasons): two lists of equal length.
        """
        total_frames = len(gripper_openness)

        if total_frames == 0:
            raise ValueError(
                "Trajectory has 0 frames; cannot run keyframe detection."
            )

        if total_frames == 1:
            raise ValueError(
                f"Trajectory has only {total_frames} frame; cannot run keyframe detection."
            )

        joint_velocities = self._compute_joint_velocity(joint_positions, dt=1.0 / fps)

        use_action_based = gripper_action is not None
        if use_action_based:
            gripper_states = gripper_action > 0
        else:
            gripper_states = np.array([self._is_gripper_open(g) for g in gripper_openness])

        keyframe_map: Dict[int, List[str]] = {}
        keyframe_map[0] = ["Start"]
        
        prev_gripper_open = gripper_states[0]
        stopped_buffer = 0
        
        for i in range(total_frames):
            stopped = self._is_stopped(
                i, joint_velocities, gripper_states, stopped_buffer, total_frames
            )
            stopped_buffer = self.stopped_buffer_size if stopped else stopped_buffer - 1
            
            last = i == (total_frames - 1)
            gripper_changed = gripper_states[i] != prev_gripper_open

            if gripper_changed and self.enable_pre_action:
                pre_idx = max(0, i - self.pre_action_offset)
                if pre_idx not in keyframe_map:
                    keyframe_map[pre_idx] = []
                keyframe_map[pre_idx].append("Pre-Action")

            if self.enable_gripper and i != 0 and gripper_changed:
                if i not in keyframe_map:
                    keyframe_map[i] = []
                keyframe_map[i].append("Gripper")

            if self.enable_stop and i != 0 and stopped:
                if stopped_buffer == self.stopped_buffer_size:
                    if i not in keyframe_map:
                        keyframe_map[i] = []
                    keyframe_map[i].append("Stop")

            if self.enable_apex and self._is_apex_or_turn(i, joint_velocities, total_frames):
                has_neighbor = any(abs(i - k) < 5 for k in keyframe_map.keys())
                if not has_neighbor:
                    if i not in keyframe_map:
                        keyframe_map[i] = []
                    keyframe_map[i].append("Apex")

            if self.enable_end and last:
                end_idx = max(0, total_frames - 1 - self.end_frame_offset)
                if end_idx > 0:
                    if end_idx not in keyframe_map:
                        keyframe_map[end_idx] = []
                    keyframe_map[end_idx].append("End")

            prev_gripper_open = gripper_states[i]

        sorted_indices = sorted(keyframe_map.keys())

        final_indices: List[int] = []
        final_reasons: List[str] = []

        if sorted_indices:
            curr_idx = sorted_indices[0]
            curr_reasons = keyframe_map[curr_idx].copy()

            for next_idx in sorted_indices[1:]:
                next_reasons = keyframe_map[next_idx]

                if next_idx - curr_idx < self.merge_threshold:
                    curr_reasons.extend(next_reasons)
                    curr_idx = next_idx
                else:
                    final_indices.append(curr_idx)
                    reason_str = " | ".join(sorted(list(set(curr_reasons))))
                    final_reasons.append(reason_str)

                    curr_idx = next_idx
                    curr_reasons = next_reasons.copy()

            final_indices.append(curr_idx)
            reason_str = " | ".join(sorted(list(set(curr_reasons))))
            final_reasons.append(reason_str)

        return final_indices, final_reasons


class KeyframeResultManager:
    """Persist and load keyframe-detection results.

    Each trajectory is written to one JSON under ``keyframe_results/{dataset}/``,
    mirroring the source dataset's directory layout. A global
    ``keyframe_index.json`` is written alongside.
    """

    def __init__(self, output_dir: Path):
        """Args:
            output_dir: Root output directory.
        """
        self.output_dir = Path(output_dir)
        self.keyframe_dir = self.output_dir / "keyframe_results"

    def save_detection_results(
        self,
        dataset: str,
        results: List[EpisodeKeyframeResult],
        config: Dict[str, Any],
    ) -> Path:
        """Save per-episode keyframe results and a global index file.

        Args:
            dataset: Dataset name (used as the top-level subdirectory).
            results: Per-episode keyframe results.
            config:  Detector configuration recorded in the index.

        Returns:
            Path to the index JSON.
        """
        self.keyframe_dir.mkdir(parents=True, exist_ok=True)

        dataset_dir = self.keyframe_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        episode_entries = []
        total_keyframes = 0

        for result in results:
            relative_path = f"{result.episode_id}.json"
            filepath = dataset_dir / relative_path
            filepath.parent.mkdir(parents=True, exist_ok=True)

            result_dict = {
                "episode_id": result.episode_id,
                "task_name": result.task_name,
                "instruction": result.instruction,
                "total_frames": result.total_frames,
                "frame_range": list(result.frame_range),
                "keyframes": result.keyframes,
                "extra": result.extra,
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

            episode_entries.append({
                "episode_id": result.episode_id,
                "keyframe_file": f"{dataset}/{relative_path}",
                "num_keyframes": result.num_keyframes,
                "total_frames": result.total_frames,
            })
            total_keyframes += result.num_keyframes

        index = {
            "dataset": dataset,
            "created_at": datetime.now().isoformat(),
            "detection_config": config,
            "total_episodes": len(results),
            "total_keyframes": total_keyframes,
            "episodes": episode_entries,
        }
        
        index_path = self.keyframe_dir / "keyframe_index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        return index_path
    
    def load_keyframe_index(self) -> Optional[Dict[str, Any]]:
        """Return the global keyframe index, or None if it does not exist."""
        index_path = self.keyframe_dir / "keyframe_index.json"
        if not index_path.exists():
            return None
        
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def load_episode_keyframes(self, episode_id: str) -> Optional[EpisodeKeyframeResult]:
        """Load keyframe results for one episode by id; returns None if missing."""
        index = self.load_keyframe_index()
        if not index:
            return None
        
        for entry in index["episodes"]:
            if entry["episode_id"] == episode_id:
                filepath = self.keyframe_dir / entry["keyframe_file"]
                if not filepath.exists():
                    return None
                
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                return EpisodeKeyframeResult(
                    episode_id=data["episode_id"],
                    task_name=data["task_name"],
                    instruction=data["instruction"],
                    total_frames=data["total_frames"],
                    frame_range=tuple(data["frame_range"]),
                    keyframes=data["keyframes"],
                    extra=data.get("extra", {}),
                )
        
        return None
    
    def get_all_keyframe_items(self) -> List[Dict[str, Any]]:
        """Flatten all stored keyframes into a list of work items.

        Each item carries: episode_id, task_name, instruction, frame_idx,
        reason, and any dataset-specific ``extra`` fields.
        """
        index = self.load_keyframe_index()
        if not index:
            return []
        
        items = []
        for entry in index["episodes"]:
            filepath = self.keyframe_dir / entry["keyframe_file"]
            if not filepath.exists():
                continue
            
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for kf in data["keyframes"]:
                items.append({
                    "episode_id": data["episode_id"],
                    "task_name": data["task_name"],
                    "instruction": data["instruction"],
                    "frame_idx": kf["frame_idx"],
                    "reason": kf["reason"],
                    "total_frames": data["total_frames"],
                    "extra": data.get("extra", {}),
                })
        
        return items
    
    def exists(self) -> bool:
        """Return True if the keyframe index has been written."""
        return (self.keyframe_dir / "keyframe_index.json").exists()

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics from the saved index."""
        index = self.load_keyframe_index()
        if not index:
            return {"exists": False}
        
        return {
            "exists": True,
            "dataset": index["dataset"],
            "created_at": index["created_at"],
            "total_episodes": index["total_episodes"],
            "total_keyframes": index["total_keyframes"],
            "detection_config": index["detection_config"],
        }
