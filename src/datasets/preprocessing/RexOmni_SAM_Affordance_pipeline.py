#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RexOmni + SAM affordance pipeline (Step 3 of the annotation flow).

Given an image and a list of language instructions, runs:
    detection (category -> bbox) -> SAM (bbox -> mask)
    -> pointing (instruction -> point) -> mask-bounded Gaussian heatmap

Outputs (per image):
    1. detections JSON (boxes + points, grouped by category/instruction).
    2. (H, W, 1) float32 heatmap in [0, 1].
    3. Optional visualization.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


def filter_best_predictions(
    predictions: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Keep at most one box and one point per category.

    Rex-Omni does not return confidence scores; the first prediction per
    category is treated as the most reliable.

    Args:
        predictions: Raw predictions in the form
            {category: [{"type": "box"/"point", "coords": [...]}, ...]}.

    Returns:
        Filtered predictions, with at most one box and one point per category.
    """
    filtered = {}
    for category, detections in predictions.items():
        if not detections:
            continue

        boxes = [d for d in detections if d["type"] == "box"]
        points = [d for d in detections if d["type"] == "point"]

        best = []
        if boxes:
            best.append(boxes[0])
        if points:
            best.append(points[0])

        if best:
            filtered[category] = best

    return filtered


def _ensure_rex_omni_path() -> Path:
    """Make sure the bundled Rex-Omni package is on sys.path."""
    script_dir = Path(__file__).resolve().parent
    for path in (script_dir, *script_dir.parents):
        rex_root = path / "Rex-Omni"
        if rex_root.exists():
            if str(rex_root) not in sys.path:
                sys.path.insert(0, str(rex_root))
            return rex_root
    raise RuntimeError("Rex-Omni directory not found in the project tree.")


@dataclass
class AffordancePipelineConfig:
    """Pipeline configuration."""
    rex_model_path: str = "IDEA-Research/Rex-Omni"
    sam_checkpoint: Optional[str] = None
    sam_model_type: str = "vit_h"
    backend: str = "transformers"

    heatmap_sigma: Optional[float] = None
    heatmap_auto_sigma_ratio: float = 0.15
    heatmap_alpha: float = 0.6
    heatmap_colormap: str = "hot"

    cache_dir: Optional[str] = None


@dataclass
class AffordancePipelineOutput:
    """Pipeline output container."""
    predictions: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    masks: List[np.ndarray] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    boxes: List[np.ndarray] = field(default_factory=list)
    affordance_points: List[Tuple[float, float]] = field(default_factory=list)

    heatmap: Optional[np.ndarray] = None

    json_path: Optional[str] = None
    heatmap_path: Optional[str] = None
    visualization_path: Optional[str] = None


class AffordancePipeline:
    """Reusable RexOmni + SAM affordance pipeline.

    Example:
        pipeline = AffordancePipeline(config)
        output = pipeline.process(
            image="image.jpg",
            affordance_instructions=["Where to grasp the cup handle?"],
            detection_categories=["cup handle"],
            output_dir="outputs/",
        )
    """

    def __init__(
        self,
        config: Optional[AffordancePipelineConfig] = None,
        rex_model=None,
        sam_predictor=None,
    ):
        """Args:
            config: Pipeline configuration. None falls back to defaults.
            rex_model: Pre-initialized Rex-Omni wrapper (avoids reload).
            sam_predictor: Pre-initialized SAM predictor (avoids reload).
        """
        self.config = config or AffordancePipelineConfig()
        self._rex_model = rex_model
        self._sam_predictor = sam_predictor
        self._initialized = False

        _ensure_rex_omni_path()

        if self.config.cache_dir is not None:
            cache_dir = Path(self.config.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(cache_dir))
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))

    def _lazy_init(self):
        """Load Rex-Omni and SAM lazily on first use."""
        if self._initialized:
            return

        from applications._1_rexomni_sam.rexomni_sam_demo import (
            setup_rex_omni,
            setup_sam_model,
        )

        if self._rex_model is None:
            print("[Pipeline] Loading Rex-Omni ...")
            self._rex_model = setup_rex_omni(
                model_path=self.config.rex_model_path,
                backend=self.config.backend,
            )

        if self._sam_predictor is None:
            print("[Pipeline] Loading SAM ...")
            if not self.config.sam_checkpoint:
                raise ValueError(
                    "AffordancePipelineConfig.sam_checkpoint is empty; "
                    "set it to a valid SAM checkpoint path."
                )
            self._sam_predictor = setup_sam_model(
                sam_checkpoint=self.config.sam_checkpoint,
                model_type=self.config.sam_model_type,
            )

        self._initialized = True

    def process(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        affordance_instructions: List[str],
        detection_categories: Optional[List[str]] = None,
        original_instruction: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        output_prefix: Optional[str] = None,
        save_visualization: bool = True,
    ) -> AffordancePipelineOutput:
        """Process a single frame.

        Args:
            image: File path, PIL.Image, or HxWx3 RGB ``np.ndarray``.
            affordance_instructions: Where-to instructions for pointing.
            detection_categories: Optional category names for object detection.
            original_instruction: Optional source instruction (logged in JSON).
            output_dir: When set, JSON / heatmap / visualization are written here.
            output_prefix: Required when ``image`` is not a path.
            save_visualization: Whether to render and save a visualization.

        Returns:
            AffordancePipelineOutput holding predictions, masks, heatmap and paths.
        """
        self._lazy_init()

        from applications._1_rexomni_sam.rexomni_sam_demo import (
            detect_objects_with_rex,
            extract_affordance_points_from_predictions,
            extract_boxes_from_predictions,
            generate_affordance_points_with_rex,
            generate_all_heatmaps,
            generate_masks_with_sam,
            visualize_results_with_heatmap,
        )

        output = AffordancePipelineOutput()

        image_path_stem = None
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            image_path_stem = image_path.stem
            pil_image = Image.open(image_path).convert("RGB")
            img_array = np.array(pil_image)
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
            img_array = np.array(pil_image)
        elif isinstance(image, np.ndarray):
            img_array = image
            pil_image = Image.fromarray(img_array)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        image = pil_image
        h, w = img_array.shape[:2]

        if detection_categories:
            detection_result = detect_objects_with_rex(
                self._rex_model, image, detection_categories
            )
            output.predictions = filter_best_predictions(
                detection_result["extracted_predictions"]
            )
            output.boxes = extract_boxes_from_predictions(output.predictions)

            if len(output.boxes) > 0:
                output.masks, output.scores = generate_masks_with_sam(
                    self._sam_predictor, img_array, output.boxes
                )

        affordance_result = generate_affordance_points_with_rex(
            self._rex_model, image, affordance_instructions
        )

        filtered_affordance = filter_best_predictions(
            affordance_result["extracted_predictions"]
        )

        for category, detections in filtered_affordance.items():
            if category in output.predictions:
                output.predictions[category].extend(detections)
            else:
                output.predictions[category] = list(detections)

        output.affordance_points = extract_affordance_points_from_predictions(
            filtered_affordance
        )

        if len(output.affordance_points) > 0:
            if len(output.masks) > 0:
                heatmaps = generate_all_heatmaps(
                    masks=output.masks,
                    affordance_points=output.affordance_points,
                    boxes=output.boxes,
                    sigma=self.config.heatmap_sigma,
                    auto_sigma_ratio=self.config.heatmap_auto_sigma_ratio,
                )
                combined_heatmap = np.zeros((h, w), dtype=np.float32)
                for hm in heatmaps:
                    combined_heatmap = np.maximum(combined_heatmap, hm)
            else:
                combined_heatmap = self._generate_points_only_heatmap(
                    h, w, output.affordance_points
                )

            if combined_heatmap.max() > 0:
                combined_heatmap = combined_heatmap / combined_heatmap.max()
            output.heatmap = combined_heatmap[:, :, np.newaxis]
        else:
            output.heatmap = np.zeros((h, w, 1), dtype=np.float32)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            prefix = output_prefix or image_path_stem
            if prefix is None:
                raise ValueError(
                    "output_prefix is required when image is a PIL.Image or np.ndarray."
                )

            output.json_path = str(output_dir / f"{prefix}_detections.json")
            self._save_json(output, affordance_instructions, original_instruction, output.json_path)

            output.heatmap_path = str(output_dir / f"{prefix}_heatmap.npy")
            np.save(output.heatmap_path, output.heatmap)

            if save_visualization:
                output.visualization_path = str(output_dir / f"{prefix}_visualization.jpg")
                if len(output.masks) > 0 and len(output.affordance_points) > 0:
                    heatmaps = generate_all_heatmaps(
                        masks=output.masks,
                        affordance_points=output.affordance_points,
                        boxes=output.boxes,
                        sigma=self.config.heatmap_sigma,
                        auto_sigma_ratio=self.config.heatmap_auto_sigma_ratio,
                    )
                    visualize_results_with_heatmap(
                        image=image,
                        predictions=output.predictions,
                        masks=output.masks,
                        heatmaps=heatmaps,
                        save_path=output.visualization_path,
                        heatmap_alpha=self.config.heatmap_alpha,
                        heatmap_colormap=self.config.heatmap_colormap,
                    )
                else:
                    self._save_simple_visualization(
                        image, output, output.visualization_path
                    )

        return output

    def _generate_points_only_heatmap(
        self,
        h: int,
        w: int,
        points: List[Tuple[float, float]],
    ) -> np.ndarray:
        """Generate a heatmap from affordance points alone (no SAM mask)."""
        heatmap = np.zeros((h, w), dtype=np.float32)

        if len(points) == 0:
            return heatmap

        sigma = max(h, w) * self.config.heatmap_auto_sigma_ratio

        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        for point_x, point_y in points:
            gaussian = np.exp(
                -((x_coords - point_x) ** 2 + (y_coords - point_y) ** 2) / (2 * sigma ** 2)
            )
            heatmap += gaussian

        return heatmap

    def _save_json(
        self,
        output: AffordancePipelineOutput,
        affordance_instructions: List[str],
        original_instruction: Optional[str],
        json_path: str,
    ):
        """Persist detections (boxes + points) to JSON."""
        result = {
            "original_instruction": original_instruction,
            "affordance_instructions": affordance_instructions,
            "detections": {},
        }

        for category, detections in output.predictions.items():
            result["detections"][category] = {
                "points": [],
                "boxes": [],
            }
            for det in detections:
                if det["type"] == "point":
                    result["detections"][category]["points"].append({
                        "coords": list(det["coords"]),
                    })
                elif det["type"] == "box":
                    result["detections"][category]["boxes"].append({
                        "coords": list(det["coords"]),
                    })

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def _save_simple_visualization(
        self,
        image: Image.Image,
        output: AffordancePipelineOutput,
        save_path: str,
    ):
        """Visualization fallback used when no SAM mask is available."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)

        if output.heatmap is not None and output.heatmap.max() > 0:
            combined_heatmap = output.heatmap[:, :, 0]
            ax.imshow(
                combined_heatmap,
                cmap='jet',
                alpha=self.config.heatmap_alpha,
            )

        for x, y in output.affordance_points:
            ax.scatter(x, y, c='red', s=100, marker='*', edgecolors='white', linewidths=2)

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def process_images(
    images: List[Union[Image.Image, np.ndarray]],
    affordance_instructions: List[str],
    output_dir: Optional[Union[str, Path]] = None,
    detection_categories: Optional[List[str]] = None,
    config: Optional[AffordancePipelineConfig] = None,
    save_visualization: bool = True,
    output_prefix_template: str = "image_{:04d}",
) -> List[AffordancePipelineOutput]:
    """Process a list of in-memory images.

    Args:
        images: PIL or NumPy images.
        affordance_instructions: Where-to instructions for pointing.
        output_dir: Optional directory to write outputs.
        detection_categories: Optional categories for object detection.
        config: Pipeline configuration.
        save_visualization: Whether to save a per-image visualization.
        output_prefix_template: ``str.format`` template indexed by sample id.
    """
    pipeline = AffordancePipeline(config)
    outputs = []

    for i, image in enumerate(images):
        print(f"[Batch] {i+1}/{len(images)}")
        output = pipeline.process(
            image=image,
            affordance_instructions=affordance_instructions,
            detection_categories=detection_categories,
            output_dir=output_dir,
            output_prefix=output_prefix_template.format(i),
            save_visualization=save_visualization,
        )
        outputs.append(output)

    return outputs


def process_batch(
    input_dir: Union[str, Path],
    affordance_instructions: List[str],
    output_dir: Union[str, Path],
    detection_categories: Optional[List[str]] = None,
    config: Optional[AffordancePipelineConfig] = None,
    save_visualization: bool = True,
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> List[AffordancePipelineOutput]:
    """Process every image in a folder."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"input_dir is not a directory: {input_dir}")

    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ])

    if len(image_paths) == 0:
        print(f"[Batch] no images found under {input_dir}")
        return []

    print(f"[Batch] {len(image_paths)} images found under {input_dir}")

    pipeline = AffordancePipeline(config)
    outputs = []

    for i, image_path in enumerate(image_paths):
        print(f"[Batch] {i+1}/{len(image_paths)}: {image_path.name}")
        output = pipeline.process(
            image=image_path,
            affordance_instructions=affordance_instructions,
            detection_categories=detection_categories,
            output_dir=output_dir,
            save_visualization=save_visualization,
        )
        outputs.append(output)

    return outputs
