# Rex-Omni

This directory is based on the official implementation of [IDEA-Research/Rex-Omni](https://github.com/IDEA-Research/Rex-Omni) (*Detect Anything via Next Point Prediction*). Rex-Omni is a 3B-parameter multimodal large language model that unifies visual perception tasks—object detection, pointing, keypointing, OCR, and visual prompting—into a single "next point prediction" framework, producing structured spatial annotations (boxes, points, polygons, etc.).

## Usage

We directly reuse the official calling code (`RexOmniWrapper` for inference + `RexOmniVisualize` for visualization) without modifying the core inference logic.

> **Visualization note**: For more aesthetically pleasing visualizations, you may need to adjust the line thickness in the relevant scripts under the `applications` subfolder (i.e., the `draw_width` parameter of `RexOmniVisualize`, along with the `font_size`). This only affects rendering and does not change the detection results themselves.

## Role in the Annotation Framework

Within this project's data annotation framework, Rex-Omni is packaged as a standalone **service** and used as one of the automated data annotation tools. Running in **parallel** with it is **Qwen3-VL**, deployed locally on 32 H200 GPUs. The framework routes between the two based on predefined rules, invoking each according to task type and scenario so that they complement one another.
