# Grounding and Pointing Data

Stage I uses 8.8M curated grounding and pointing samples from the source pools in Table 1 of the paper. Native spatial annotations are normalized into shared Which2Act and Where2Act targets, with aligned SAM-3D shape and layout supervision.

Dataset adapters follow the `AffordanceSample` interface in `src/datasets/base_dataset.py`.
