# Third-party notices

Original AffordanceVLA contributions are licensed under the [MIT License](LICENSE).
Third-party portions retain their own licenses, including portions adapted into
local Python files. The MIT license does not replace the licenses of those
portions, separately installed software, pretrained weights, tokenizers, datasets,
or simulator assets. No pretrained weights or training datasets are bundled here.

This inventory was checked against upstream license files and model cards on
2026-10-18. Commit links identify versions used to verify notices; unless an
environment file pins that commit, they are **not a claim that local adaptations
were originally made from that exact revision**. Unmodified license copies and
their source URLs/SHA-256 hashes are recorded in
[`licenses/UPSTREAM_LICENSE_SOURCES.json`](licenses/UPSTREAM_LICENSE_SOURCES.json).

## Adapted source code

| Upstream | Local scope | License and attribution |
| --- | --- | --- |
| [Hugging Face LeRobot](https://github.com/huggingface/lerobot/tree/b883328e6c95681ca90a18b102e4ae5e1f91e2bf/src/lerobot/policies/pi0) | PaliGemma expert/RoPE in `src/models/paligemma_with_expert.py`; pi0 flow-matching portions in `src/models/modeling_affordance.py`; policy loading/constants in `src/policies/affordance_policy.py`; attention/padding/image helpers in `src/utils/` | [Apache-2.0](licenses/LeRobot-Apache-2.0.txt). Copyright 2024 The HuggingFace Inc. team. All rights reserved. Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved. |
| [Physical Intelligence openpi](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479) | pi0 foundation referenced by the backbone; `convert_to_uint8` / `_resize_with_pad_pil` in `src/utils/image_tools.py` correspond to `packages/openpi-client/src/openpi_client/image_tools.py` | [Apache-2.0 source code](licenses/openpi-Apache-2.0.txt). Downloaded pi0/PaliGemma weights have separate terms below. |
| [FoundationVision/VAR](https://github.com/FoundationVision/VAR/tree/78b95394fc5896192e3a003e4b295f8ea743c48f) | `src/models/vqvae/{vqvae,quant,basic_vae}.py` and `sample_with_top_k_top_p_` in `src/utils/model_utils.py` | [MIT](licenses/VAR-MIT.txt). Copyright (c) 2024 FoundationVision. |
| [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers/tree/3ba01b241669f5ade541ce990f7650a3b8f65318) | Quantizer references/material in `src/models/vqvae/quant.py` | [MIT](licenses/taming-transformers-MIT.txt). Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. |
| [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion/tree/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229) | Autoencoder references/material in `src/models/vqvae/basic_vae.py` and `vqvae.py` | [CreativeML Open RAIL-M](licenses/stable-diffusion-OpenRAIL-M.txt). Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors. |
| [IDEA Research Rex-Omni](https://github.com/IDEA-Research/Rex-Omni/tree/ee58b1e887f94da49d0749945388defede6b52e6) | Box extraction / SAM mask helper portions in `src/services/rexomni_service.py` correspond to `applications/_1_rexomni_sam/rexomni_sam_demo.py`; preprocessing also imports that demo from the separately installed project | [IDEA License 1.0](licenses/Rex-Omni-IDEA-1.0.txt), **non-commercial research purposes**. Copyright (c) IDEA. All Rights Reserved. |
| [Meta SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects/tree/f91db411c50efee93d8db7aeb323885650f6f722) | Pipeline portions in `src/services/sam3d_service.py`, especially `_run_latent_only_pipeline` and `_run_stage1_pipeline`, adapt sparse-structure generation and pointmap/layout processing | [SAM License](licenses/SAM3D-SAM-License.txt), a custom license covering code and checkpoints. Copyright (c) Meta Platforms, Inc. and affiliates. These adapted portions and their modifications remain under the SAM License. |

These implementations have been modified for AffordanceVLA (the three-expert
backbone, affordance losses, service modes, input/output handling, and helper
interfaces). Source comments identify the applicable upstream notices. Copyright
statements and complete license terms are retained in `licenses/`.

The Stable Diffusion license distinguishes source code ("Complementary Material")
from models and model derivatives. Section III's model-use restrictions are not
automatically restrictions on unrelated AffordanceVLA code; the grants and
conditions applicable to the cited upstream portions still apply. The VAR model
card is separately identified below rather than assigning a license to its
weights solely from the source-code license.

Rex-Omni's upstream README attributes it as licensed under the IDEA License 1.0,
Copyright (c) IDEA. All Rights Reserved. The literal attribution required by
section 2(b) of its pinned license is retained without correcting its wording:

> T  is licensed under the IDEA License 1.0, Copyright (c) IDEA. All Rights Reserved.

AffordanceVLA's annotation workflow uses SAM 3D Objects by Meta. Its SAM License
requires acknowledging the use of SAM Materials when publishing research using
them. Its terms differ from the Apache-2.0 license of the original 2D SAM.

## Separately installed software and benchmarks

The placeholder directories `LIBERO/`, `calvin/`, `Rex-Omni/`, and `SAM3D/` contain
setup references, not full copies of those projects. Install the actual releases
with their upstream license/notice files. Their third-party assets and
subcomponents may have additional terms.

| Project | Use in this repository | Source-code terms |
| --- | --- | --- |
| [Rex-Omni](https://github.com/IDEA-Research/Rex-Omni/tree/ee58b1e887f94da49d0749945388defede6b52e6) | Detection/pointing; pinned in `env/requirements_rexomni.txt` | [IDEA License 1.0](licenses/Rex-Omni-IDEA-1.0.txt), non-commercial research. Its Qwen-derived weights also have the conditions below. |
| [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL/tree/96588727e44c78b25ba03ea03b8e12f7e64fd0da) | Optional annotation backend | [Apache-2.0 repository code](licenses/Qwen3-VL-Apache-2.0.txt); consult the selected checkpoint's own model card/license for weights. |
| [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything/tree/dca509fe793f601edb92606367a655c15ac00fdf) | 2D segmentation; pinned environment dependency | [Apache-2.0](licenses/SAM-Apache-2.0.txt). The [upstream README](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/README.md#license) applies it to original SAM code/model; SA-1B data has a separate research license. |
| [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects/tree/f91db411c50efee93d8db7aeb323885650f6f722) | 3D shape/layout supervision | [SAM License](licenses/SAM3D-SAM-License.txt) for both code and checkpoints. Retain its third-party notices too. |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01) | Target benchmark/environment; pinned LIBERO snapshots | [MIT](licenses/LIBERO-MIT.txt), Copyright (c) 2023 Lifelong Robot Learning. |
| [CALVIN](https://github.com/mees/calvin/tree/fa03f01f19c65920e18cf37398a9ce859274af76) / [calvin_env](https://github.com/mees/calvin_env/tree/1431a46bd36bde5903fb6345e68b5ccc30def666) | ABC→D benchmark/environment; pinned CALVIN snapshots | [MIT](licenses/CALVIN-MIT.txt), Copyright (c) 2021 Oier Mees. Both pinned root licenses have identical contents; this does not relicense assets or dependent packages. |
| [OpenVLA](https://github.com/openvla/openvla/tree/c8f03f48af692657d3060c19588038c7220e9af9) | Referenced LIBERO evaluation conventions | [MIT code](licenses/OpenVLA-MIT.txt), Copyright (c) 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti. This reference is not a license grant for its separate weights or training data. |
| [RoboCasa](https://github.com/robocasa/robocasa/tree/a9e152e2eea16d342dac40e4577fb4c68b932854) | Evaluation benchmark in README | [MIT code with MuJoCo Apache-2.0 notice](licenses/RoboCasa-MIT.txt), Copyright (c) 2026 the RoboCasa Team. Asset/data terms are separate. |
| [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin/tree/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f) | `env/requirements_RoboTwin.txt` environment snapshot | [MIT code](licenses/RoboTwin-MIT.txt), Copyright (c) 2025 Tianxing Chen (陈天行). No RoboTwin simulator source/assets are bundled here. |
| [frankapy](https://github.com/iamlab-cmu/frankapy/tree/4ba2872ddfa460fd40efd372f72b6c0258e31abb) | Real-robot interface referenced in README | [MIT](licenses/frankapy-MIT.txt), Copyright (c) 2021 Carnegie Mellon University. Other drivers and SDKs keep their own terms. |
| [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) | Evaluation benchmark in README | No repository-level license was identified in the checked upstream repository on 2026-10-18. Do not assume original LIBERO's MIT license covers all added code/assets; obtain the applicable terms from its maintainers. Only the benchmark reference is included here. |

For installed packages, see the [Python dependency inventory](licenses/PYTHON_DEPENDENCIES.md)
and [Git/URL dependency notices](licenses/VCS_DEPENDENCIES.md). These cover declared
core and optional environments under `env/`. Environment snapshots list software
to install separately; package binaries are not redistributed here. CUDA,
toolchains, and platform SDKs retain their distributors' terms.

## Pretrained models and tokenizers

| Download/reference | Use | Applicable terms |
| --- | --- | --- |
| [lerobot/pi0_base](https://huggingface.co/lerobot/pi0_base) | MoT backbone initialization | The model card declares `gemma` and identifies `google/paligemma-3b-pt-224` as its base. Follow the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and [Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). Apache-2.0 for LeRobot/openpi code does not relicense these weights. |
| [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) | Tokenizer and associated model resources | Gemma terms and model access conditions. PaliGemma incorporates Gemma and SigLIP; its assets are not relicensed by this repository. |
| [John6666/flux1-dev-fp8-flux `/vae`](https://huggingface.co/John6666/flux1-dev-fp8-flux/tree/main/vae) | Which2Act VAE via Diffusers `AutoencoderKL` | [Black Forest Labs explicitly releases its autoencoder weights under Apache-2.0](https://github.com/black-forest-labs/flux/blob/802fb4713906133fcbd0d8dc5351620ca4773036/README.md); [license copy](licenses/FLUX-Apache-2.0.txt). The mirror's overall model card declares FLUX.1-dev non-commercial terms. Identify and retain terms for the exact VAE artifact downloaded; a mirror-wide tag does not erase an upstream component-specific license. |
| [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) / [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | Upstream context; full denoisers are not required here | Full **dev** weights have the [FLUX.1-dev Non-Commercial License](licenses/FLUX1-dev-Non-Commercial.txt); **schnell** is Apache-2.0 per the [upstream model table](https://github.com/black-forest-labs/flux/blob/802fb4713906133fcbd0d8dc5351620ca4773036/README.md). These differ from each other and from Diffusers' source-code license. |
| [FoundationVision/var `vae_ch160v4096z32.pth`](https://huggingface.co/FoundationVision/var/blob/2fc3ff144f97ddd12207727b60dd6b997bc4cb69/vae_ch160v4096z32.pth) | Legacy Which2Act VQ-VAE | The [model card at this revision](https://huggingface.co/FoundationVision/var/blob/2fc3ff144f97ddd12207727b60dd6b997bc4cb69/README.md) declares MIT. Retain attribution and conditions shipped with the checkpoint. |
| [IDEA-Research/Rex-Omni](https://huggingface.co/IDEA-Research/Rex-Omni) | Annotation weights | IDEA License 1.0. Its model card additionally identifies its Qwen2.5-VL-3B base and the [Qwen Research License](licenses/Qwen2.5-VL-3B-Research.txt), which grants non-commercial use and requires separate permission for commercial use. This differs from Qwen3-VL's Apache-2.0 repository code. |
| [SAM](https://github.com/facebookresearch/segment-anything#model-checkpoints) / [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects#installation) | 2D masks / 3D supervision | Original SAM checkpoints: Apache-2.0 per its README. SAM 3D Objects checkpoints: custom SAM License. Use the original distributors' access/download procedures. |

For workflows using Rex-Omni's Qwen-derived model or its annotations: **Built with
Qwen**. Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c)
Alibaba Cloud. All Rights Reserved. Retain the model license's required notices
with any separately released models improved using its outputs.

Gemma model redistribution has agreement-copy, notice, modification, and
use-restriction requirements. This source repository is not a substitute for the
license package required when distributing trained or fine-tuned model weights.

## External datasets and research references

The README's grounding/pointing mixture, InternData-A1, DROID, LIBERO, CALVIN,
RoboInter, and other dataset names refer to external data, not bundled datasets
or new license grants. Original media, annotations, simulator assets, and derived
annotations may have different terms. Preserve each source's conditions and
citations; a model/software license does not license its training data.

[InternData-A1](https://huggingface.co/datasets/InternRobotics/InternData-A1) states
**CC BY-NC-SA 4.0** for the data and code in that dataset repository and has access
conditions. [SA-1B](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/README.md#dataset)
uses a **Dataset Research License**, separate from SAM's Apache-2.0 code/model.
The remaining source collections keep their own published terms; this notice
does not certify one license for the combined training mixture or authorize its
redistribution.

BLIP-2, Seer, DETR, AURORA, and ReconVLA are design/research references named in
source comments. A citation alone is not an assertion that the project's
implementation is included or that its license governs unrelated code. Additional
code or assets taken from those projects must retain their own notices and terms.
