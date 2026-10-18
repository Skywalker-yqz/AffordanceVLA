# Git and direct-download dependency licenses

These projects are installed separately by the environment snapshots; their
distributions are not bundled in this repository. The links below identify the
actual revisions named in `env/requirements*.txt`, rather than substituting the
license of a newer version. License files were checked on 2026-10-18.

See the [Python dependency inventory](PYTHON_DEPENDENCIES.md) for the complete
list of 14 distinct Git/wheel declarations and their configuration files.
The [main third-party notices](../THIRD_PARTY_NOTICES.md) cover Rex-Omni,
Segment Anything, LIBERO, CALVIN, and `calvin_env`, including their pinned
revisions and separate model/data terms.

| Project | Declared revision / release | Source-code license at that revision |
| --- | --- | --- |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | `2323de5905d5e90e035f792fe65bad0fedd413e7` | [Apache-2.0](https://github.com/nerfstudio-project/gsplat/blob/2323de5905d5e90e035f792fe65bad0fedd413e7/LICENSE), Copyright 2025 Nerfstudio Team. |
| [MoGe](https://github.com/microsoft/MoGe) | `a8c37341bc0325ca99b9d57981cc3bb2bd3e255b` | The [license file](https://github.com/microsoft/MoGe/blob/a8c37341bc0325ca99b9d57981cc3bb2bd3e255b/LICENSE) contains the Microsoft MIT notice followed by Apache-2.0 terms. Preserve the complete file and component notices; this inventory does not infer an AND/OR relationship or assign the code's license to separately downloaded weights. |
| [utils3d](https://github.com/EasternJournalist/utils3d) | `3913c65d81e05e47b9f367250cf8c0f7462a0900` | [MIT](https://github.com/EasternJournalist/utils3d/blob/3913c65d81e05e47b9f367250cf8c0f7462a0900/LICENSE), Copyright (c) 2022 EasternJournalist. |
| [PyTorch3D](https://github.com/facebookresearch/pytorch3d) | `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba`; `75ebeeaea0908c5527e7b1e305fbc7681382db47` | BSD-3-Clause at [first revision](https://github.com/facebookresearch/pytorch3d/blob/33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba/LICENSE) and [second revision](https://github.com/facebookresearch/pytorch3d/blob/75ebeeaea0908c5527e7b1e305fbc7681382db47/LICENSE), Copyright (c) Meta Platforms, Inc. and affiliates. |
| [OpenAI CLIP](https://github.com/openai/CLIP) | `dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1` | [MIT](https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/LICENSE), Copyright (c) 2021 OpenAI. |
| [PyOpenGL fork](https://github.com/mmatl/pyopengl) | `76d1261adee2d3fd99b418e75b0416bb7d2865e6` | [BSD-style PyOpenGL/OpenGL-ctypes licenses and additional component notices](https://github.com/mmatl/pyopengl/blob/76d1261adee2d3fd99b418e75b0416bb7d2865e6/license.txt). The full file also describes conditional GLE, FreeGLUT, Togl, OS-Mesa, and Regal notices; retain those applicable to the actual distribution. |
| [TACTO fork](https://github.com/lukashermann/tacto) | `dd53360d9a8c186f0d6439372ec0be0fa5e21731` | [MIT](https://github.com/lukashermann/tacto/blob/dd53360d9a8c186f0d6439372ec0be0fa5e21731/LICENSE), Copyright (c) Facebook, Inc. and its affiliates. |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | `v2.5.9.post1` release wheel | [BSD-3-Clause](https://github.com/Dao-AILab/flash-attention/blob/v2.5.9.post1/LICENSE), Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file. The CUDA/PyTorch-specific wheel also retains the licenses of any bundled components. |

The FlashAttention declaration pins a wheel SHA-256 in the requirements files.
This notice verifies the corresponding source release's license, not the contents
or license bundle of that binary wheel. Keep the actual wheel/environment's
license files when redistributing it. The same distinction applies to native
libraries, simulator assets, and model checkpoints obtained by any dependency.
