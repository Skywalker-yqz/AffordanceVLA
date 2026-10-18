# Third-party notices

Original AffordanceVLA contributions are licensed under the [MIT License](LICENSE).
The adapted VQ-VAE implementation in `src/models/vqvae/` includes material from,
and references to, the upstream projects listed below. These third-party portions
retain their respective upstream notices and terms. The original source references
in the Python files are preserved.

| Upstream project | Relevant local files | Upstream license copy |
| --- | --- | --- |
| [FoundationVision/VAR](https://github.com/FoundationVision/VAR/tree/78b95394fc5896192e3a003e4b295f8ea743c48f) | `src/models/vqvae/vqvae.py`, `quant.py`, `basic_vae.py` | [MIT](licenses/VAR-MIT.txt) |
| [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers/tree/3ba01b241669f5ade541ce990f7650a3b8f65318) | `src/models/vqvae/quant.py` | [MIT](licenses/taming-transformers-MIT.txt) |
| [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion/tree/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229) | `src/models/vqvae/basic_vae.py`, `vqvae.py` | [CreativeML Open RAIL-M](licenses/stable-diffusion-OpenRAIL-M.txt) |

The license files are unmodified copies from the linked upstream revisions.
The Stable Diffusion notice distinguishes models and model derivatives from
accompanying source code ("Complementary Material"); it is included for the cited
upstream material and does not replace this project's MIT license for original code.

No pretrained weights are bundled in this repository. Separately downloaded
models, datasets, and optional dependencies remain subject to their own terms.
