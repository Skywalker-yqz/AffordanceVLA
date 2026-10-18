# Python dependency license inventory — AffordanceVLA

This inventory records upstream-reported license metadata for the software explicitly declared by this repository. The repository license does not replace the licenses of these packages. See [Third-party notices](../THIRD_PARTY_NOTICES.md) for source-code attributions, external repositories, model checkpoints, benchmark assets, and non-Python software.

**Metadata checked:** 2026-10-18. **Scope:** 522 normalized PyPI package names and 751 declared package/version entries, plus any direct Git/URL requirements listed below.

This is a declaration inventory, not a complete resolved or installed SBOM. Optional environment snapshots can contain different versions and unused development tools; their inclusion does not mean that every package is required by the core model. This inventory does not install packages, certify version availability or compatibility, or establish that an optional component is redistributed here.

The table uses each exact release's official [PyPI JSON metadata](https://docs.pypi.org/api/json/). An SPDX expression is reproduced when present; otherwise a short license field or classifier is reported. Classifiers can be less precise than the actual license text. Where multiple labels are reported without an SPDX expression, no AND/OR relationship is inferred. Empty metadata and unavailable releases are explicitly marked; a different release's license is not silently substituted.

The linked distribution and upstream license files control. Wheels and source archives can include additional licenses for bundled native libraries, fonts, codecs, and other code. In particular, NVIDIA CUDA packages have NVIDIA terms, and FFmpeg-related packages can bundle components with additional terms. Preserve the actual distribution's license and notice files when redistributing it.

## Declaration sources

- [README.md](../README.md) and [env/README.md](../env/README.md): the documented PyTorch installation.
- [env/README.md](../env/README.md)
- [env/requirements_AffordanceVLA.txt](../env/requirements_AffordanceVLA.txt)
- [env/requirements_AffordanceVLA_calvin.txt](../env/requirements_AffordanceVLA_calvin.txt)
- [env/requirements_AffordanceVLA_libero.txt](../env/requirements_AffordanceVLA_libero.txt)
- [env/requirements_SAM3D.txt](../env/requirements_SAM3D.txt)
- [env/requirements_calvin_venv.txt](../env/requirements_calvin_venv.txt)
- [env/requirements_core.txt](../env/requirements_core.txt)
- [env/requirements_libero.txt](../env/requirements_libero.txt)
- [env/requirements_rexomni.txt](../env/requirements_rexomni.txt)
- [env/requirements_robointer.txt](../env/requirements_robointer.txt)

Some declarations use ranges or an unversioned install. The table below covers pinned versions present elsewhere in the repository; it does not extend those license checks to every version permitted by a range. Check the distribution actually selected by your installer.

| Declaration | Source |
| --- | --- |
| `pip (unversioned upgrade)` | [env/README.md](../env/README.md) |

## Declared packages

Each version in the source column links to that release's metadata. Rows are grouped only when the reported license summary is identical.

| Package | Declared version(s) | Reported license | Metadata source |
| --- | --- | --- | --- |
| `absl-py` | `2.3.1` | Apache-2.0 | [2.3.1](https://pypi.org/pypi/absl-py/2.3.1/json) |
| `accelerate` | `1.10.1`, `1.12.0` | Apache | [1.10.1](https://pypi.org/pypi/accelerate/1.10.1/json), [1.12.0](https://pypi.org/pypi/accelerate/1.12.0/json) |
| `addict` | `2.4.0` | MIT License | [2.4.0](https://pypi.org/pypi/addict/2.4.0/json) |
| `aiofiles` | `23.2.1`, `24.1.0` | Apache-2.0 | [23.2.1](https://pypi.org/pypi/aiofiles/23.2.1/json), [24.1.0](https://pypi.org/pypi/aiofiles/24.1.0/json) |
| `aiohappyeyeballs` | `2.4.4`, `2.6.1` | PSF-2.0 | [2.4.4](https://pypi.org/pypi/aiohappyeyeballs/2.4.4/json), [2.6.1](https://pypi.org/pypi/aiohappyeyeballs/2.6.1/json) |
| `aiohttp` | `3.10.11` | Apache 2 | [3.10.11](https://pypi.org/pypi/aiohttp/3.10.11/json) |
| `aiohttp` | `3.13.2` | Apache-2.0 AND MIT | [3.13.2](https://pypi.org/pypi/aiohttp/3.13.2/json) |
| `aiosignal` | `1.3.1`, `1.4.0` | Apache 2.0 | [1.3.1](https://pypi.org/pypi/aiosignal/1.3.1/json), [1.4.0](https://pypi.org/pypi/aiosignal/1.4.0/json) |
| `airportsdata` | `20250909` | MIT (license text) | [20250909](https://pypi.org/pypi/airportsdata/20250909/json) |
| `aliyun-python-sdk-core` | `2.16.0` | Apache License 2.0 | [2.16.0](https://pypi.org/pypi/aliyun-python-sdk-core/2.16.0/json) |
| `aliyun-python-sdk-kms` | `2.16.5` | Apache | [2.16.5](https://pypi.org/pypi/aliyun-python-sdk-kms/2.16.5/json) |
| `annotated-doc` | `0.0.4` | MIT | [0.0.4](https://pypi.org/pypi/annotated-doc/0.0.4/json) |
| `annotated-types` | `0.7.0` | MIT License | [0.7.0](https://pypi.org/pypi/annotated-types/0.7.0/json) |
| `antlr4-python3-runtime` | `4.8`, `4.9.3` | BSD | [4.8](https://pypi.org/pypi/antlr4-python3-runtime/4.8/json), [4.9.3](https://pypi.org/pypi/antlr4-python3-runtime/4.9.3/json) |
| `anyio` | `4.12.0` | MIT | [4.12.0](https://pypi.org/pypi/anyio/4.12.0/json) |
| `appdirs` | `1.4.4` | MIT | [1.4.4](https://pypi.org/pypi/appdirs/1.4.4/json) |
| `argon2-cffi` | `25.1.0` | MIT | [25.1.0](https://pypi.org/pypi/argon2-cffi/25.1.0/json) |
| `argon2-cffi-bindings` | `25.1.0` | MIT | [25.1.0](https://pypi.org/pypi/argon2-cffi-bindings/25.1.0/json) |
| `arrow` | `1.4.0` | Apache Software License | [1.4.0](https://pypi.org/pypi/arrow/1.4.0/json) |
| `astor` | `0.8.1` | BSD-3-Clause | [0.8.1](https://pypi.org/pypi/astor/0.8.1/json) |
| `asttokens` | `3.0.1` | Apache 2.0 | [3.0.1](https://pypi.org/pypi/asttokens/3.0.1/json) |
| `async-lru` | `2.0.5` | MIT License | [2.0.5](https://pypi.org/pypi/async-lru/2.0.5/json) |
| `async-timeout` | `4.0.3`, `5.0.1` | Apache 2 | [4.0.3](https://pypi.org/pypi/async-timeout/4.0.3/json), [5.0.1](https://pypi.org/pypi/async-timeout/5.0.1/json) |
| `attrs` | `25.3.0`, `25.4.0` | MIT | [25.3.0](https://pypi.org/pypi/attrs/25.3.0/json), [25.4.0](https://pypi.org/pypi/attrs/25.4.0/json) |
| `attrs` | `23.2.0` | MIT License | [23.2.0](https://pypi.org/pypi/attrs/23.2.0/json) |
| `audioread` | `3.1.0` | MIT | [3.1.0](https://pypi.org/pypi/audioread/3.1.0/json) |
| `auto-gptq` | `0.7.1` | MIT License | [0.7.1](https://pypi.org/pypi/auto-gptq/0.7.1/json) |
| `autoflake` | `2.3.1` | MIT | [2.3.1](https://pypi.org/pypi/autoflake/2.3.1/json) |
| `av` | `12.0.0` | BSD | [12.0.0](https://pypi.org/pypi/av/12.0.0/json) |
| `av` | `16.0.1` | BSD-3-Clause | [16.0.1](https://pypi.org/pypi/av/16.0.1/json) |
| `azure-core` | `1.36.0` | MIT License | [1.36.0](https://pypi.org/pypi/azure-core/1.36.0/json) |
| `azure-identity` | `1.25.1` | Unspecified in release metadata; consult upstream | [1.25.1](https://pypi.org/pypi/azure-identity/1.25.1/json), [upstream](https://github.com/Azure/azure-sdk-for-python) |
| `azure-storage-blob` | `12.27.1` | MIT License | [12.27.1](https://pypi.org/pypi/azure-storage-blob/12.27.1/json) |
| `azure-storage-file-datalake` | `12.22.0` | MIT License | [12.22.0](https://pypi.org/pypi/azure-storage-file-datalake/12.22.0/json) |
| `babel` | `2.17.0` | BSD-3-Clause | [2.17.0](https://pypi.org/pypi/babel/2.17.0/json) |
| `bcrypt` | `5.0.0` | Apache-2.0 | [5.0.0](https://pypi.org/pypi/bcrypt/5.0.0/json) |
| `bddl` | `1.0.1` | Unspecified in release metadata; consult upstream | [1.0.1](https://pypi.org/pypi/bddl/1.0.1/json), [upstream](https://github.com/StanfordVL/bddl) |
| `beautifulsoup4` | `4.14.3` | MIT License | [4.14.3](https://pypi.org/pypi/beautifulsoup4/4.14.3/json) |
| `bitsandbytes` | `0.43.0` | MIT | [0.43.0](https://pypi.org/pypi/bitsandbytes/0.43.0/json) |
| `black` | `24.3.0` | MIT | [24.3.0](https://pypi.org/pypi/black/24.3.0/json) |
| `blake3` | `1.0.8` | CC0-1.0 OR Apache-2.0 | [1.0.8](https://pypi.org/pypi/blake3/1.0.8/json) |
| `bleach` | `6.3.0` | Apache Software License | [6.3.0](https://pypi.org/pypi/bleach/6.3.0/json) |
| `blinker` | `1.9.0` | MIT License | [1.9.0](https://pypi.org/pypi/blinker/1.9.0/json) |
| `boto3` | `1.42.2` | Apache-2.0 | [1.42.2](https://pypi.org/pypi/boto3/1.42.2/json) |
| `botocore` | `1.42.2` | Apache-2.0 | [1.42.2](https://pypi.org/pypi/botocore/1.42.2/json) |
| `bpy` | `4.3.0` | GPL-3.0 | [4.3.0](https://pypi.org/pypi/bpy/4.3.0/json) |
| `braceexpand` | `0.1.7` | MIT | [0.1.7](https://pypi.org/pypi/braceexpand/0.1.7/json) |
| `brotli` | `1.2.0` | MIT | [1.2.0](https://pypi.org/pypi/brotli/1.2.0/json) |
| `cachetools` | `6.2.2` | MIT | [6.2.2](https://pypi.org/pypi/cachetools/6.2.2/json) |
| `calmsize` | `0.1.3` | ZPL 2.1 | [0.1.3](https://pypi.org/pypi/calmsize/0.1.3/json) |
| `ccimport` | `0.4.4` | MIT | [0.4.4](https://pypi.org/pypi/ccimport/0.4.4/json) |
| `certifi` | `2025.11.12`, `2026.1.4` | MPL-2.0 | [2025.11.12](https://pypi.org/pypi/certifi/2025.11.12/json), [2026.1.4](https://pypi.org/pypi/certifi/2026.1.4/json) |
| `cffi` | `1.17.1`, `2.0.0` | MIT | [1.17.1](https://pypi.org/pypi/cffi/1.17.1/json), [2.0.0](https://pypi.org/pypi/cffi/2.0.0/json) |
| `charset-normalizer` | `3.4.4` | MIT | [3.4.4](https://pypi.org/pypi/charset-normalizer/3.4.4/json) |
| `circuitbreaker` | `2.1.3` | BSD-3-Clause | [2.1.3](https://pypi.org/pypi/circuitbreaker/2.1.3/json) |
| `click` | `8.1.8` | BSD License | [8.1.8](https://pypi.org/pypi/click/8.1.8/json) |
| `click` | `8.2.1`, `8.3.1` | BSD-3-Clause | [8.2.1](https://pypi.org/pypi/click/8.2.1/json), [8.3.1](https://pypi.org/pypi/click/8.3.1/json) |
| `cloudpickle` | `2.1.0` | BSD 3-Clause License | [2.1.0](https://pypi.org/pypi/cloudpickle/2.1.0/json) |
| `cloudpickle` | `3.1.2` | BSD-3-Clause | [3.1.2](https://pypi.org/pypi/cloudpickle/3.1.2/json) |
| `cmake` | `3.18.4` | Unverified: exact release metadata unavailable | [3.18.4](https://pypi.org/pypi/cmake/3.18.4/json), [upstream](https://pypi.org/project/cmake/) |
| `colorama` | `0.4.6` | BSD License | [0.4.6](https://pypi.org/pypi/colorama/0.4.6/json) |
| `coloredlogs` | `15.0.1` | MIT | [15.0.1](https://pypi.org/pypi/coloredlogs/15.0.1/json) |
| `colorlog` | `6.10.1` | MIT License | [6.10.1](https://pypi.org/pypi/colorlog/6.10.1/json) |
| `comm` | `0.2.3` | BSD License (classifier; full notices in metadata) | [0.2.3](https://pypi.org/pypi/comm/0.2.3/json) |
| `compressed-tensors` | `0.10.1` | Apache 2.0 | [0.10.1](https://pypi.org/pypi/compressed-tensors/0.10.1/json) |
| `conda-pack` | `0.7.1` | BSD | [0.7.1](https://pypi.org/pypi/conda-pack/0.7.1/json) |
| `configargparse` | `1.7.1` | MIT | [1.7.1](https://pypi.org/pypi/configargparse/1.7.1/json) |
| `contourpy` | `1.1.1`, `1.3.0`, `1.3.2`, `1.3.3` | BSD License (classifier; full notices in metadata) | [1.1.1](https://pypi.org/pypi/contourpy/1.1.1/json), [1.3.0](https://pypi.org/pypi/contourpy/1.3.0/json), [1.3.2](https://pypi.org/pypi/contourpy/1.3.2/json), [1.3.3](https://pypi.org/pypi/contourpy/1.3.3/json) |
| `cramjam` | `2.11.0` | MIT | [2.11.0](https://pypi.org/pypi/cramjam/2.11.0/json) |
| `crcmod` | `1.7` | MIT | [1.7](https://pypi.org/pypi/crcmod/1.7/json) |
| `cryptography` | `45.0.7`, `46.0.3`, `46.0.5` | Apache-2.0 OR BSD-3-Clause | [45.0.7](https://pypi.org/pypi/cryptography/45.0.7/json), [46.0.3](https://pypi.org/pypi/cryptography/46.0.3/json), [46.0.5](https://pypi.org/pypi/cryptography/46.0.5/json) |
| `cuda-python` | `12.1.0` | NVIDIA Proprietary License | [12.1.0](https://pypi.org/pypi/cuda-python/12.1.0/json) |
| `cumm-cu121` | `0.7.11` | MIT | [0.7.11](https://pypi.org/pypi/cumm-cu121/0.7.11/json) |
| `cupy-cuda12x` | `13.6.0` | MIT | [13.6.0](https://pypi.org/pypi/cupy-cuda12x/13.6.0/json) |
| `cycler` | `0.12.1` | BSD License (classifier; full notices in metadata) | [0.12.1](https://pypi.org/pypi/cycler/0.12.1/json) |
| `cython` | `3.2.2` | Apache-2.0 | [3.2.2](https://pypi.org/pypi/cython/3.2.2/json) |
| `dash` | `3.3.0` | MIT | [3.3.0](https://pypi.org/pypi/dash/3.3.0/json) |
| `dataclasses` | `0.6` | Apache | [0.6](https://pypi.org/pypi/dataclasses/0.6/json) |
| `dataclasses-json` | `0.6.7` | MIT | [0.6.7](https://pypi.org/pypi/dataclasses-json/0.6.7/json) |
| `datasets` | `2.14.4` | Apache 2.0 | [2.14.4](https://pypi.org/pypi/datasets/2.14.4/json) |
| `debugpy` | `1.8.17` | MIT | [1.8.17](https://pypi.org/pypi/debugpy/1.8.17/json) |
| `decorator` | `5.2.1` | BSD-2-Clause | [5.2.1](https://pypi.org/pypi/decorator/5.2.1/json) |
| `decorator` | `4.4.2` | new BSD License | [4.4.2](https://pypi.org/pypi/decorator/4.4.2/json) |
| `decord` | `0.6.0` | APACHE | [0.6.0](https://pypi.org/pypi/decord/0.6.0/json) |
| `defusedxml` | `0.7.1` | PSFL | [0.7.1](https://pypi.org/pypi/defusedxml/0.7.1/json) |
| `deprecated` | `1.3.1` | MIT | [1.3.1](https://pypi.org/pypi/deprecated/1.3.1/json) |
| `deprecation` | `2.1.0` | Apache 2 | [2.1.0](https://pypi.org/pypi/deprecation/2.1.0/json) |
| `depyf` | `0.18.0` | MIT | [0.18.0](https://pypi.org/pypi/depyf/0.18.0/json) |
| `diffusers` | `0.31.0`, `0.36.0` | Apache 2.0 License | [0.31.0](https://pypi.org/pypi/diffusers/0.31.0/json), [0.36.0](https://pypi.org/pypi/diffusers/0.36.0/json) |
| `dill` | `0.3.7`, `0.4.0` | BSD-3-Clause | [0.3.7](https://pypi.org/pypi/dill/0.3.7/json), [0.4.0](https://pypi.org/pypi/dill/0.4.0/json) |
| `diskcache` | `5.6.3` | Apache 2.0 | [5.6.3](https://pypi.org/pypi/diskcache/5.6.3/json) |
| `distro` | `1.9.0` | Apache License, Version 2.0 | [1.9.0](https://pypi.org/pypi/distro/1.9.0/json) |
| `dnspython` | `2.8.0` | ISC | [2.8.0](https://pypi.org/pypi/dnspython/2.8.0/json) |
| `docker` | `7.1.0` | Apache Software License | [7.1.0](https://pypi.org/pypi/docker/7.1.0/json) |
| `docker-pycreds` | `0.4.0` | Apache License 2.0 | [0.4.0](https://pypi.org/pypi/docker-pycreds/0.4.0/json) |
| `docopt` | `0.6.2` | MIT | [0.6.2](https://pypi.org/pypi/docopt/0.6.2/json) |
| `easydict` | `1.13` | LGPL-3.0 | [1.13](https://pypi.org/pypi/easydict/1.13/json) |
| `easydict` | `1.9` | LPGL, see LICENSE file. | [1.9](https://pypi.org/pypi/easydict/1.9/json) |
| `egl-probe` | `1.0.2` | Unspecified in release metadata; consult upstream | [1.0.2](https://pypi.org/pypi/egl-probe/1.0.2/json), [upstream](https://github.com/StanfordVL/egl_probe) |
| `einops` | `0.8.1` | MIT | [0.8.1](https://pypi.org/pypi/einops/0.8.1/json) |
| `einops` | `0.4.1` | Unspecified in release metadata; consult upstream | [0.4.1](https://pypi.org/pypi/einops/0.4.1/json), [upstream](https://github.com/arogozhnikov/einops) |
| `einops-exts` | `0.0.4` | MIT | [0.0.4](https://pypi.org/pypi/einops-exts/0.0.4/json) |
| `email-validator` | `2.3.0` | Unlicense | [2.3.0](https://pypi.org/pypi/email-validator/2.3.0/json) |
| `entrypoints` | `0.4` | MIT License | [0.4](https://pypi.org/pypi/entrypoints/0.4/json) |
| `etils` | `1.13.0`, `1.3.0` | Apache Software License | [1.13.0](https://pypi.org/pypi/etils/1.13.0/json), [1.3.0](https://pypi.org/pypi/etils/1.3.0/json) |
| `eval-type-backport` | `0.3.1` | MIT | [0.3.1](https://pypi.org/pypi/eval-type-backport/0.3.1/json) |
| `exceptiongroup` | `1.2.0`, `1.3.1` | MIT License | [1.2.0](https://pypi.org/pypi/exceptiongroup/1.2.0/json), [1.3.1](https://pypi.org/pypi/exceptiongroup/1.3.1/json) |
| `executing` | `2.2.1` | MIT | [2.2.1](https://pypi.org/pypi/executing/2.2.1/json) |
| `fastapi` | `0.123.5`, `0.124.0`, `0.124.4` | MIT | [0.123.5](https://pypi.org/pypi/fastapi/0.123.5/json), [0.124.0](https://pypi.org/pypi/fastapi/0.124.0/json), [0.124.4](https://pypi.org/pypi/fastapi/0.124.4/json) |
| `fastapi-cli` | `0.0.16` | MIT | [0.0.16](https://pypi.org/pypi/fastapi-cli/0.0.16/json) |
| `fastapi-cloud-cli` | `0.6.0` | MIT | [0.6.0](https://pypi.org/pypi/fastapi-cloud-cli/0.6.0/json) |
| `fastar` | `0.8.0` | MIT License | [0.8.0](https://pypi.org/pypi/fastar/0.8.0/json) |
| `fastavro` | `1.9.4` | MIT | [1.9.4](https://pypi.org/pypi/fastavro/1.9.4/json) |
| `fastcore` | `1.8.16`, `1.9.2` | Apache Software License 2.0 | [1.8.16](https://pypi.org/pypi/fastcore/1.8.16/json), [1.9.2](https://pypi.org/pypi/fastcore/1.9.2/json) |
| `fasteners` | `0.19` | Apache-2.0 | [0.19](https://pypi.org/pypi/fasteners/0.19/json) |
| `fastjsonschema` | `2.21.2` | BSD | [2.21.2](https://pypi.org/pypi/fastjsonschema/2.21.2/json) |
| `fastrlock` | `0.8.3` | MIT style | [0.8.3](https://pypi.org/pypi/fastrlock/0.8.3/json) |
| `ffmpy` | `1.0.0` | MIT | [1.0.0](https://pypi.org/pypi/ffmpy/1.0.0/json) |
| `filelock` | `3.16.1` | The Unlicense (Unlicense) | [3.16.1](https://pypi.org/pypi/filelock/3.16.1/json) |
| `filelock` | `3.20.0`, `3.20.2` | Unlicense | [3.20.0](https://pypi.org/pypi/filelock/3.20.0/json), [3.20.2](https://pypi.org/pypi/filelock/3.20.2/json) |
| `findpydeps` | `0.2.8` | MIT License | [0.2.8](https://pypi.org/pypi/findpydeps/0.2.8/json) |
| `fire` | `0.7.1` | Apache-2.0 | [0.7.1](https://pypi.org/pypi/fire/0.7.1/json) |
| `flake8` | `7.0.0` | MIT | [7.0.0](https://pypi.org/pypi/flake8/7.0.0/json) |
| `flamingo-pytorch` | `0.1.2` | MIT | [0.1.2](https://pypi.org/pypi/flamingo-pytorch/0.1.2/json) |
| `flash-attn` | `2.8.3` | BSD License | [2.8.3](https://pypi.org/pypi/flash-attn/2.8.3/json) |
| `flask` | `3.0.3` | BSD License | [3.0.3](https://pypi.org/pypi/flask/3.0.3/json) |
| `flask` | `3.1.2` | BSD-3-Clause | [3.1.2](https://pypi.org/pypi/flask/3.1.2/json) |
| `fonttools` | `4.57.0`, `4.60.2`, `4.61.0`, `4.61.1` | MIT | [4.57.0](https://pypi.org/pypi/fonttools/4.57.0/json), [4.60.2](https://pypi.org/pypi/fonttools/4.60.2/json), [4.61.0](https://pypi.org/pypi/fonttools/4.61.0/json), [4.61.1](https://pypi.org/pypi/fonttools/4.61.1/json) |
| `fqdn` | `1.5.1` | MPL 2.0 | [1.5.1](https://pypi.org/pypi/fqdn/1.5.1/json) |
| `freetype-py` | `2.5.1` | BSD License | [2.5.1](https://pypi.org/pypi/freetype-py/2.5.1/json) |
| `frozenlist` | `1.5.0` | Apache 2 | [1.5.0](https://pypi.org/pypi/frozenlist/1.5.0/json) |
| `frozenlist` | `1.8.0` | Apache-2.0 | [1.8.0](https://pypi.org/pypi/frozenlist/1.8.0/json) |
| `fsspec` | `2025.3.0` | BSD License (classifier; full notices in metadata) | [2025.3.0](https://pypi.org/pypi/fsspec/2025.3.0/json) |
| `fsspec` | `2025.12.0` | BSD-3-Clause | [2025.12.0](https://pypi.org/pypi/fsspec/2025.12.0/json) |
| `ftfy` | `6.2.0`, `6.3.1` | Apache-2.0 | [6.2.0](https://pypi.org/pypi/ftfy/6.2.0/json), [6.3.1](https://pypi.org/pypi/ftfy/6.3.1/json) |
| `future` | `0.18.2` | MIT | [0.18.2](https://pypi.org/pypi/future/0.18.2/json) |
| `fvcore` | `0.1.5.post20221221` | Apache 2.0 | [0.1.5.post20221221](https://pypi.org/pypi/fvcore/0.1.5.post20221221/json) |
| `gdown` | `5.2.0` | MIT | [5.2.0](https://pypi.org/pypi/gdown/5.2.0/json) |
| `gekko` | `1.3.0` | MIT | [1.3.0](https://pypi.org/pypi/gekko/1.3.0/json) |
| `gguf` | `0.17.1` | MIT License | [0.17.1](https://pypi.org/pypi/gguf/0.17.1/json) |
| `gitdb` | `4.0.12` | BSD License | [4.0.12](https://pypi.org/pypi/gitdb/4.0.12/json) |
| `gitpython` | `3.1.45`, `3.1.46` | BSD-3-Clause | [3.1.45](https://pypi.org/pypi/gitpython/3.1.45/json), [3.1.46](https://pypi.org/pypi/gitpython/3.1.46/json) |
| `glcontext` | `3.0.0` | MIT | [3.0.0](https://pypi.org/pypi/glcontext/3.0.0/json) |
| `glfw` | `2.10.0` | MIT | [2.10.0](https://pypi.org/pypi/glfw/2.10.0/json) |
| `google-api-core` | `2.28.1` | Apache 2.0 | [2.28.1](https://pypi.org/pypi/google-api-core/2.28.1/json) |
| `google-auth` | `2.43.0`, `2.48.0` | Apache 2.0 | [2.43.0](https://pypi.org/pypi/google-auth/2.43.0/json), [2.48.0](https://pypi.org/pypi/google-auth/2.48.0/json) |
| `google-auth-oauthlib` | `1.0.0` | Apache 2.0 | [1.0.0](https://pypi.org/pypi/google-auth-oauthlib/1.0.0/json) |
| `google-cloud-core` | `2.5.0` | Apache 2.0 | [2.5.0](https://pypi.org/pypi/google-cloud-core/2.5.0/json) |
| `google-cloud-storage` | `2.10.0` | Apache 2.0 | [2.10.0](https://pypi.org/pypi/google-cloud-storage/2.10.0/json) |
| `google-crc32c` | `1.7.1` | Apache 2.0 | [1.7.1](https://pypi.org/pypi/google-crc32c/1.7.1/json) |
| `google-pasta` | `0.2.0` | Apache 2.0 | [0.2.0](https://pypi.org/pypi/google-pasta/0.2.0/json) |
| `google-resumable-media` | `2.8.0` | Apache 2.0 | [2.8.0](https://pypi.org/pypi/google-resumable-media/2.8.0/json) |
| `googleapis-common-protos` | `1.72.0` | Apache 2.0 | [1.72.0](https://pypi.org/pypi/googleapis-common-protos/1.72.0/json) |
| `gradio` | `4.44.1` | Apache Software License | [4.44.1](https://pypi.org/pypi/gradio/4.44.1/json) |
| `gradio` | `6.1.0` | Apache-2.0 | [6.1.0](https://pypi.org/pypi/gradio/6.1.0/json) |
| `gradio-client` | `1.3.0` | Apache Software License | [1.3.0](https://pypi.org/pypi/gradio-client/1.3.0/json) |
| `gradio-client` | `2.0.1` | Apache-2.0 | [2.0.1](https://pypi.org/pypi/gradio-client/2.0.1/json) |
| `gradio-image-prompter` | `0.1.0` | Apache Software License | [0.1.0](https://pypi.org/pypi/gradio-image-prompter/0.1.0/json) |
| `groovy` | `0.1.2` | MIT License | [0.1.2](https://pypi.org/pypi/groovy/0.1.2/json) |
| `grpcio` | `1.70.0`, `1.76.0` | Apache License 2.0 | [1.70.0](https://pypi.org/pypi/grpcio/1.70.0/json), [1.76.0](https://pypi.org/pypi/grpcio/1.76.0/json) |
| `gym` | `0.25.2`, `0.26.2` | MIT | [0.25.2](https://pypi.org/pypi/gym/0.25.2/json), [0.26.2](https://pypi.org/pypi/gym/0.26.2/json) |
| `gym-notices` | `0.1.0` | MIT License | [0.1.0](https://pypi.org/pypi/gym-notices/0.1.0/json) |
| `h11` | `0.16.0` | MIT | [0.16.0](https://pypi.org/pypi/h11/0.16.0/json) |
| `h5py` | `3.11.0`, `3.15.1` | BSD-3-Clause | [3.11.0](https://pypi.org/pypi/h5py/3.11.0/json), [3.15.1](https://pypi.org/pypi/h5py/3.15.1/json) |
| `hdfs` | `2.7.3` | MIT | [2.7.3](https://pypi.org/pypi/hdfs/2.7.3/json) |
| `hf-xet` | `1.2.0` | Apache Software License | [1.2.0](https://pypi.org/pypi/hf-xet/1.2.0/json) |
| `httpcore` | `1.0.9` | BSD-3-Clause | [1.0.9](https://pypi.org/pypi/httpcore/1.0.9/json) |
| `httplib2` | `0.22.0` | MIT | [0.22.0](https://pypi.org/pypi/httplib2/0.22.0/json) |
| `httptools` | `0.7.1` | MIT | [0.7.1](https://pypi.org/pypi/httptools/0.7.1/json) |
| `httpx` | `0.28.1` | BSD-3-Clause | [0.28.1](https://pypi.org/pypi/httpx/0.28.1/json) |
| `huggingface-hub` | `0.29.3`, `0.36.0`, `0.36.2` | Apache | [0.29.3](https://pypi.org/pypi/huggingface-hub/0.29.3/json), [0.36.0](https://pypi.org/pypi/huggingface-hub/0.36.0/json), [0.36.2](https://pypi.org/pypi/huggingface-hub/0.36.2/json) |
| `humanfriendly` | `10.0` | MIT | [10.0](https://pypi.org/pypi/humanfriendly/10.0/json) |
| `hydra-colorlog` | `1.2.0` | MIT License | [1.2.0](https://pypi.org/pypi/hydra-colorlog/1.2.0/json) |
| `hydra-core` | `1.1.1`, `1.2.0`, `1.3.2` | MIT | [1.1.1](https://pypi.org/pypi/hydra-core/1.1.1/json), [1.2.0](https://pypi.org/pypi/hydra-core/1.2.0/json), [1.3.2](https://pypi.org/pypi/hydra-core/1.3.2/json) |
| `hydra-submitit-launcher` | `1.2.0` | MIT License | [1.2.0](https://pypi.org/pypi/hydra-submitit-launcher/1.2.0/json) |
| `idna` | `3.11` | BSD-3-Clause | [3.11](https://pypi.org/pypi/idna/3.11/json) |
| `igraph` | `0.11.8`, `1.0.0` | GNU General Public License (GPL) | [0.11.8](https://pypi.org/pypi/igraph/0.11.8/json), [1.0.0](https://pypi.org/pypi/igraph/1.0.0/json) |
| `imageio` | `2.35.1`, `2.37.2` | BSD-2-Clause | [2.35.1](https://pypi.org/pypi/imageio/2.35.1/json), [2.37.2](https://pypi.org/pypi/imageio/2.37.2/json) |
| `imageio-ffmpeg` | `0.5.1`, `0.6.0` | BSD-2-Clause | [0.5.1](https://pypi.org/pypi/imageio-ffmpeg/0.5.1/json), [0.6.0](https://pypi.org/pypi/imageio-ffmpeg/0.6.0/json) |
| `imath` | `0.0.2` | MIT License | [0.0.2](https://pypi.org/pypi/imath/0.0.2/json) |
| `importlib-metadata` | `6.11.0`, `8.5.0`, `8.7.0` | Apache Software License | [6.11.0](https://pypi.org/pypi/importlib-metadata/6.11.0/json), [8.5.0](https://pypi.org/pypi/importlib-metadata/8.5.0/json), [8.7.0](https://pypi.org/pypi/importlib-metadata/8.7.0/json) |
| `importlib-resources` | `6.4.5`, `6.5.2` | Apache Software License | [6.4.5](https://pypi.org/pypi/importlib-resources/6.4.5/json), [6.5.2](https://pypi.org/pypi/importlib-resources/6.5.2/json) |
| `iniconfig` | `2.1.0`, `2.3.0` | MIT | [2.1.0](https://pypi.org/pypi/iniconfig/2.1.0/json), [2.3.0](https://pypi.org/pypi/iniconfig/2.3.0/json) |
| `inquirerpy` | `0.3.4` | MIT | [0.3.4](https://pypi.org/pypi/inquirerpy/0.3.4/json) |
| `interegular` | `0.3.3` | MIT | [0.3.3](https://pypi.org/pypi/interegular/0.3.3/json) |
| `iopath` | `0.1.10` | MIT licensed, as found in the LICENSE file | [0.1.10](https://pypi.org/pypi/iopath/0.1.10/json) |
| `ipycanvas` | `0.14.2`, `0.14.3` | BSD License (classifier; full notices in metadata) | [0.14.2](https://pypi.org/pypi/ipycanvas/0.14.2/json), [0.14.3](https://pypi.org/pypi/ipycanvas/0.14.3/json) |
| `ipyevents` | `2.0.4` | BSD 3-clause | [2.0.4](https://pypi.org/pypi/ipyevents/2.0.4/json) |
| `ipykernel` | `7.1.0` | BSD-3-Clause | [7.1.0](https://pypi.org/pypi/ipykernel/7.1.0/json) |
| `ipython` | `8.37.0`, `9.8.0` | BSD-3-Clause | [8.37.0](https://pypi.org/pypi/ipython/8.37.0/json), [9.8.0](https://pypi.org/pypi/ipython/9.8.0/json) |
| `ipython-pygments-lexers` | `1.1.1` | BSD License | [1.1.1](https://pypi.org/pypi/ipython-pygments-lexers/1.1.1/json) |
| `ipywidgets` | `8.1.8` | BSD 3-Clause License | [8.1.8](https://pypi.org/pypi/ipywidgets/8.1.8/json) |
| `isodate` | `0.7.2` | BSD License (classifier; full notices in metadata) | [0.7.2](https://pypi.org/pypi/isodate/0.7.2/json) |
| `isoduration` | `20.11.0` | ISC License (ISCL) | [20.11.0](https://pypi.org/pypi/isoduration/20.11.0/json) |
| `itsdangerous` | `2.2.0` | BSD License | [2.2.0](https://pypi.org/pypi/itsdangerous/2.2.0/json) |
| `jaxtyping` | `0.3.3` | MIT License (classifier; full notices in metadata) | [0.3.3](https://pypi.org/pypi/jaxtyping/0.3.3/json) |
| `jedi` | `0.19.2` | MIT | [0.19.2](https://pypi.org/pypi/jedi/0.19.2/json) |
| `jinja2` | `3.1.6` | BSD License | [3.1.6](https://pypi.org/pypi/jinja2/3.1.6/json) |
| `jiter` | `0.12.0` | MIT License | [0.12.0](https://pypi.org/pypi/jiter/0.12.0/json) |
| `jmespath` | `0.10.0`, `1.0.1` | MIT | [0.10.0](https://pypi.org/pypi/jmespath/0.10.0/json), [1.0.1](https://pypi.org/pypi/jmespath/1.0.1/json) |
| `joblib` | `1.4.2`, `1.5.2` | BSD 3-Clause | [1.4.2](https://pypi.org/pypi/joblib/1.4.2/json), [1.5.2](https://pypi.org/pypi/joblib/1.5.2/json) |
| `joblib` | `1.5.3` | BSD-3-Clause | [1.5.3](https://pypi.org/pypi/joblib/1.5.3/json) |
| `json5` | `0.12.1` | Apache Software License (classifier; full notices in metadata) | [0.12.1](https://pypi.org/pypi/json5/0.12.1/json) |
| `jsonlines` | `4.0.0` | BSD | [4.0.0](https://pypi.org/pypi/jsonlines/4.0.0/json) |
| `jsonpickle` | `3.0.4` | BSD License | [3.0.4](https://pypi.org/pypi/jsonpickle/3.0.4/json) |
| `jsonpointer` | `2.4` | Modified BSD License | [2.4](https://pypi.org/pypi/jsonpointer/2.4/json) |
| `jsonschema` | `4.22.0`, `4.23.0`, `4.25.1` | MIT | [4.22.0](https://pypi.org/pypi/jsonschema/4.22.0/json), [4.23.0](https://pypi.org/pypi/jsonschema/4.23.0/json), [4.25.1](https://pypi.org/pypi/jsonschema/4.25.1/json) |
| `jsonschema-specifications` | `2023.12.1`, `2025.9.1` | MIT | [2023.12.1](https://pypi.org/pypi/jsonschema-specifications/2023.12.1/json), [2025.9.1](https://pypi.org/pypi/jsonschema-specifications/2025.9.1/json) |
| `jupyter` | `1.1.1` | BSD | [1.1.1](https://pypi.org/pypi/jupyter/1.1.1/json) |
| `jupyter-client` | `7.4.9` | BSD License (classifier; full notices in metadata) | [7.4.9](https://pypi.org/pypi/jupyter-client/7.4.9/json) |
| `jupyter-console` | `6.6.3` | BSD License (classifier; full notices in metadata) | [6.6.3](https://pypi.org/pypi/jupyter-console/6.6.3/json) |
| `jupyter-core` | `5.8.1`, `5.9.1` | BSD-3-Clause | [5.8.1](https://pypi.org/pypi/jupyter-core/5.8.1/json), [5.9.1](https://pypi.org/pypi/jupyter-core/5.9.1/json) |
| `jupyter-events` | `0.12.0` | BSD License (classifier; full notices in metadata) | [0.12.0](https://pypi.org/pypi/jupyter-events/0.12.0/json) |
| `jupyter-lsp` | `2.3.0` | BSD-3-Clause | [2.3.0](https://pypi.org/pypi/jupyter-lsp/2.3.0/json) |
| `jupyter-server` | `2.17.0` | BSD License (classifier; full notices in metadata) | [2.17.0](https://pypi.org/pypi/jupyter-server/2.17.0/json) |
| `jupyter-server-terminals` | `0.5.3` | BSD License (classifier; full notices in metadata) | [0.5.3](https://pypi.org/pypi/jupyter-server-terminals/0.5.3/json) |
| `jupyterlab` | `4.5.0` | BSD License (classifier; full notices in metadata) | [4.5.0](https://pypi.org/pypi/jupyterlab/4.5.0/json) |
| `jupyterlab-pygments` | `0.3.0` | BSD License (classifier; full notices in metadata) | [0.3.0](https://pypi.org/pypi/jupyterlab-pygments/0.3.0/json) |
| `jupyterlab-server` | `2.28.0` | BSD License (classifier; full notices in metadata) | [2.28.0](https://pypi.org/pypi/jupyterlab-server/2.28.0/json) |
| `jupyterlab-widgets` | `3.0.16` | BSD License (classifier; full notices in metadata) | [3.0.16](https://pypi.org/pypi/jupyterlab-widgets/3.0.16/json) |
| `jupytext` | `1.17.1`, `1.19.1` | MIT License (classifier; full notices in metadata) | [1.17.1](https://pypi.org/pypi/jupytext/1.17.1/json), [1.19.1](https://pypi.org/pypi/jupytext/1.19.1/json) |
| `kaolin` | `0.17.0` | Unverified: exact release metadata unavailable | [0.17.0](https://pypi.org/pypi/kaolin/0.17.0/json), [upstream](https://pypi.org/project/kaolin/) |
| `kiwisolver` | `1.4.7`, `1.4.9` | BSD License (classifier; full notices in metadata) | [1.4.7](https://pypi.org/pypi/kiwisolver/1.4.7/json), [1.4.9](https://pypi.org/pypi/kiwisolver/1.4.9/json) |
| `lark` | `1.2.2`, `1.3.1` | MIT | [1.2.2](https://pypi.org/pypi/lark/1.2.2/json), [1.3.1](https://pypi.org/pypi/lark/1.3.1/json) |
| `lazy-loader` | `0.4` | BSD License (classifier; full notices in metadata) | [0.4](https://pypi.org/pypi/lazy-loader/0.4/json) |
| `libcst` | `1.8.6` | MIT License (classifier; full notices in metadata) | [1.8.6](https://pypi.org/pypi/libcst/1.8.6/json) |
| `librosa` | `0.10.1` | ISC | [0.10.1](https://pypi.org/pypi/librosa/0.10.1/json) |
| `lightning` | `2.3.3`, `2.6.0` | Apache-2.0 | [2.3.3](https://pypi.org/pypi/lightning/2.3.3/json), [2.6.0](https://pypi.org/pypi/lightning/2.6.0/json) |
| `lightning-lite` | `1.8.6` | Apache-2.0 | [1.8.6](https://pypi.org/pypi/lightning-lite/1.8.6/json) |
| `lightning-utilities` | `0.11.9`, `0.15.2` | Apache-2.0 | [0.11.9](https://pypi.org/pypi/lightning-utilities/0.11.9/json), [0.15.2](https://pypi.org/pypi/lightning-utilities/0.15.2/json) |
| `llguidance` | `0.7.30` | MIT | [0.7.30](https://pypi.org/pypi/llguidance/0.7.30/json) |
| `llvmlite` | `0.41.1`, `0.44.0`, `0.45.1` | BSD | [0.41.1](https://pypi.org/pypi/llvmlite/0.41.1/json), [0.44.0](https://pypi.org/pypi/llvmlite/0.44.0/json), [0.45.1](https://pypi.org/pypi/llvmlite/0.45.1/json) |
| `lm-format-enforcer` | `0.10.12` | MIT | [0.10.12](https://pypi.org/pypi/lm-format-enforcer/0.10.12/json) |
| `lmdb` | `2.2.0` | OLDAP-2.8 | [2.2.0](https://pypi.org/pypi/lmdb/2.2.0/json) |
| `loguru` | `0.7.3` | MIT License | [0.7.3](https://pypi.org/pypi/loguru/0.7.3/json) |
| `loguru` | `0.7.2` | MIT license | [0.7.2](https://pypi.org/pypi/loguru/0.7.2/json) |
| `lovely-numpy` | `0.2.18` | MIT License | [0.2.18](https://pypi.org/pypi/lovely-numpy/0.2.18/json) |
| `lovely-tensors` | `0.1.20` | MIT License | [0.1.20](https://pypi.org/pypi/lovely-tensors/0.1.20/json) |
| `lxml` | `6.0.2` | BSD-3-Clause | [6.0.2](https://pypi.org/pypi/lxml/6.0.2/json) |
| `mako` | `1.3.10` | MIT | [1.3.10](https://pypi.org/pypi/mako/1.3.10/json) |
| `markdown` | `3.7` | BSD License (classifier; full notices in metadata) | [3.7](https://pypi.org/pypi/markdown/3.7/json) |
| `markdown` | `3.10` | BSD-3-Clause | [3.10](https://pypi.org/pypi/markdown/3.10/json) |
| `markdown-it-py` | `3.0.0`, `4.0.0` | MIT License | [3.0.0](https://pypi.org/pypi/markdown-it-py/3.0.0/json), [4.0.0](https://pypi.org/pypi/markdown-it-py/4.0.0/json) |
| `markupsafe` | `2.1.5`, `3.0.3` | BSD-3-Clause | [2.1.5](https://pypi.org/pypi/markupsafe/2.1.5/json), [3.0.3](https://pypi.org/pypi/markupsafe/3.0.3/json) |
| `marshmallow` | `3.26.1` | MIT License | [3.26.1](https://pypi.org/pypi/marshmallow/3.26.1/json) |
| `matplotlib` | `3.5.3`, `3.7.5` | PSF | [3.5.3](https://pypi.org/pypi/matplotlib/3.5.3/json), [3.7.5](https://pypi.org/pypi/matplotlib/3.7.5/json) |
| `matplotlib` | `3.10.6`, `3.10.7`, `3.10.8`, `3.9.4` | Python Software Foundation License (classifier; full notices in metadata) | [3.10.6](https://pypi.org/pypi/matplotlib/3.10.6/json), [3.10.7](https://pypi.org/pypi/matplotlib/3.10.7/json), [3.10.8](https://pypi.org/pypi/matplotlib/3.10.8/json), [3.9.4](https://pypi.org/pypi/matplotlib/3.9.4/json) |
| `matplotlib-inline` | `0.2.1` | Unspecified in release metadata; consult upstream | [0.2.1](https://pypi.org/pypi/matplotlib-inline/0.2.1/json), [upstream](https://github.com/ipython/matplotlib-inline) |
| `mccabe` | `0.7.0` | Expat license | [0.7.0](https://pypi.org/pypi/mccabe/0.7.0/json) |
| `mdit-py-plugins` | `0.4.2`, `0.5.0` | MIT License | [0.4.2](https://pypi.org/pypi/mdit-py-plugins/0.4.2/json), [0.5.0](https://pypi.org/pypi/mdit-py-plugins/0.5.0/json) |
| `mdurl` | `0.1.2` | MIT License | [0.1.2](https://pypi.org/pypi/mdurl/0.1.2/json) |
| `mistral-common` | `1.8.6` | Apache-2.0 | [1.8.6](https://pypi.org/pypi/mistral-common/1.8.6/json) |
| `mistune` | `3.1.4` | BSD-3-Clause | [3.1.4](https://pypi.org/pypi/mistune/3.1.4/json) |
| `mock` | `4.0.3` | BSD License | [4.0.3](https://pypi.org/pypi/mock/4.0.3/json) |
| `moderngl` | `5.12.0` | MIT | [5.12.0](https://pypi.org/pypi/moderngl/5.12.0/json) |
| `moreorless` | `0.5.0` | MIT | [0.5.0](https://pypi.org/pypi/moreorless/0.5.0/json) |
| `mosaicml-streaming` | `0.7.5` | Unspecified in release metadata; consult upstream | [0.7.5](https://pypi.org/pypi/mosaicml-streaming/0.7.5/json), [upstream](https://github.com/mosaicml/streaming/) |
| `moviepy` | `1.0.3` | MIT License | [1.0.3](https://pypi.org/pypi/moviepy/1.0.3/json) |
| `mpmath` | `1.3.0` | BSD | [1.3.0](https://pypi.org/pypi/mpmath/1.3.0/json) |
| `msal` | `1.34.0` | MIT | [1.34.0](https://pypi.org/pypi/msal/1.34.0/json) |
| `msal-extensions` | `1.3.1` | MIT License | [1.3.1](https://pypi.org/pypi/msal-extensions/1.3.1/json) |
| `msgpack` | `1.1.2` | Apache-2.0 | [1.1.2](https://pypi.org/pypi/msgpack/1.1.2/json) |
| `msgspec` | `0.20.0` | BSD-3-Clause | [0.20.0](https://pypi.org/pypi/msgspec/0.20.0/json) |
| `mujoco` | `3.2.3` | Apache License 2.0 | [3.2.3](https://pypi.org/pypi/mujoco/3.2.3/json) |
| `mujoco` | `3.5.0` | Apache-2.0 | [3.5.0](https://pypi.org/pypi/mujoco/3.5.0/json) |
| `multicoretsne` | `0.1` | Unspecified in release metadata; consult upstream | [0.1](https://pypi.org/pypi/multicoretsne/0.1/json), [upstream](https://github.com/DmitryUlyanov/Multicore-TSNE) |
| `multidict` | `6.1.0` | Apache 2 | [6.1.0](https://pypi.org/pypi/multidict/6.1.0/json) |
| `multidict` | `6.7.0` | Apache License 2.0 | [6.7.0](https://pypi.org/pypi/multidict/6.7.0/json) |
| `multiprocess` | `0.70.15` | BSD-3-Clause | [0.70.15](https://pypi.org/pypi/multiprocess/0.70.15/json) |
| `mypy-extensions` | `1.1.0` | Unspecified in release metadata; consult upstream | [1.1.0](https://pypi.org/pypi/mypy-extensions/1.1.0/json), [upstream](https://github.com/python/mypy_extensions) |
| `narwhals` | `1.42.1`, `2.13.0` | MIT License | [1.42.1](https://pypi.org/pypi/narwhals/1.42.1/json), [2.13.0](https://pypi.org/pypi/narwhals/2.13.0/json) |
| `natsort` | `8.4.0` | MIT | [8.4.0](https://pypi.org/pypi/natsort/8.4.0/json) |
| `nbclient` | `0.10.2` | BSD License (classifier; full notices in metadata) | [0.10.2](https://pypi.org/pypi/nbclient/0.10.2/json) |
| `nbconvert` | `7.16.6` | BSD License (classifier; full notices in metadata) | [7.16.6](https://pypi.org/pypi/nbconvert/7.16.6/json) |
| `nbformat` | `5.10.4` | BSD License (classifier; full notices in metadata) | [5.10.4](https://pypi.org/pypi/nbformat/5.10.4/json) |
| `nest-asyncio` | `1.6.0` | BSD | [1.6.0](https://pypi.org/pypi/nest-asyncio/1.6.0/json) |
| `networkx` | `2.2` | BSD | [2.2](https://pypi.org/pypi/networkx/2.2/json) |
| `networkx` | `3.1`, `3.4.2` | BSD License | [3.1](https://pypi.org/pypi/networkx/3.1/json), [3.4.2](https://pypi.org/pypi/networkx/3.4.2/json) |
| `networkx` | `3.6`, `3.6.1` | BSD-3-Clause | [3.6](https://pypi.org/pypi/networkx/3.6/json), [3.6.1](https://pypi.org/pypi/networkx/3.6.1/json) |
| `ninja` | `1.13.0` | Apache Software License; BSD License | [1.13.0](https://pypi.org/pypi/ninja/1.13.0/json) |
| `notebook` | `7.5.0` | BSD License (classifier; full notices in metadata) | [7.5.0](https://pypi.org/pypi/notebook/7.5.0/json) |
| `notebook-shim` | `0.2.4` | BSD License (classifier; full notices in metadata) | [0.2.4](https://pypi.org/pypi/notebook-shim/0.2.4/json) |
| `numba` | `0.58.1`, `0.61.2`, `0.62.1` | BSD | [0.58.1](https://pypi.org/pypi/numba/0.58.1/json), [0.61.2](https://pypi.org/pypi/numba/0.61.2/json), [0.62.1](https://pypi.org/pypi/numba/0.62.1/json) |
| `numpy` | `1.22.4` | BSD | [1.22.4](https://pypi.org/pypi/numpy/1.22.4/json) |
| `numpy` | `1.26.4`, `2.0.2` | BSD License (classifier; full notices in metadata) | [1.26.4](https://pypi.org/pypi/numpy/1.26.4/json), [2.0.2](https://pypi.org/pypi/numpy/2.0.2/json) |
| `numpy` | `1.24.4` | BSD-3-Clause | [1.24.4](https://pypi.org/pypi/numpy/1.24.4/json) |
| `numpy-quaternion` | `2024.0.13` | MIT | [2024.0.13](https://pypi.org/pypi/numpy-quaternion/2024.0.13/json) |
| `numpy-quaternion` | `2023.0.4` | Unspecified in release metadata; consult upstream | [2023.0.4](https://pypi.org/pypi/numpy-quaternion/2023.0.4/json), [upstream](https://github.com/moble/quaternion) |
| `nvidia-cublas-cu11` | `11.10.3.66` | NVIDIA Proprietary Software | [11.10.3.66](https://pypi.org/pypi/nvidia-cublas-cu11/11.10.3.66/json) |
| `nvidia-cublas-cu12` | `12.1.3.1`, `12.8.3.14` | NVIDIA Proprietary Software | [12.1.3.1](https://pypi.org/pypi/nvidia-cublas-cu12/12.1.3.1/json), [12.8.3.14](https://pypi.org/pypi/nvidia-cublas-cu12/12.8.3.14/json) |
| `nvidia-cuda-cupti-cu12` | `12.1.105`, `12.8.57` | NVIDIA Proprietary Software | [12.1.105](https://pypi.org/pypi/nvidia-cuda-cupti-cu12/12.1.105/json), [12.8.57](https://pypi.org/pypi/nvidia-cuda-cupti-cu12/12.8.57/json) |
| `nvidia-cuda-nvcc-cu12` | `12.1.105` | NVIDIA Proprietary Software | [12.1.105](https://pypi.org/pypi/nvidia-cuda-nvcc-cu12/12.1.105/json) |
| `nvidia-cuda-nvrtc-cu11` | `11.7.99` | NVIDIA Proprietary Software | [11.7.99](https://pypi.org/pypi/nvidia-cuda-nvrtc-cu11/11.7.99/json) |
| `nvidia-cuda-nvrtc-cu12` | `12.1.105`, `12.8.61` | NVIDIA Proprietary Software | [12.1.105](https://pypi.org/pypi/nvidia-cuda-nvrtc-cu12/12.1.105/json), [12.8.61](https://pypi.org/pypi/nvidia-cuda-nvrtc-cu12/12.8.61/json) |
| `nvidia-cuda-runtime-cu11` | `11.7.99` | NVIDIA Proprietary Software | [11.7.99](https://pypi.org/pypi/nvidia-cuda-runtime-cu11/11.7.99/json) |
| `nvidia-cuda-runtime-cu12` | `12.1.105`, `12.8.57` | NVIDIA Proprietary Software | [12.1.105](https://pypi.org/pypi/nvidia-cuda-runtime-cu12/12.1.105/json), [12.8.57](https://pypi.org/pypi/nvidia-cuda-runtime-cu12/12.8.57/json) |
| `nvidia-cudnn-cu11` | `8.5.0.96` | NVIDIA Proprietary Software | [8.5.0.96](https://pypi.org/pypi/nvidia-cudnn-cu11/8.5.0.96/json) |
| `nvidia-cudnn-cu12` | `8.9.2.26`, `9.1.0.70`, `9.7.1.26` | NVIDIA Proprietary Software | [8.9.2.26](https://pypi.org/pypi/nvidia-cudnn-cu12/8.9.2.26/json), [9.1.0.70](https://pypi.org/pypi/nvidia-cudnn-cu12/9.1.0.70/json), [9.7.1.26](https://pypi.org/pypi/nvidia-cudnn-cu12/9.7.1.26/json) |
| `nvidia-cufft-cu12` | `11.0.2.54`, `11.3.3.41` | NVIDIA Proprietary Software | [11.0.2.54](https://pypi.org/pypi/nvidia-cufft-cu12/11.0.2.54/json), [11.3.3.41](https://pypi.org/pypi/nvidia-cufft-cu12/11.3.3.41/json) |
| `nvidia-cufile-cu12` | `1.13.0.11`, `1.13.1.3` | NVIDIA Proprietary Software | [1.13.0.11](https://pypi.org/pypi/nvidia-cufile-cu12/1.13.0.11/json), [1.13.1.3](https://pypi.org/pypi/nvidia-cufile-cu12/1.13.1.3/json) |
| `nvidia-curand-cu12` | `10.3.2.106`, `10.3.9.55` | NVIDIA Proprietary Software | [10.3.2.106](https://pypi.org/pypi/nvidia-curand-cu12/10.3.2.106/json), [10.3.9.55](https://pypi.org/pypi/nvidia-curand-cu12/10.3.9.55/json) |
| `nvidia-cusolver-cu12` | `11.4.5.107`, `11.7.2.55` | NVIDIA Proprietary Software | [11.4.5.107](https://pypi.org/pypi/nvidia-cusolver-cu12/11.4.5.107/json), [11.7.2.55](https://pypi.org/pypi/nvidia-cusolver-cu12/11.7.2.55/json) |
| `nvidia-cusparse-cu12` | `12.1.0.106`, `12.5.7.53` | NVIDIA Proprietary Software | [12.1.0.106](https://pypi.org/pypi/nvidia-cusparse-cu12/12.1.0.106/json), [12.5.7.53](https://pypi.org/pypi/nvidia-cusparse-cu12/12.5.7.53/json) |
| `nvidia-cusparselt-cu12` | `0.6.3`, `0.7.1` | NVIDIA Proprietary Software | [0.6.3](https://pypi.org/pypi/nvidia-cusparselt-cu12/0.6.3/json), [0.7.1](https://pypi.org/pypi/nvidia-cusparselt-cu12/0.7.1/json) |
| `nvidia-ml-py` | `13.580.82`, `13.590.44` | BSD | [13.580.82](https://pypi.org/pypi/nvidia-ml-py/13.580.82/json), [13.590.44](https://pypi.org/pypi/nvidia-ml-py/13.590.44/json) |
| `nvidia-nccl-cu12` | `2.26.2` | BSD-3-Clause | [2.26.2](https://pypi.org/pypi/nvidia-nccl-cu12/2.26.2/json) |
| `nvidia-nccl-cu12` | `2.19.3`, `2.20.5`, `2.21.5` | NVIDIA Proprietary Software | [2.19.3](https://pypi.org/pypi/nvidia-nccl-cu12/2.19.3/json), [2.20.5](https://pypi.org/pypi/nvidia-nccl-cu12/2.20.5/json), [2.21.5](https://pypi.org/pypi/nvidia-nccl-cu12/2.21.5/json) |
| `nvidia-nvjitlink-cu12` | `12.9.86` | LicenseRef-NVIDIA-Proprietary | [12.9.86](https://pypi.org/pypi/nvidia-nvjitlink-cu12/12.9.86/json) |
| `nvidia-nvjitlink-cu12` | `12.8.61` | NVIDIA Proprietary Software | [12.8.61](https://pypi.org/pypi/nvidia-nvjitlink-cu12/12.8.61/json) |
| `nvidia-nvshmem-cu12` | `3.3.20` | BSD-3-Clause | [3.3.20](https://pypi.org/pypi/nvidia-nvshmem-cu12/3.3.20/json) |
| `nvidia-nvtx-cu12` | `12.8.55` | Apache 2.0 | [12.8.55](https://pypi.org/pypi/nvidia-nvtx-cu12/12.8.55/json) |
| `nvidia-nvtx-cu12` | `12.1.105` | NVIDIA Proprietary Software | [12.1.105](https://pypi.org/pypi/nvidia-nvtx-cu12/12.1.105/json) |
| `nvidia-pyindex` | `1.0.9` | Apache2 | [1.0.9](https://pypi.org/pypi/nvidia-pyindex/1.0.9/json) |
| `oauthlib` | `3.3.1` | BSD-3-Clause | [3.3.1](https://pypi.org/pypi/oauthlib/3.3.1/json) |
| `objsize` | `0.7.0` | BSD-3-Clause | [0.7.0](https://pypi.org/pypi/objsize/0.7.0/json) |
| `oci` | `2.164.0` | Universal Permissive License 1.0 or Apache License 2.0 | [2.164.0](https://pypi.org/pypi/oci/2.164.0/json) |
| `omegaconf` | `2.1.2`, `2.3.0` | BSD License | [2.1.2](https://pypi.org/pypi/omegaconf/2.1.2/json), [2.3.0](https://pypi.org/pypi/omegaconf/2.3.0/json) |
| `open3d` | `0.18.0`, `0.19.0` | MIT | [0.18.0](https://pypi.org/pypi/open3d/0.18.0/json), [0.19.0](https://pypi.org/pypi/open3d/0.19.0/json) |
| `openai` | `2.9.0` | Apache-2.0 | [2.9.0](https://pypi.org/pypi/openai/2.9.0/json) |
| `opencv-python` | `4.13.0.92`, `4.9.0.80` | Apache 2.0 | [4.13.0.92](https://pypi.org/pypi/opencv-python/4.13.0.92/json), [4.9.0.80](https://pypi.org/pypi/opencv-python/4.9.0.80/json) |
| `opencv-python` | `4.6.0.66` | MIT | [4.6.0.66](https://pypi.org/pypi/opencv-python/4.6.0.66/json) |
| `opencv-python-headless` | `4.11.0.86` | Apache 2.0 | [4.11.0.86](https://pypi.org/pypi/opencv-python-headless/4.11.0.86/json) |
| `openexr` | `3.3.3` | Unspecified in release metadata; consult upstream | [3.3.3](https://pypi.org/pypi/openexr/3.3.3/json), [upstream](https://github.com/AcademySoftwareFoundation/OpenEXR) |
| `opentelemetry-api` | `1.39.0` | Apache-2.0 | [1.39.0](https://pypi.org/pypi/opentelemetry-api/1.39.0/json) |
| `opentelemetry-exporter-otlp` | `1.39.0` | Apache-2.0 | [1.39.0](https://pypi.org/pypi/opentelemetry-exporter-otlp/1.39.0/json) |
| `opentelemetry-exporter-otlp-proto-common` | `1.39.0` | Apache-2.0 | [1.39.0](https://pypi.org/pypi/opentelemetry-exporter-otlp-proto-common/1.39.0/json) |
| `opentelemetry-exporter-otlp-proto-grpc` | `1.39.0` | Apache-2.0 | [1.39.0](https://pypi.org/pypi/opentelemetry-exporter-otlp-proto-grpc/1.39.0/json) |
| `opentelemetry-exporter-otlp-proto-http` | `1.39.0` | Apache-2.0 | [1.39.0](https://pypi.org/pypi/opentelemetry-exporter-otlp-proto-http/1.39.0/json) |
| `opentelemetry-proto` | `1.39.0` | Apache-2.0 | [1.39.0](https://pypi.org/pypi/opentelemetry-proto/1.39.0/json) |
| `opentelemetry-sdk` | `1.39.0` | Apache-2.0 | [1.39.0](https://pypi.org/pypi/opentelemetry-sdk/1.39.0/json) |
| `opentelemetry-semantic-conventions` | `0.60b0` | Apache-2.0 | [0.60b0](https://pypi.org/pypi/opentelemetry-semantic-conventions/0.60b0/json) |
| `opentelemetry-semantic-conventions-ai` | `0.4.13` | Apache-2.0 | [0.4.13](https://pypi.org/pypi/opentelemetry-semantic-conventions-ai/0.4.13/json) |
| `optimum` | `1.18.1` | Apache | [1.18.1](https://pypi.org/pypi/optimum/1.18.1/json) |
| `optree` | `0.14.1` | Apache License, Version 2.0 | [0.14.1](https://pypi.org/pypi/optree/0.14.1/json) |
| `optree` | `0.18.0` | Apache-2.0 | [0.18.0](https://pypi.org/pypi/optree/0.18.0/json) |
| `orjson` | `3.10.0`, `3.11.5` | Apache-2.0 OR MIT | [3.10.0](https://pypi.org/pypi/orjson/3.10.0/json), [3.11.5](https://pypi.org/pypi/orjson/3.11.5/json) |
| `oss2` | `2.17.0` | MIT License | [2.17.0](https://pypi.org/pypi/oss2/2.17.0/json) |
| `outlines` | `0.1.11` | Apache-2.0 | [0.1.11](https://pypi.org/pypi/outlines/0.1.11/json) |
| `outlines-core` | `0.1.26` | Apache-2.0 | [0.1.26](https://pypi.org/pypi/outlines-core/0.1.26/json) |
| `overrides` | `7.7.0` | Apache License, Version 2.0 | [7.7.0](https://pypi.org/pypi/overrides/7.7.0/json) |
| `packaging` | `24.0`, `24.2`, `25.0` | Apache Software License; BSD License | [24.0](https://pypi.org/pypi/packaging/24.0/json), [24.2](https://pypi.org/pypi/packaging/24.2/json), [25.0](https://pypi.org/pypi/packaging/25.0/json) |
| `packaging` | `26.0` | Apache-2.0 OR BSD-2-Clause | [26.0](https://pypi.org/pypi/packaging/26.0/json) |
| `panda3d` | `1.10.15` | Modified BSD License | [1.10.15](https://pypi.org/pypi/panda3d/1.10.15/json) |
| `panda3d-gltf` | `1.2.1` | BSD-3-Clause | [1.2.1](https://pypi.org/pypi/panda3d-gltf/1.2.1/json) |
| `panda3d-simplepbr` | `0.13.1` | BSD-3-Clause | [0.13.1](https://pypi.org/pypi/panda3d-simplepbr/0.13.1/json) |
| `pandas` | `2.0.3`, `2.3.3` | BSD License (classifier; full notices in metadata) | [2.0.3](https://pypi.org/pypi/pandas/2.0.3/json), [2.3.3](https://pypi.org/pypi/pandas/2.3.3/json) |
| `pandocfilters` | `1.5.1` | BSD-3-Clause | [1.5.1](https://pypi.org/pypi/pandocfilters/1.5.1/json) |
| `paramiko` | `3.5.1` | LGPL | [3.5.1](https://pypi.org/pypi/paramiko/3.5.1/json) |
| `parso` | `0.8.5` | MIT | [0.8.5](https://pypi.org/pypi/parso/0.8.5/json) |
| `partial-json-parser` | `0.2.1.1.post7` | MIT | [0.2.1.1.post7](https://pypi.org/pypi/partial-json-parser/0.2.1.1.post7/json) |
| `pathos` | `0.3.1` | BSD-3-Clause | [0.3.1](https://pypi.org/pypi/pathos/0.3.1/json) |
| `pathspec` | `0.12.1` | Mozilla Public License 2.0 (MPL 2.0) | [0.12.1](https://pypi.org/pypi/pathspec/0.12.1/json) |
| `pathtools` | `0.1.2` | MIT License | [0.1.2](https://pypi.org/pypi/pathtools/0.1.2/json) |
| `pccm` | `0.4.16` | MIT | [0.4.16](https://pypi.org/pypi/pccm/0.4.16/json) |
| `pdoc3` | `0.10.0` | AGPL-3.0 | [0.10.0](https://pypi.org/pypi/pdoc3/0.10.0/json) |
| `peft` | `0.10.0`, `0.18.0`, `0.18.1` | Apache | [0.10.0](https://pypi.org/pypi/peft/0.10.0/json), [0.18.0](https://pypi.org/pypi/peft/0.18.0/json), [0.18.1](https://pypi.org/pypi/peft/0.18.1/json) |
| `pexpect` | `4.9.0` | ISC license | [4.9.0](https://pypi.org/pypi/pexpect/4.9.0/json) |
| `pfzy` | `0.3.4` | MIT | [0.3.4](https://pypi.org/pypi/pfzy/0.3.4/json) |
| `pillow` | `10.4.0` | HPND | [10.4.0](https://pypi.org/pypi/pillow/10.4.0/json) |
| `pillow` | `11.1.0`, `11.3.0`, `12.0.0` | MIT-CMU | [11.1.0](https://pypi.org/pypi/pillow/11.1.0/json), [11.3.0](https://pypi.org/pypi/pillow/11.3.0/json), [12.0.0](https://pypi.org/pypi/pillow/12.0.0/json) |
| `pip-system-certs` | `4.0` | BSD | [4.0](https://pypi.org/pypi/pip-system-certs/4.0/json) |
| `pipdeptree` | `2.28.0`, `2.30.0` | MIT | [2.28.0](https://pypi.org/pypi/pipdeptree/2.28.0/json), [2.30.0](https://pypi.org/pypi/pipdeptree/2.30.0/json) |
| `pkgutil-resolve-name` | `1.3.10` | MIT License | [1.3.10](https://pypi.org/pypi/pkgutil-resolve-name/1.3.10/json) |
| `platformdirs` | `4.5.0`, `4.5.1` | MIT | [4.5.0](https://pypi.org/pypi/platformdirs/4.5.0/json), [4.5.1](https://pypi.org/pypi/platformdirs/4.5.1/json) |
| `platformdirs` | `4.3.6` | MIT License | [4.3.6](https://pypi.org/pypi/platformdirs/4.3.6/json) |
| `plotly` | `6.5.0`, `6.5.2` | MIT License (classifier; full notices in metadata) | [6.5.0](https://pypi.org/pypi/plotly/6.5.0/json), [6.5.2](https://pypi.org/pypi/plotly/6.5.2/json) |
| `pluggy` | `1.5.0`, `1.6.0` | MIT | [1.5.0](https://pypi.org/pypi/pluggy/1.5.0/json), [1.6.0](https://pypi.org/pypi/pluggy/1.6.0/json) |
| `plyfile` | `1.1.3` | GNU General Public License v3 or later (GPLv3+) (classifier; full notices in metadata) | [1.1.3](https://pypi.org/pypi/plyfile/1.1.3/json) |
| `point-cloud-utils` | `0.29.5` | GNU General Public License v3 (GPLv3) | [0.29.5](https://pypi.org/pypi/point-cloud-utils/0.29.5/json) |
| `point-cloud-utils` | `0.34.0` | MIT License | [0.34.0](https://pypi.org/pypi/point-cloud-utils/0.34.0/json) |
| `polyscope` | `2.3.0` | MIT | [2.3.0](https://pypi.org/pypi/polyscope/2.3.0/json) |
| `polyscope` | `2.5.0` | MIT License (classifier; full notices in metadata) | [2.5.0](https://pypi.org/pypi/polyscope/2.5.0/json) |
| `pooch` | `1.8.2` | BSD-3-Clause | [1.8.2](https://pypi.org/pypi/pooch/1.8.2/json) |
| `portalocker` | `3.2.0` | BSD-3-Clause | [3.2.0](https://pypi.org/pypi/portalocker/3.2.0/json) |
| `pox` | `0.3.6` | BSD-3-Clause | [0.3.6](https://pypi.org/pypi/pox/0.3.6/json) |
| `ppft` | `1.7.7` | BSD-3-Clause | [1.7.7](https://pypi.org/pypi/ppft/1.7.7/json) |
| `proglog` | `0.1.12` | MIT | [0.1.12](https://pypi.org/pypi/proglog/0.1.12/json) |
| `prometheus-client` | `0.23.1` | Apache-2.0 AND BSD-2-Clause | [0.23.1](https://pypi.org/pypi/prometheus-client/0.23.1/json) |
| `prometheus-fastapi-instrumentator` | `7.1.0` | ISC | [7.1.0](https://pypi.org/pypi/prometheus-fastapi-instrumentator/7.1.0/json) |
| `promise` | `2.3` | MIT | [2.3](https://pypi.org/pypi/promise/2.3/json) |
| `prompt-toolkit` | `3.0.52` | BSD License | [3.0.52](https://pypi.org/pypi/prompt-toolkit/3.0.52/json) |
| `propcache` | `0.2.0`, `0.4.1` | Apache-2.0 | [0.2.0](https://pypi.org/pypi/propcache/0.2.0/json), [0.4.1](https://pypi.org/pypi/propcache/0.4.1/json) |
| `proto-plus` | `1.26.1` | Apache 2.0 | [1.26.1](https://pypi.org/pypi/proto-plus/1.26.1/json) |
| `protobuf` | `5.29.3`, `5.29.5`, `5.29.6`, `6.33.2` | 3-Clause BSD License | [5.29.3](https://pypi.org/pypi/protobuf/5.29.3/json), [5.29.5](https://pypi.org/pypi/protobuf/5.29.5/json), [5.29.6](https://pypi.org/pypi/protobuf/5.29.6/json), [6.33.2](https://pypi.org/pypi/protobuf/6.33.2/json) |
| `protobuf` | `3.20.3` | BSD-3-Clause | [3.20.3](https://pypi.org/pypi/protobuf/3.20.3/json) |
| `psutil` | `7.1.3`, `7.2.2` | BSD-3-Clause | [7.1.3](https://pypi.org/pypi/psutil/7.1.3/json), [7.2.2](https://pypi.org/pypi/psutil/7.2.2/json) |
| `ptyprocess` | `0.7.0` | ISC License (ISCL) | [0.7.0](https://pypi.org/pypi/ptyprocess/0.7.0/json) |
| `pure-eval` | `0.2.3` | MIT | [0.2.3](https://pypi.org/pypi/pure-eval/0.2.3/json) |
| `py-cpuinfo` | `9.0.0` | MIT | [9.0.0](https://pypi.org/pypi/py-cpuinfo/9.0.0/json) |
| `pyarrow` | `22.0.0` | Apache Software License | [22.0.0](https://pypi.org/pypi/pyarrow/22.0.0/json) |
| `pyasn1` | `0.6.1`, `0.6.2` | BSD-2-Clause | [0.6.1](https://pypi.org/pypi/pyasn1/0.6.1/json), [0.6.2](https://pypi.org/pypi/pyasn1/0.6.2/json) |
| `pyasn1-modules` | `0.4.2` | BSD | [0.4.2](https://pypi.org/pypi/pyasn1-modules/0.4.2/json) |
| `pybind11` | `3.0.1` | BSD-3-Clause | [3.0.1](https://pypi.org/pypi/pybind11/3.0.1/json) |
| `pybullet` | `3.2.7` | zlib | [3.2.7](https://pypi.org/pypi/pybullet/3.2.7/json) |
| `pycocotools` | `2.0.10`, `2.0.7` | FreeBSD | [2.0.10](https://pypi.org/pypi/pycocotools/2.0.10/json), [2.0.7](https://pypi.org/pypi/pycocotools/2.0.7/json) |
| `pycodestyle` | `2.11.1` | MIT | [2.11.1](https://pypi.org/pypi/pycodestyle/2.11.1/json) |
| `pycollada` | `0.6` | BSD | [0.6](https://pypi.org/pypi/pycollada/0.6/json) |
| `pycountry` | `24.6.1` | LGPL-2.1-only | [24.6.1](https://pypi.org/pypi/pycountry/24.6.1/json) |
| `pycparser` | `2.23` | BSD-3-Clause | [2.23](https://pypi.org/pypi/pycparser/2.23/json) |
| `pycryptodome` | `3.23.0` | BSD, Public Domain | [3.23.0](https://pypi.org/pypi/pycryptodome/3.23.0/json) |
| `pydantic` | `2.11.10`, `2.12.4` | MIT | [2.11.10](https://pypi.org/pypi/pydantic/2.11.10/json), [2.12.4](https://pypi.org/pypi/pydantic/2.12.4/json) |
| `pydantic` | `2.10.6` | MIT License | [2.10.6](https://pypi.org/pypi/pydantic/2.10.6/json) |
| `pydantic-core` | `2.27.2`, `2.33.2`, `2.41.5` | MIT | [2.27.2](https://pypi.org/pypi/pydantic-core/2.27.2/json), [2.33.2](https://pypi.org/pypi/pydantic-core/2.33.2/json), [2.41.5](https://pypi.org/pypi/pydantic-core/2.41.5/json) |
| `pydantic-extra-types` | `2.10.6` | MIT | [2.10.6](https://pypi.org/pypi/pydantic-extra-types/2.10.6/json) |
| `pydot` | `1.4.2`, `4.0.1` | MIT | [1.4.2](https://pypi.org/pypi/pydot/1.4.2/json), [4.0.1](https://pypi.org/pypi/pydot/4.0.1/json) |
| `pydub` | `0.25.1` | MIT | [0.25.1](https://pypi.org/pypi/pydub/0.25.1/json) |
| `pyflakes` | `3.2.0` | MIT | [3.2.0](https://pypi.org/pypi/pyflakes/3.2.0/json) |
| `pyglet` | `2.1.11`, `2.1.13` | Unspecified in release metadata; consult upstream | [2.1.11](https://pypi.org/pypi/pyglet/2.1.11/json), [2.1.13](https://pypi.org/pypi/pyglet/2.1.13/json), [upstream](https://pypi.org/project/pyglet/) |
| `pygltflib` | `1.16.5` | MIT License | [1.16.5](https://pypi.org/pypi/pygltflib/1.16.5/json) |
| `pygments` | `2.19.2` | BSD-2-Clause | [2.19.2](https://pypi.org/pypi/pygments/2.19.2/json) |
| `pyhash` | `0.9.3` | Apache Software License | [0.9.3](https://pypi.org/pypi/pyhash/0.9.3/json) |
| `pyjwt` | `2.10.1` | MIT | [2.10.1](https://pypi.org/pypi/pyjwt/2.10.1/json) |
| `pymeshfix` | `0.17.0`, `0.17.2` | GPLv3 | [0.17.0](https://pypi.org/pypi/pymeshfix/0.17.0/json), [0.17.2](https://pypi.org/pypi/pymeshfix/0.17.2/json) |
| `pymongo` | `4.6.3` | Apache Software License (classifier; full notices in metadata) | [4.6.3](https://pypi.org/pypi/pymongo/4.6.3/json) |
| `pymupdf` | `1.26.7` | Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial License | [1.26.7](https://pypi.org/pypi/pymupdf/1.26.7/json) |
| `pynacl` | `1.6.1` | Apache-2.0 | [1.6.1](https://pypi.org/pypi/pynacl/1.6.1/json) |
| `pynvml` | `13.0.1` | BSD-3-Clause | [13.0.1](https://pypi.org/pypi/pynvml/13.0.1/json) |
| `pyopengl` | `3.1.0` | BSD | [3.1.0](https://pypi.org/pypi/pyopengl/3.1.0/json) |
| `pyopengl` | `3.1.10` | BSD License | [3.1.10](https://pypi.org/pypi/pyopengl/3.1.10/json) |
| `pyopenssl` | `25.1.0` | Apache License, Version 2.0 | [25.1.0](https://pypi.org/pypi/pyopenssl/25.1.0/json) |
| `pyparsing` | `3.2.5`, `3.3.2` | MIT | [3.2.5](https://pypi.org/pypi/pyparsing/3.2.5/json), [3.3.2](https://pypi.org/pypi/pyparsing/3.3.2/json) |
| `pyparsing` | `3.1.4` | MIT License | [3.1.4](https://pypi.org/pypi/pyparsing/3.1.4/json) |
| `pyquaternion` | `0.9.9` | MIT | [0.9.9](https://pypi.org/pypi/pyquaternion/0.9.9/json) |
| `pyrender` | `0.1.45` | MIT License | [0.1.45](https://pypi.org/pypi/pyrender/0.1.45/json) |
| `pysocks` | `1.7.1` | BSD | [1.7.1](https://pypi.org/pypi/pysocks/1.7.1/json) |
| `pytest` | `8.1.1`, `8.3.5` | MIT | [8.1.1](https://pypi.org/pypi/pytest/8.1.1/json), [8.3.5](https://pypi.org/pypi/pytest/8.3.5/json) |
| `python-dateutil` | `2.9.0.post0` | Dual License (classifiers: Apache Software License; BSD License) | [2.9.0.post0](https://pypi.org/pypi/python-dateutil/2.9.0.post0/json) |
| `python-dotenv` | `1.2.1` | BSD-3-Clause | [1.2.1](https://pypi.org/pypi/python-dotenv/1.2.1/json) |
| `python-json-logger` | `4.0.0` | BSD-2-Clause License | [4.0.0](https://pypi.org/pypi/python-json-logger/4.0.0/json) |
| `python-multipart` | `0.0.20` | Apache Software License | [0.0.20](https://pypi.org/pypi/python-multipart/0.0.20/json) |
| `python-pycg` | `0.9.2` | MIT License | [0.9.2](https://pypi.org/pypi/python-pycg/0.9.2/json) |
| `python-snappy` | `0.7.3` | BSD | [0.7.3](https://pypi.org/pypi/python-snappy/0.7.3/json) |
| `pytorch-lightning` | `1.8.6`, `2.6.0` | Apache-2.0 | [1.8.6](https://pypi.org/pypi/pytorch-lightning/1.8.6/json), [2.6.0](https://pypi.org/pypi/pytorch-lightning/2.6.0/json) |
| `pytz` | `2023.4`, `2025.2` | MIT | [2023.4](https://pypi.org/pypi/pytz/2023.4/json), [2025.2](https://pypi.org/pypi/pytz/2025.2/json) |
| `pyvista` | `0.46.4` | MIT | [0.46.4](https://pypi.org/pypi/pyvista/0.46.4/json) |
| `pyyaml` | `6.0.3` | MIT | [6.0.3](https://pypi.org/pypi/pyyaml/6.0.3/json) |
| `pyzmq` | `27.1.0` | BSD License (classifier; full notices in metadata) | [27.1.0](https://pypi.org/pypi/pyzmq/27.1.0/json) |
| `qwen-vl-utils` | `0.0.14` | Apache-2.0 | [0.0.14](https://pypi.org/pypi/qwen-vl-utils/0.0.14/json) |
| `randomname` | `0.2.1` | MIT License | [0.2.1](https://pypi.org/pypi/randomname/0.2.1/json) |
| `ray` | `2.52.1` | Apache 2.0 | [2.52.1](https://pypi.org/pypi/ray/2.52.1/json) |
| `referencing` | `0.37.0` | MIT | [0.37.0](https://pypi.org/pypi/referencing/0.37.0/json) |
| `referencing` | `0.35.1` | MIT License | [0.35.1](https://pypi.org/pypi/referencing/0.35.1/json) |
| `regex` | `2024.11.6` | Apache Software License | [2024.11.6](https://pypi.org/pypi/regex/2024.11.6/json) |
| `regex` | `2025.11.3` | Apache-2.0 AND CNRI-Python | [2025.11.3](https://pypi.org/pypi/regex/2025.11.3/json) |
| `requests` | `2.32.3`, `2.32.4`, `2.32.5` | Apache-2.0 | [2.32.3](https://pypi.org/pypi/requests/2.32.3/json), [2.32.4](https://pypi.org/pypi/requests/2.32.4/json), [2.32.5](https://pypi.org/pypi/requests/2.32.5/json) |
| `requests-oauthlib` | `2.0.0` | ISC | [2.0.0](https://pypi.org/pypi/requests-oauthlib/2.0.0/json) |
| `retrying` | `1.4.2` | Apache-2.0 | [1.4.2](https://pypi.org/pypi/retrying/1.4.2/json) |
| `rfc3339-validator` | `0.1.4` | MIT license | [0.1.4](https://pypi.org/pypi/rfc3339-validator/0.1.4/json) |
| `rfc3986-validator` | `0.1.1` | MIT license | [0.1.1](https://pypi.org/pypi/rfc3986-validator/0.1.1/json) |
| `rich` | `14.2.0`, `14.3.2` | MIT | [14.2.0](https://pypi.org/pypi/rich/14.2.0/json), [14.3.2](https://pypi.org/pypi/rich/14.3.2/json) |
| `rich-toolkit` | `0.17.0` | MIT | [0.17.0](https://pypi.org/pypi/rich-toolkit/0.17.0/json) |
| `rignore` | `0.7.6` | MIT | [0.7.6](https://pypi.org/pypi/rignore/0.7.6/json) |
| `robomimic` | `0.2.0` | Unspecified in release metadata; consult upstream | [0.2.0](https://pypi.org/pypi/robomimic/0.2.0/json), [upstream](https://github.com/ARISE-Initiative/robomimic) |
| `robosuite` | `1.4.0` | Unspecified in release metadata; consult upstream | [1.4.0](https://pypi.org/pypi/robosuite/1.4.0/json), [upstream](https://github.com/ARISE-Initiative/robosuite) |
| `roma` | `1.5.1` | BSD License | [1.5.1](https://pypi.org/pypi/roma/1.5.1/json) |
| `rootutils` | `1.0.7` | MIT | [1.0.7](https://pypi.org/pypi/rootutils/1.0.7/json) |
| `rouge` | `1.0.1` | LICENCE.txt | [1.0.1](https://pypi.org/pypi/rouge/1.0.1/json) |
| `rpds-py` | `0.20.1`, `0.30.0` | MIT | [0.20.1](https://pypi.org/pypi/rpds-py/0.20.1/json), [0.30.0](https://pypi.org/pypi/rpds-py/0.30.0/json) |
| `rsa` | `4.9.1` | Apache-2.0 | [4.9.1](https://pypi.org/pypi/rsa/4.9.1/json) |
| `rtree` | `1.3.0` | MIT | [1.3.0](https://pypi.org/pypi/rtree/1.3.0/json) |
| `ruff` | `0.14.7`, `0.14.8` | MIT License | [0.14.7](https://pypi.org/pypi/ruff/0.14.7/json), [0.14.8](https://pypi.org/pypi/ruff/0.14.8/json) |
| `s3transfer` | `0.16.0` | Apache License 2.0 | [0.16.0](https://pypi.org/pypi/s3transfer/0.16.0/json) |
| `safehttpx` | `0.1.7` | MIT License | [0.1.7](https://pypi.org/pypi/safehttpx/0.1.7/json) |
| `safetensors` | `0.5.3`, `0.7.0` | Apache Software License | [0.5.3](https://pypi.org/pypi/safetensors/0.5.3/json), [0.7.0](https://pypi.org/pypi/safetensors/0.7.0/json) |
| `sagemaker` | `2.242.0` | Apache Software License | [2.242.0](https://pypi.org/pypi/sagemaker/2.242.0/json) |
| `sagemaker-core` | `1.0.71` | Apache Software License | [1.0.71](https://pypi.org/pypi/sagemaker-core/1.0.71/json) |
| `schema` | `0.7.8` | MIT | [0.7.8](https://pypi.org/pypi/schema/0.7.8/json) |
| `scikit-image` | `0.23.1`, `0.25.2` | BSD License (classifier; full notices in metadata) | [0.23.1](https://pypi.org/pypi/scikit-image/0.23.1/json), [0.25.2](https://pypi.org/pypi/scikit-image/0.25.2/json) |
| `scikit-learn` | `1.7.2`, `1.8.0` | BSD-3-Clause | [1.7.2](https://pypi.org/pypi/scikit-learn/1.7.2/json), [1.8.0](https://pypi.org/pypi/scikit-learn/1.8.0/json) |
| `scikit-learn` | `1.3.2` | new BSD | [1.3.2](https://pypi.org/pypi/scikit-learn/1.3.2/json) |
| `scipy` | `1.10.1`, `1.15.2`, `1.15.3`, `1.16.3` | BSD License (classifier; full notices in metadata) | [1.10.1](https://pypi.org/pypi/scipy/1.10.1/json), [1.15.2](https://pypi.org/pypi/scipy/1.15.2/json), [1.15.3](https://pypi.org/pypi/scipy/1.15.3/json), [1.16.3](https://pypi.org/pypi/scipy/1.16.3/json) |
| `scooby` | `0.11.0` | MIT License | [0.11.0](https://pypi.org/pypi/scooby/0.11.0/json) |
| `screeninfo` | `0.8.1` | MIT | [0.8.1](https://pypi.org/pypi/screeninfo/0.8.1/json) |
| `seaborn` | `0.13.2` | BSD License | [0.13.2](https://pypi.org/pypi/seaborn/0.13.2/json) |
| `semantic-version` | `2.10.0` | BSD | [2.10.0](https://pypi.org/pypi/semantic-version/2.10.0/json) |
| `send2trash` | `1.8.3` | BSD License | [1.8.3](https://pypi.org/pypi/send2trash/1.8.3/json) |
| `sentence-transformers` | `3.2.1` | Apache 2.0 | [3.2.1](https://pypi.org/pypi/sentence-transformers/3.2.1/json) |
| `sentence-transformers` | `2.6.1` | Apache License 2.0 | [2.6.1](https://pypi.org/pypi/sentence-transformers/2.6.1/json) |
| `sentencepiece` | `0.2.0` | Apache | [0.2.0](https://pypi.org/pypi/sentencepiece/0.2.0/json) |
| `sentencepiece` | `0.2.1` | Unspecified in release metadata; consult upstream | [0.2.1](https://pypi.org/pypi/sentencepiece/0.2.1/json), [upstream](https://github.com/google/sentencepiece) |
| `sentry-sdk` | `2.47.0`, `2.52.0` | MIT | [2.47.0](https://pypi.org/pypi/sentry-sdk/2.47.0/json), [2.52.0](https://pypi.org/pypi/sentry-sdk/2.52.0/json) |
| `setproctitle` | `1.3.7` | BSD-3-Clause | [1.3.7](https://pypi.org/pypi/setproctitle/1.3.7/json) |
| `shellingham` | `1.5.4` | ISC License | [1.5.4](https://pypi.org/pypi/shellingham/1.5.4/json) |
| `shortuuid` | `1.0.13` | BSD-3-Clause | [1.0.13](https://pypi.org/pypi/shortuuid/1.0.13/json) |
| `simplejson` | `3.19.2` | MIT License | [3.19.2](https://pypi.org/pypi/simplejson/3.19.2/json) |
| `six` | `1.17.0` | MIT | [1.17.0](https://pypi.org/pypi/six/1.17.0/json) |
| `smdebug-rulesconfig` | `1.0.1` | Apache License Version 2.0 | [1.0.1](https://pypi.org/pypi/smdebug-rulesconfig/1.0.1/json) |
| `smmap` | `5.0.2` | BSD-3-Clause | [5.0.2](https://pypi.org/pypi/smmap/5.0.2/json) |
| `smplx` | `0.1.28` | Unspecified in release metadata; consult upstream | [0.1.28](https://pypi.org/pypi/smplx/0.1.28/json), [upstream](http://smpl-x.is.tuebingen.mpg.de) |
| `sniffio` | `1.3.1` | MIT OR Apache-2.0 | [1.3.1](https://pypi.org/pypi/sniffio/1.3.1/json) |
| `soundfile` | `0.13.1` | BSD 3-Clause License | [0.13.1](https://pypi.org/pypi/soundfile/0.13.1/json) |
| `soupsieve` | `2.8` | MIT | [2.8](https://pypi.org/pypi/soupsieve/2.8/json) |
| `soxr` | `1.0.0` | LGPL-2.1-or-later | [1.0.0](https://pypi.org/pypi/soxr/1.0.0/json) |
| `spconv-cu121` | `2.3.8` | Apache License 2.0 | [2.3.8](https://pypi.org/pypi/spconv-cu121/2.3.8/json) |
| `stack-data` | `0.6.3` | MIT | [0.6.3](https://pypi.org/pypi/stack-data/0.6.3/json) |
| `starlette` | `0.50.0` | BSD-3-Clause | [0.50.0](https://pypi.org/pypi/starlette/0.50.0/json) |
| `stdlibs` | `2025.10.28` | MIT License | [2025.10.28](https://pypi.org/pypi/stdlibs/2025.10.28/json) |
| `submitit` | `1.5.3` | MIT License | [1.5.3](https://pypi.org/pypi/submitit/1.5.3/json) |
| `svgwrite` | `1.4.3` | MIT License | [1.4.3](https://pypi.org/pypi/svgwrite/1.4.3/json) |
| `sympy` | `1.13.1`, `1.13.3`, `1.14.0` | BSD | [1.13.1](https://pypi.org/pypi/sympy/1.13.1/json), [1.13.3](https://pypi.org/pypi/sympy/1.13.3/json), [1.14.0](https://pypi.org/pypi/sympy/1.14.0/json) |
| `tabulate` | `0.9.0` | MIT | [0.9.0](https://pypi.org/pypi/tabulate/0.9.0/json) |
| `tblib` | `3.2.2` | BSD-2-Clause | [3.2.2](https://pypi.org/pypi/tblib/3.2.2/json) |
| `tensorboard` | `2.14.0`, `2.16.2` | Apache 2.0 | [2.14.0](https://pypi.org/pypi/tensorboard/2.14.0/json), [2.16.2](https://pypi.org/pypi/tensorboard/2.16.2/json) |
| `tensorboard-data-server` | `0.7.2` | Apache 2.0 | [0.7.2](https://pypi.org/pypi/tensorboard-data-server/0.7.2/json) |
| `tensorboardx` | `2.6.4` | MIT | [2.6.4](https://pypi.org/pypi/tensorboardx/2.6.4/json) |
| `tensorboardx` | `2.6.2.2` | MIT license | [2.6.2.2](https://pypi.org/pypi/tensorboardx/2.6.2.2/json) |
| `termcolor` | `2.4.0`, `3.2.0` | MIT | [2.4.0](https://pypi.org/pypi/termcolor/2.4.0/json), [3.2.0](https://pypi.org/pypi/termcolor/3.2.0/json) |
| `terminado` | `0.18.1` | BSD License (classifier; full notices in metadata) | [0.18.1](https://pypi.org/pypi/terminado/0.18.1/json) |
| `texttable` | `1.7.0` | MIT | [1.7.0](https://pypi.org/pypi/texttable/1.7.0/json) |
| `thop` | `0.1.1.post2209072238` | MIT | [0.1.1.post2209072238](https://pypi.org/pypi/thop/0.1.1.post2209072238/json) |
| `threadpoolctl` | `3.5.0`, `3.6.0` | BSD-3-Clause | [3.5.0](https://pypi.org/pypi/threadpoolctl/3.5.0/json), [3.6.0](https://pypi.org/pypi/threadpoolctl/3.6.0/json) |
| `tifffile` | `2025.10.16`, `2025.12.12` | BSD-3-Clause | [2025.10.16](https://pypi.org/pypi/tifffile/2025.10.16/json), [2025.12.12](https://pypi.org/pypi/tifffile/2025.12.12/json) |
| `tiktoken` | `0.12.0` | MIT (license text) | [0.12.0](https://pypi.org/pypi/tiktoken/0.12.0/json) |
| `timm` | `0.9.16`, `1.0.22` | Apache-2.0 | [0.9.16](https://pypi.org/pypi/timm/0.9.16/json), [1.0.22](https://pypi.org/pypi/timm/1.0.22/json) |
| `tinycss2` | `1.4.0` | BSD License | [1.4.0](https://pypi.org/pypi/tinycss2/1.4.0/json) |
| `tokenizers` | `0.12.1` | Apache License 2.0 | [0.12.1](https://pypi.org/pypi/tokenizers/0.12.1/json) |
| `tokenizers` | `0.20.3`, `0.21.4`, `0.22.1` | Apache Software License | [0.20.3](https://pypi.org/pypi/tokenizers/0.20.3/json), [0.21.4](https://pypi.org/pypi/tokenizers/0.21.4/json), [0.22.1](https://pypi.org/pypi/tokenizers/0.22.1/json) |
| `toml` | `0.10.2` | MIT | [0.10.2](https://pypi.org/pypi/toml/0.10.2/json) |
| `tomli` | `2.3.0`, `2.4.0` | MIT | [2.3.0](https://pypi.org/pypi/tomli/2.3.0/json), [2.4.0](https://pypi.org/pypi/tomli/2.4.0/json) |
| `tomli` | `2.0.1` | MIT License | [2.0.1](https://pypi.org/pypi/tomli/2.0.1/json) |
| `tomlkit` | `0.12.0`, `0.13.3` | MIT | [0.12.0](https://pypi.org/pypi/tomlkit/0.12.0/json), [0.13.3](https://pypi.org/pypi/tomlkit/0.13.3/json) |
| `torch` | `1.13.1`, `2.2.0` | BSD-3 | [1.13.1](https://pypi.org/pypi/torch/1.13.1/json), [2.2.0](https://pypi.org/pypi/torch/2.2.0/json) |
| `torch` | `2.5.1+cu121`, `2.7.0+cu128` | BSD-3-Clause; base-release metadata only* | [2.5.1](https://pypi.org/pypi/torch/2.5.1/json), [2.7.0](https://pypi.org/pypi/torch/2.7.0/json) |
| `torch` | `1.11.0+cu113`, `2.2.0+cu121` | BSD-3; base-release metadata only* | [1.11.0](https://pypi.org/pypi/torch/1.11.0/json), [2.2.0](https://pypi.org/pypi/torch/2.2.0/json) |
| `torchaudio` | `0.11.0+cu113`, `2.2.0+cu121`, `2.5.1+cu121`, `2.7.0+cu128` | BSD License; base-release metadata only* | [0.11.0](https://pypi.org/pypi/torchaudio/0.11.0/json), [2.2.0](https://pypi.org/pypi/torchaudio/2.2.0/json), [2.5.1](https://pypi.org/pypi/torchaudio/2.5.1/json), [2.7.0](https://pypi.org/pypi/torchaudio/2.7.0/json) |
| `torchmetrics` | `1.5.2`, `1.8.2` | Apache-2.0 | [1.5.2](https://pypi.org/pypi/torchmetrics/1.5.2/json), [1.8.2](https://pypi.org/pypi/torchmetrics/1.8.2/json) |
| `torchvision` | `0.14.1`, `0.17.0` | BSD | [0.14.1](https://pypi.org/pypi/torchvision/0.14.1/json), [0.17.0](https://pypi.org/pypi/torchvision/0.17.0/json) |
| `torchvision` | `0.12.0+cu113`, `0.17.0+cu121`, `0.20.1+cu121`, `0.22.0+cu128` | BSD; base-release metadata only* | [0.12.0](https://pypi.org/pypi/torchvision/0.12.0/json), [0.17.0](https://pypi.org/pypi/torchvision/0.17.0/json), [0.20.1](https://pypi.org/pypi/torchvision/0.20.1/json), [0.22.0](https://pypi.org/pypi/torchvision/0.22.0/json) |
| `tornado` | `6.5.2`, `6.5.3` | Apache-2.0 | [6.5.2](https://pypi.org/pypi/tornado/6.5.2/json), [6.5.3](https://pypi.org/pypi/tornado/6.5.3/json) |
| `tqdm` | `4.65.2`, `4.67.1`, `4.67.3` | MPL-2.0 AND MIT | [4.65.2](https://pypi.org/pypi/tqdm/4.65.2/json), [4.67.1](https://pypi.org/pypi/tqdm/4.67.1/json), [4.67.3](https://pypi.org/pypi/tqdm/4.67.3/json) |
| `trailrunner` | `1.4.0` | MIT License | [1.4.0](https://pypi.org/pypi/trailrunner/1.4.0/json) |
| `traitlets` | `5.14.3` | BSD License (classifier; full notices in metadata) | [5.14.3](https://pypi.org/pypi/traitlets/5.14.3/json) |
| `transformers` | `4.21.1` | Apache | [4.21.1](https://pypi.org/pypi/transformers/4.21.1/json) |
| `transformers` | `4.46.3`, `4.49.0`, `4.51.3`, `4.57.3` | Apache 2.0 License | [4.46.3](https://pypi.org/pypi/transformers/4.46.3/json), [4.49.0](https://pypi.org/pypi/transformers/4.49.0/json), [4.51.3](https://pypi.org/pypi/transformers/4.51.3/json), [4.57.3](https://pypi.org/pypi/transformers/4.57.3/json) |
| `tree` | `0.2.4` | MIT | [0.2.4](https://pypi.org/pypi/tree/0.2.4/json) |
| `trimesh` | `4.10.0`, `4.10.1`, `4.11.2` | MIT License (classifier; full notices in metadata) | [4.10.0](https://pypi.org/pypi/trimesh/4.10.0/json), [4.10.1](https://pypi.org/pypi/trimesh/4.10.1/json), [4.11.2](https://pypi.org/pypi/trimesh/4.11.2/json) |
| `triton` | `2.2.0`, `3.0.0`, `3.1.0`, `3.3.0` | MIT License | [2.2.0](https://pypi.org/pypi/triton/2.2.0/json), [3.0.0](https://pypi.org/pypi/triton/3.0.0/json), [3.1.0](https://pypi.org/pypi/triton/3.1.0/json), [3.3.0](https://pypi.org/pypi/triton/3.3.0/json) |
| `typer` | `0.20.0` | MIT License | [0.20.0](https://pypi.org/pypi/typer/0.20.0/json) |
| `typing-extensions` | `4.13.2`, `4.15.0` | PSF-2.0 | [4.13.2](https://pypi.org/pypi/typing-extensions/4.13.2/json), [4.15.0](https://pypi.org/pypi/typing-extensions/4.15.0/json) |
| `typing-inspect` | `0.9.0` | MIT | [0.9.0](https://pypi.org/pypi/typing-inspect/0.9.0/json) |
| `typing-inspection` | `0.4.2` | MIT | [0.4.2](https://pypi.org/pypi/typing-inspection/0.4.2/json) |
| `tzdata` | `2025.2`, `2025.3` | Apache-2.0 | [2025.2](https://pypi.org/pypi/tzdata/2025.2/json), [2025.3](https://pypi.org/pypi/tzdata/2025.3/json) |
| `urdfpy` | `0.0.22` | MIT License | [0.0.22](https://pypi.org/pypi/urdfpy/0.0.22/json) |
| `uri-template` | `1.3.0` | MIT License | [1.3.0](https://pypi.org/pypi/uri-template/1.3.0/json) |
| `urllib3` | `2.5.0`, `2.6.2` | MIT | [2.5.0](https://pypi.org/pypi/urllib3/2.5.0/json), [2.6.2](https://pypi.org/pypi/urllib3/2.6.2/json) |
| `urllib3` | `2.2.3` | MIT License | [2.2.3](https://pypi.org/pypi/urllib3/2.2.3/json) |
| `usd-core` | `25.11` | LicenseRef-TOST-1.0 | [25.11](https://pypi.org/pypi/usd-core/25.11/json) |
| `usort` | `1.0.8.post1` | MIT | [1.0.8.post1](https://pypi.org/pypi/usort/1.0.8.post1/json) |
| `uvicorn` | `0.38.0` | BSD-3-Clause | [0.38.0](https://pypi.org/pypi/uvicorn/0.38.0/json) |
| `uvloop` | `0.22.1` | MIT License | [0.22.1](https://pypi.org/pypi/uvloop/0.22.1/json) |
| `vllm` | `0.9.1` | Apache-2.0 | [0.9.1](https://pypi.org/pypi/vllm/0.9.1/json) |
| `vtk` | `9.5.2` | BSD | [9.5.2](https://pypi.org/pypi/vtk/9.5.2/json) |
| `wadler-lindig` | `0.1.7` | Apache Software License (classifier; full notices in metadata) | [0.1.7](https://pypi.org/pypi/wadler-lindig/0.1.7/json) |
| `wandb` | `0.23.1`, `0.24.2`, `0.25.0` | MIT License (classifier; full notices in metadata) | [0.23.1](https://pypi.org/pypi/wandb/0.23.1/json), [0.24.2](https://pypi.org/pypi/wandb/0.24.2/json), [0.25.0](https://pypi.org/pypi/wandb/0.25.0/json) |
| `wandb` | `0.13.1` | MIT license | [0.13.1](https://pypi.org/pypi/wandb/0.13.1/json) |
| `warp-lang` | `1.10.1` | Apache-2.0 | [1.10.1](https://pypi.org/pypi/warp-lang/1.10.1/json) |
| `watchfiles` | `1.1.1` | MIT | [1.1.1](https://pypi.org/pypi/watchfiles/1.1.1/json) |
| `wcwidth` | `0.2.14` | MIT | [0.2.14](https://pypi.org/pypi/wcwidth/0.2.14/json) |
| `webcolors` | `1.13` | BSD-3-Clause | [1.13](https://pypi.org/pypi/webcolors/1.13/json) |
| `webdataset` | `1.0.2` | BSD-3-Clause | [1.0.2](https://pypi.org/pypi/webdataset/1.0.2/json) |
| `webdataset` | `0.2.86` | MIT | [0.2.86](https://pypi.org/pypi/webdataset/0.2.86/json) |
| `webencodings` | `0.5.1` | BSD | [0.5.1](https://pypi.org/pypi/webencodings/0.5.1/json) |
| `websocket-client` | `1.9.0` | Apache-2.0 | [1.9.0](https://pypi.org/pypi/websocket-client/1.9.0/json) |
| `websockets` | `12.0` | BSD-3-Clause | [12.0](https://pypi.org/pypi/websockets/12.0/json) |
| `werkzeug` | `3.0.6` | BSD License | [3.0.6](https://pypi.org/pypi/werkzeug/3.0.6/json) |
| `werkzeug` | `3.1.4` | BSD-3-Clause | [3.1.4](https://pypi.org/pypi/werkzeug/3.1.4/json) |
| `widgetsnbextension` | `4.0.15` | BSD 3-Clause License | [4.0.15](https://pypi.org/pypi/widgetsnbextension/4.0.15/json) |
| `wrapt` | `2.0.1` | BSD-2-Clause (license text) | [2.0.1](https://pypi.org/pypi/wrapt/2.0.1/json) |
| `xatlas` | `0.0.11`, `0.0.9` | MIT (license text) | [0.0.11](https://pypi.org/pypi/xatlas/0.0.11/json), [0.0.9](https://pypi.org/pypi/xatlas/0.0.9/json) |
| `xformers` | `0.0.24`, `0.0.28.post3`, `0.0.30` | BSD License | [0.0.24](https://pypi.org/pypi/xformers/0.0.24/json), [0.0.28.post3](https://pypi.org/pypi/xformers/0.0.28.post3/json), [0.0.30](https://pypi.org/pypi/xformers/0.0.30/json) |
| `xgrammar` | `0.1.19` | Apache 2.0 | [0.1.19](https://pypi.org/pypi/xgrammar/0.1.19/json) |
| `xxhash` | `3.6.0` | BSD | [3.6.0](https://pypi.org/pypi/xxhash/3.6.0/json) |
| `yacs` | `0.1.8` | Apache Software License | [0.1.8](https://pypi.org/pypi/yacs/0.1.8/json) |
| `yarl` | `1.15.2`, `1.22.0` | Apache-2.0 | [1.15.2](https://pypi.org/pypi/yarl/1.15.2/json), [1.22.0](https://pypi.org/pypi/yarl/1.22.0/json) |
| `zipp` | `3.23.0`, `3.23.1` | MIT | [3.23.0](https://pypi.org/pypi/zipp/3.23.0/json), [3.23.1](https://pypi.org/pypi/zipp/3.23.1/json) |
| `zipp` | `3.20.2` | MIT License | [3.20.2](https://pypi.org/pypi/zipp/3.20.2/json) |
| `zstandard` | `0.25.0` | BSD-3-Clause | [0.25.0](https://pypi.org/pypi/zstandard/0.25.0/json) |
| `zstd` | `1.5.7.2` | BSD | [1.5.7.2](https://pypi.org/pypi/zstd/1.5.7.2/json) |

\* PyTorch `+cpu` / `+cu...` wheels come from the PyTorch wheel indexes, not these base PyPI releases. The corresponding base release metadata is shown only for the Python project; this does not verify the exact wheel's bundled libraries or their licenses. Consult the [PyTorch wheel index](https://download.pytorch.org/whl/) and the license/notice files shipped in the chosen wheel.

## Direct Git and wheel requirements

These declarations point to a specific repository commit or release asset rather than a PyPI release. Their identities are retained below so that they are not confused with a similarly named PyPI package. See [Git and wheel dependency licenses](VCS_DEPENDENCIES.md) and [Third-party notices](../THIRD_PARTY_NOTICES.md) for the applicable upstream license information.

| Package | Pinned revision or asset | Upstream source |
| --- | --- | --- |
| `rex_omni` | `ee58b1e887f94da49d0749945388defede6b52e6` | [IDEA-Research/Rex-Omni](https://github.com/IDEA-Research/Rex-Omni/tree/ee58b1e887f94da49d0749945388defede6b52e6) |
| `libero` | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01) |
| `tacto` | `dd53360d9a8c186f0d6439372ec0be0fa5e21731` | [lukashermann/tacto](https://github.com/lukashermann/tacto/tree/dd53360d9a8c186f0d6439372ec0be0fa5e21731) |
| `Calvin` | `fa03f01f19c65920e18cf37398a9ce859274af76` | [mees/calvin](https://github.com/mees/calvin/tree/fa03f01f19c65920e18cf37398a9ce859274af76) |
| `calvin_env` | `1431a46bd36bde5903fb6345e68b5ccc30def666` | [mees/calvin_env](https://github.com/mees/calvin_env/tree/1431a46bd36bde5903fb6345e68b5ccc30def666) |
| `PyOpenGL` | `76d1261adee2d3fd99b418e75b0416bb7d2865e6` | [mmatl/pyopengl](https://github.com/mmatl/pyopengl/tree/76d1261adee2d3fd99b418e75b0416bb7d2865e6) |
| `clip` | `dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1` | [openai/CLIP](https://github.com/openai/CLIP/tree/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1) |
| `flash-attn` | `flash_attn-2.5.9.post1+cu122torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl` | [Release asset](https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl) |
| `gsplat` | `2323de5905d5e90e035f792fe65bad0fedd413e7` | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat/tree/2323de5905d5e90e035f792fe65bad0fedd413e7) |
| `moge` | `a8c37341bc0325ca99b9d57981cc3bb2bd3e255b` | [microsoft/MoGe](https://github.com/microsoft/MoGe/tree/a8c37341bc0325ca99b9d57981cc3bb2bd3e255b) |
| `pytorch3d` | `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba` | [facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d/tree/33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba) |
| `pytorch3d` | `75ebeeaea0908c5527e7b1e305fbc7681382db47` | [facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d/tree/75ebeeaea0908c5527e7b1e305fbc7681382db47) |
| `segment_anything` | `dca509fe793f601edb92606367a655c15ac00fdf` | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything/tree/dca509fe793f601edb92606367a655c15ac00fdf) |
| `utils3d` | `3913c65d81e05e47b9f367250cf8c0f7462a0900` | [EasternJournalist/utils3d](https://github.com/EasternJournalist/utils3d/tree/3913c65d81e05e47b9f367250cf8c0f7462a0900) |
