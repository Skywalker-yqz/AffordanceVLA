# Conda Environments

This directory contains the frozen dependency lists (`pip freeze`) for every
conda environment used by the AffordanceVLA project. Each `requirements_<env>.txt`
file is an exact snapshot of the packages installed in the corresponding
environment and can be used to reproduce it.

## Environment Overview

| Environment | Python | Requirements file | Key packages | Purpose |
|-------------|--------|--------------------|--------------|---------|
| `AffordanceVLA` | 3.11 | `requirements_AffordanceVLA.txt` | torch 2.2.0+cu121, transformers 4.49.0, diffusers 0.31.0, pytorch-lightning 2.6.0 | Main environment for training and inference of the AffordanceVLA model. |
| `AffordanceVLA_calvin` | 3.11 | `requirements_AffordanceVLA_calvin.txt` | torch 2.2.0+cu121, transformers 4.49.0, pytorch-lightning 2.6.0 | Running AffordanceVLA on the CALVIN benchmark (model stack + CALVIN evaluation deps). |
| `AffordanceVLA_libero` | 3.11 | `requirements_AffordanceVLA_libero.txt` | torch 2.2.0+cu121, transformers 4.49.0, robosuite 1.4.0, robomimic 0.2.0 | Running AffordanceVLA on the LIBERO benchmark (model stack + LIBERO/robosuite simulation). |
| `SAM3D` | 3.11 | `requirements_SAM3D.txt` | torch 2.5.1+cu121, transformers 4.57.3, diffusers 0.36.0 | SAM / 3D segmentation and related vision preprocessing. |
| `calvin_venv` | 3.8 | `requirements_calvin_venv.txt` | torch 1.13.1, transformers 4.46.3, pytorch-lightning 1.8.6 | Native CALVIN simulator environment (legacy CALVIN toolchain). |
| `libero` | 3.8 | `requirements_libero.txt` | torch 1.11.0+cu113, robosuite 1.4.0, robomimic 0.2.0 | Native LIBERO simulator environment (legacy LIBERO toolchain). |
| `rexomni` | 3.10 | `requirements_rexomni.txt` | torch 2.7.0+cu128, transformers 4.51.3, accelerate 1.10.1 | Rex-Omni service / grounding model environment. |
| `robointer` | 3.9 | `requirements_robointer.txt` | matplotlib 3.9.4, opencv-python 4.13.0, lmdb 2.2.0, numpy 2.0.2 | Lightweight data processing / visualization utilities. |

## Why Multiple Environments

The full AffordanceVLA pipeline integrates many heavy and rapidly-evolving
components (the VLA model stack, multiple simulator/benchmark toolchains, and
several large vision/grounding models). These components have **mutually
incompatible dependencies** — they require different Python versions
(3.8 / 3.9 / 3.10 / 3.11), conflicting CUDA/PyTorch builds (cu113 / cu121 / cu128),
and incompatible versions of shared libraries such as `transformers`, `numpy`,
and `pytorch-lightning`. As a result, they **cannot coexist in a single
environment** and are intentionally split into the isolated environments listed
above.

Because of this isolation, the two model components that the main pipeline must
call at runtime — **`SAM3D`** and **`rexomni`** — cannot be imported directly
into the main `AffordanceVLA` process. Instead, each is **wrapped as a standalone
microservice**:

- `SAM3D` runs as a separate service in its own environment and exposes SAM /
  3D-segmentation inference over an API.
- `rexomni` runs as a separate service in its own environment and exposes the
  Rex-Omni grounding model over an API.

The main `AffordanceVLA` environment communicates with these microservices via
inter-process / network calls, so each side keeps its own incompatible
dependency stack while still being usable together in one pipeline.

## Notes

- The files were produced with `pip freeze`, so they pin exact versions and may
  include platform-specific wheels (e.g. CUDA builds such as `+cu121`,
  `+cu128`). Adjust the CUDA suffix to match your hardware/driver if needed.
- CUDA-tagged PyTorch wheels are not available on the default PyPI index. Install
  them from the matching PyTorch index.

### Manual steps required before `pip install -r`

Due to environment-deployment conflicts, two entries in the requirements files
cannot be resolved from a public index and must be handled manually:

**1. SAM3D editable install** (`requirements_AffordanceVLA.txt`, `requirements_AffordanceVLA_calvin.txt`, `requirements_AffordanceVLA_libero.txt`)

Each file contains a line like:

```
-e /mnt/users/AffordanceVLA/SAM3D
```

This is a local editable install of the SAM3D package. Before running
`pip install -r`, **delete that line** from the requirements file, then install
SAM3D manually by cloning its repository and running `pip install -e .` from
the cloned directory.

**2. `flash-attn` local wheel** (`requirements_rexomni.txt`)

The file contains a line like:

```
flash_attn @ file:///mnt/oss/Downloads/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

This points to a pre-built wheel that is no longer accessible outside our
cluster. Before running `pip install -r`, **delete that line**, then download
the matching wheel for your CUDA / Python version from the
[flash-attn releases page](https://github.com/Dao-AILab/flash-attention/releases)
and install it with:

```bash
pip install flash_attn-<version>-<python>-<platform>.whl
```

The wheel filename encodes the required CUDA toolkit, PyTorch, and Python
versions — choose the one that matches your environment.

- `requirements_RoboTwin.txt` is intentionally minimal because the `RoboTwin`
  environment has not yet had its project dependencies installed. Re-run the
  freeze step after setting it up to refresh the file.
