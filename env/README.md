# Installation

The model environment supports training and inference on NVIDIA CUDA GPUs, including H800 and H200. Run the commands below from the repository root. The model, simulator, and annotation services use separate environments.

## Model environment

Install Miniforge or Miniconda and an NVIDIA driver compatible with CUDA 12.1. Check that `nvidia-smi` recognizes the GPU, then create the environment:

```bash
conda create -n affordancevla python=3.11 -y
conda activate affordancevla
python -m pip install --upgrade pip
python -m pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r env/requirements_core.txt
python -m pip check
python -c "import torch; from src.models.configuration_affordance import AffordanceVLAConfig; print(torch.__version__, torch.cuda.is_available())"
python -m src.train --help
```

The PyTorch wheel includes its CUDA runtime. A local CUDA toolkit is needed when building additional CUDA extensions. The core model uses PyTorch attention and requires no FlashAttention extension. Use `model.attention_implementation=sdpa` for PyTorch scaled dot-product attention or `eager` for attention visualization.

Place public model weights under `pretrained/` as described in the main [README](../README.md). Training configurations use relative paths and accept command-line overrides. The effective batch size is controlled by `training.effective_batch_size`; the trainer derives gradient accumulation from the per-device `training.batch_size` and the distributed world size. Reduce the per-device batch size for smaller memory budgets while keeping the effective batch size fixed.

## Optional services and benchmarks

| Component | Setup |
|-----------|-------|
| LIBERO | Follow [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and the [OpenVLA evaluation setup](https://github.com/openvla/openvla). |
| CALVIN | Follow [CALVIN](https://github.com/mees/calvin) in its simulator environment. |
| Rex-Omni | Follow [Rex-Omni](https://github.com/IDEA-Research/Rex-Omni); start the wrapper in `src/services/rexomni_service.py`. |
| SAM-3D | Follow [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects); start the wrapper in `src/services/sam3d_service.py`. |

The `requirements_AffordanceVLA*.txt`, `requirements_calvin_venv.txt`, `requirements_libero.txt`, `requirements_rexomni.txt`, `requirements_SAM3D.txt`, and `requirements_robointer.txt` files record the corresponding development environments. Use `requirements_core.txt` for the model installation above and the upstream installation guides for each optional component.
