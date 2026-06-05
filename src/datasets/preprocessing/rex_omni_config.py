#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RexOmni inference configuration.

Two backends are supported:
    - "transformers": single-GPU inference (default).
    - "vllm":         high-throughput batched inference for large-scale
                      preprocessing.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RexOmniConfig:
    """RexOmni model configuration.

    Attributes:
        backend: Inference backend, ``"transformers"`` or ``"vllm"``.
        model_path: Model path; ``None`` keeps the wrapper default.
        gpu_memory_utilization: vLLM-only — fraction of GPU memory to use.
        tensor_parallel_size: vLLM-only — tensor-parallel size.
        batch_size: vLLM-only — batch size.
        max_model_len: vLLM-only — maximum model context length.
        enforce_eager: vLLM-only — disable CUDA graphs.
        trust_remote_code: Whether to trust HuggingFace remote code.
    """
    backend: str = "transformers"  # "transformers" | "vllm"

    model_path: Optional[str] = None

    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 1
    batch_size: int = 16
    max_model_len: int = 4096
    enforce_eager: bool = False

    trust_remote_code: bool = True

    def to_dict(self) -> dict:
        """Serialize the configuration to a plain dict."""
        return {
            "backend": self.backend,
            "model_path": self.model_path,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "batch_size": self.batch_size,
            "max_model_len": self.max_model_len,
            "enforce_eager": self.enforce_eager,
            "trust_remote_code": self.trust_remote_code,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RexOmniConfig":
        """Build a config from a dict, ignoring unknown keys."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def validate(self):
        """Sanity-check the field values."""
        if self.backend not in ("transformers", "vllm"):
            raise ValueError(f"Unsupported backend: {self.backend}")

        if self.gpu_memory_utilization <= 0 or self.gpu_memory_utilization > 1:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1]: {self.gpu_memory_utilization}"
            )

        if self.tensor_parallel_size < 1:
            raise ValueError(f"tensor_parallel_size must be >= 1: {self.tensor_parallel_size}")

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1: {self.batch_size}")
