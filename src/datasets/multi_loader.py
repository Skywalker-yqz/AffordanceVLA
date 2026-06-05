"""
Multi-dataset loader for AffordanceVLA.

Provides per-dataset batching: each batch contains samples from exactly ONE
dataset, avoiding cross-dataset size mismatches and enabling zero-overhead
torch.stack within each batch.

Datasets are sampled according to configurable weights (default: proportional
to dataset size). Each dataset independently tracks batch consumption:
within one round every batch is seen exactly once before the dataset resets.
Small datasets naturally complete more rounds than large ones.

DDP support:
    When distributed=True, each dataset gets a DistributedSampler that
    automatically shards data across ranks. Call set_epoch(epoch) before
    each epoch/round to ensure proper shuffling across ranks.

Usage:
    # Single GPU
    loader = MultiDatasetLoader(
        datasets={"prism": prism_ds, "agd20k": agd20k_ds},
        batch_size=4,
    )

    # Multi-GPU (DDP)
    loader = MultiDatasetLoader(
        datasets={"prism": prism_ds, "agd20k": agd20k_ds},
        batch_size=4,
        distributed=True,
    )
"""

from __future__ import annotations

import logging
import random
from typing import Dict, Iterator, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler

from src.datasets.collate import affordance_collate_fn

logger = logging.getLogger(__name__)


class MultiDatasetLoader:
    """Yields batches from multiple datasets, one dataset per batch.

    Each dataset gets its own DataLoader (independent shuffle, workers).
    At each step, a dataset is chosen by weighted random sampling, and
    one batch is drawn from that dataset's DataLoader.

    Args:
        datasets: Dict mapping dataset name to Dataset instance.
        batch_size: Default batch size (shared across all datasets).
        batch_sizes: Optional per-dataset batch sizes.
        weights: Optional per-dataset sampling weights.
        num_workers: DataLoader workers per dataset.
        pin_memory: Whether to pin memory.
        seed: Random seed for dataset sampling order.
        distributed: Whether to use DistributedSampler for DDP training.
    """

    def __init__(
        self,
        datasets: Dict[str, Dataset],
        batch_size: int = 4,
        batch_sizes: Optional[Dict[str, int]] = None,
        weights: Optional[Dict[str, float]] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
        distributed: bool = False,
    ):
        if not datasets:
            raise ValueError("No datasets provided")

        self.dataset_names = sorted(datasets.keys())
        self.rng = random.Random(seed)
        self._distributed = distributed
        self._seed = seed
        self._epoch = 0

        # ── Build per-dataset DataLoaders ──
        self._loaders: Dict[str, DataLoader] = {}
        self._samplers: Dict[str, Optional[DistributedSampler]] = {}
        self._generators: Dict[str, torch.Generator] = {}
        self._batch_sizes: Dict[str, int] = {}

        for name in self.dataset_names:
            ds = datasets[name]
            bs = (batch_sizes or {}).get(name, batch_size)
            self._batch_sizes[name] = bs

            if distributed:
                sampler = DistributedSampler(
                    ds,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=True,
                    seed=seed,
                    drop_last=True,
                )
                self._samplers[name] = sampler
                self._loaders[name] = DataLoader(
                    ds,
                    batch_size=bs,
                    sampler=sampler,
                    num_workers=num_workers,
                    collate_fn=affordance_collate_fn,
                    pin_memory=pin_memory,
                    drop_last=True,
                    persistent_workers=num_workers > 0,
                )
            else:
                gen = torch.Generator()
                gen.manual_seed(self._round_seed(name, 0))
                self._generators[name] = gen
                self._samplers[name] = None
                self._loaders[name] = DataLoader(
                    ds,
                    batch_size=bs,
                    sampler=RandomSampler(ds, generator=gen),
                    num_workers=num_workers,
                    collate_fn=affordance_collate_fn,
                    pin_memory=pin_memory,
                    drop_last=True,
                    persistent_workers=num_workers > 0,
                )

        # ── Compute sampling weights ──
        if weights is not None:
            self._weights = {
                name: weights[name]
                for name in self.dataset_names
                if name in weights
            }
        else:
            total = sum(len(datasets[n]) for n in self.dataset_names)
            self._weights = {
                name: len(datasets[name]) / total
                for name in self.dataset_names
            }

        w_sum = sum(self._weights.values())
        self._weights = {k: v / w_sum for k, v in self._weights.items()}

        self._weight_names = list(self._weights.keys())
        self._weight_values = [self._weights[n] for n in self._weight_names]

        # ── Per-dataset batch tracking ──
        if distributed:
            world_size = dist.get_world_size()
            self._total_batches: Dict[str, int] = {
                name: len(datasets[name]) // (self._batch_sizes[name] * world_size)
                for name in self.dataset_names
            }
        else:
            self._total_batches = {
                name: len(datasets[name]) // self._batch_sizes[name]
                for name in self.dataset_names
            }
        for name, nb in self._total_batches.items():
            if nb <= 0:
                raise ValueError(
                    f"Dataset '{name}' has {len(datasets[name])} samples with "
                    f"batch_size={self._batch_sizes[name]} → 0 batches (too small). "
                    f"Reduce batch_size or remove this dataset."
                )
        self._remaining: Dict[str, int] = dict(self._total_batches)
        self._rounds: Dict[str, int] = {name: 0 for name in self.dataset_names}
        self._iters: Dict[str, Iterator] = {}

        # ── Logging ──
        rank_info = f" (rank {dist.get_rank()}/{dist.get_world_size()})" if distributed else ""
        for name in self.dataset_names:
            logger.info(
                f"  {name:15s}: {len(datasets[name]):>7,} samples, "
                f"bs={self._batch_sizes[name]}, "
                f"{self._total_batches[name]} batches/round{rank_info}, "
                f"weight={self._weights.get(name, 0):.3f}"
            )

    def set_epoch(self, epoch: int):
        """Set epoch for DistributedSamplers (ensures proper shuffling across ranks)."""
        self._epoch = epoch
        for name, sampler in self._samplers.items():
            if sampler is not None:
                sampler.set_epoch(epoch)

    def _round_seed(self, name: str, round_num: int) -> int:
        """Deterministic seed for a (dataset, round) pair.

        Mirrors DistributedSampler's epoch-based seeding so that both
        distributed and non-distributed modes produce reproducible
        per-round permutations independent of global RNG state.
        """
        name_idx = self.dataset_names.index(name) + 1
        return self._seed + (self._epoch + round_num) * (len(self.dataset_names) + 1) + name_idx

    def _get_batch(self, name: str) -> dict:
        """Get one unused batch from the named dataset."""
        if name not in self._iters:
            self._iters[name] = iter(self._loaders[name])

        if self._remaining[name] <= 0:
            self._rounds[name] += 1
            self._remaining[name] = self._total_batches[name]
            if self._samplers.get(name) is not None:
                self._samplers[name].set_epoch(self._epoch + self._rounds[name])
            if name in self._generators:
                self._generators[name].manual_seed(self._round_seed(name, self._rounds[name]))
            self._iters[name] = iter(self._loaders[name])

        batch = next(self._iters[name])
        self._remaining[name] -= 1
        return batch

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        name = self.rng.choices(
            self._weight_names, weights=self._weight_values, k=1
        )[0]
        return self._get_batch(name)

    def state_dict(self) -> dict:
        """Serialize loader state for checkpoint resume."""
        return {
            "rng_state": self.rng.getstate(),
            "remaining": dict(self._remaining),
            "rounds": dict(self._rounds),
            "epoch": self._epoch,
        }

    def load_state_dict(self, state: dict):
        """Restore loader state from checkpoint, including iterator positions.

        For each dataset mid-round at checkpoint time, recreates the iterator
        with the same deterministic permutation and fast-forwards past the
        already-consumed batches so the next batch is identical to what
        uninterrupted training would have produced.
        """
        self.rng.setstate(state["rng_state"])
        self._remaining = {k: state["remaining"][k] for k in self._remaining if k in state["remaining"]}
        self._rounds = {k: state["rounds"][k] for k in self._rounds if k in state["rounds"]}
        self._epoch = state["epoch"]

        # Restore sampler epochs (distributed) / generator seeds (non-distributed)
        # so that iter(DataLoader) produces the same permutation as the current round.
        for name in self.dataset_names:
            round_num = self._rounds.get(name, 0)
            if self._samplers.get(name) is not None:
                self._samplers[name].set_epoch(self._epoch + round_num)
            if name in self._generators:
                self._generators[name].manual_seed(self._round_seed(name, round_num))

        self._iters.clear()

        # Fast-forward each dataset's iterator past already-consumed batches.
        for name in self.dataset_names:
            if self._remaining[name] <= 0:
                continue
            consumed = self._total_batches[name] - self._remaining[name]
            if consumed > 0:
                logger.info(
                    f"  {name}: skipping {consumed}/{self._total_batches[name]} "
                    f"batches to restore position..."
                )
                it = iter(self._loaders[name])
                for _ in range(consumed):
                    next(it)
                self._iters[name] = it

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._weights)

    @property
    def rounds(self) -> Dict[str, int]:
        return dict(self._rounds)

    @property
    def remaining(self) -> Dict[str, int]:
        return dict(self._remaining)
