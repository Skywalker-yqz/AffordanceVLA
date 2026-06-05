"""
AffordanceVLA training script — supports single-GPU and multi-GPU (DDP).

Launch:
    # Single GPU
    python -m src.train --config configs/stage1.yaml

    # Multi-GPU (2 cards for debug, 8 cards for production)
    torchrun --nproc_per_node=2 -m src.train --config configs/stage1.yaml
    torchrun --nproc_per_node=8 -m src.train --config configs/stage1.yaml

DDP design:
    - ALL ranks: create model, load weights, create datasets, run training loop
    - ONLY rank 0: logging to file, wandb, checkpoint saving, config saving
    - DistributedSampler: each rank sees a disjoint shard of data
    - DDP wrapper: gradients automatically averaged across ranks
    - Barrier after save: all ranks wait until rank 0 finishes saving

Fatal bug prevention:
    1. Dataset initialization is OUTSIDE the training loop
       (see the ``create_datasets`` call in ``main``).
       Never re-created during training — shuffle state preserved.
    2. ALL ranks create datasets simultaneously with DistributedSampler
       for correct data sharding. No file lock conflicts.
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import datetime
import logging
import math
import random
import shutil
import signal
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf

from src.models.configuration_affordance import AffordanceVLAConfig
from src.policies.affordance_policy import AffordanceVLAPolicy
from src.processors.optimizer_scheduler import create_optimizer
try:
    from src.datasets import (
        PRISMDataset, AGD20KDataset, RefSpatialDataset,
        A1Dataset, LiberoDataset, CalvinDataset,
    )
except ImportError:
    PRISMDataset = AGD20KDataset = RefSpatialDataset = None
    A1Dataset = LiberoDataset = CalvinDataset = None

from src.datasets.multi_loader import MultiDatasetLoader

logger = logging.getLogger("train")

# Stage I trains the How2Act decoder when True, freezes it when False.
STAGE1_TRAIN_HOW2ACT = True
# Stage II/III freezes the vision encoder when True; when False it is fine-tuned at vision_encoder_lr.
STAGE2_FREEZE_VISION_ENCODER = False

try:
    import wandb as _wandb
    _HAS_WANDB = True
except ImportError:
    _wandb = None
    _HAS_WANDB = False


# ═══════════════════════════════════════════════════════════════════════
# Graceful Shutdown (SIGINT / SIGTERM)
# ═══════════════════════════════════════════════════════════════════════

_SHUTDOWN_REQUESTED = False


def _shutdown_handler(signum, frame):
    """Set flag on first signal; force-exit on second."""
    global _SHUTDOWN_REQUESTED
    if _SHUTDOWN_REQUESTED:
        logger.warning("Second signal received — force exit.")
        sys.exit(1)
    _SHUTDOWN_REQUESTED = True
    sig_name = signal.Signals(signum).name
    logger.info(
        f"Received {sig_name}, will save checkpoint after current step and exit..."
    )


signal.signal(signal.SIGINT, _shutdown_handler)
signal.signal(signal.SIGTERM, _shutdown_handler)


# ═══════════════════════════════════════════════════════════════════════
# Distributed Helpers
# ═══════════════════════════════════════════════════════════════════════


def is_dist() -> bool:
    """Check if distributed training is active."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier():
    """Synchronize all processes."""
    if is_dist():
        dist.barrier()


def setup_distributed():
    """Initialize distributed process group (NCCL backend).

    torchrun sets LOCAL_RANK, RANK, WORLD_SIZE automatically.
    """
    if "RANK" not in os.environ:
        return  # Not launched with torchrun → single GPU

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    logger.info(
        f"DDP initialized: rank={dist.get_rank()}/{dist.get_world_size()}, "
        f"local_rank={local_rank}, device=cuda:{local_rank}"
    )


def cleanup_distributed():
    """Destroy distributed process group."""
    if is_dist():
        dist.destroy_process_group()


# ═══════════════════════════════════════════════════════════════════════
# W&B Integration (rank 0 only)
# ═══════════════════════════════════════════════════════════════════════


def setup_wandb(cfg, output_dir: str) -> bool:
    if not is_main_process():
        return False
    if not cfg.training.get("wandb_enabled", False):
        logger.info("W&B: disabled")
        return False
    if not _HAS_WANDB:
        raise ImportError(
            "W&B is enabled (wandb_enabled=True) but wandb is not installed. "
            "Please install it: pip install wandb"
        )

    project = cfg.training.get("wandb_project", "AffordanceVLA")
    entity = cfg.training.get("wandb_entity", None)
    run_name = cfg.training.get("wandb_run_name", None)
    tags = list(cfg.training.get("wandb_tags", []))
    if not run_name:
        run_name = f"{cfg.training.stage}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
    tags.append(cfg.training.stage)

    _wandb.init(
        project=project, entity=entity, name=run_name, tags=tags,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=output_dir, resume="allow",
    )
    logger.info(f"W&B: initialized — project={project}, run={run_name}")
    return True


def wandb_log(metrics: dict, step: int):
    if _HAS_WANDB and _wandb.run is not None:
        _wandb.log(metrics, step=step)


def wandb_finish():
    if _HAS_WANDB and _wandb.run is not None:
        _wandb.finish()


# ═══════════════════════════════════════════════════════════════════════
# Logging Setup (rank 0: console + file; other ranks: WARNING only)
# ═══════════════════════════════════════════════════════════════════════


def setup_logging(output_dir: str, log_level: str = "INFO"):
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = "[%(levelname)s|%(name)s:%(lineno)s] %(asctime)s >> %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    if is_main_process():
        root_logger.setLevel(level)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        try:
            latest_link = log_dir / "latest.log"
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(log_file.name)
        except OSError:
            pass

        logger.info(f"Logging to: {log_file}")
    else:
        # Non-rank-0: suppress all INFO, only WARNING+
        root_logger.setLevel(logging.WARNING)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    return log_file


# ═══════════════════════════════════════════════════════════════════════
# Seed
# ═══════════════════════════════════════════════════════════════════════


def set_seed(seed: int):
    seed = seed + get_rank()  # Different seed per rank for data diversity
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════
# Pretrained Weight Loading
# ═══════════════════════════════════════════════════════════════════════


def _remap_pi0_key(key: str) -> str:
    k = key
    pali = "paligemma_with_expert.paligemma."

    if k.startswith(f"{pali}model.vision_tower."):
        k = k.replace(f"{pali}model.vision_tower.", f"{pali}vision_tower.")
    elif k.startswith(f"{pali}model.multi_modal_projector."):
        k = k.replace(f"{pali}model.multi_modal_projector.", f"{pali}multi_modal_projector.")
    elif k.startswith(f"{pali}model.language_model."):
        k = k.replace(f"{pali}model.language_model.", f"{pali}language_model.model.")
    elif k.startswith(f"{pali}model.embed_tokens."):
        k = k.replace(f"{pali}model.embed_tokens.", f"{pali}language_model.model.embed_tokens.")
    elif k.startswith(f"{pali}model.norm."):
        k = k.replace(f"{pali}model.norm.", f"{pali}language_model.model.norm.")
    elif k.startswith(f"{pali}lm_head."):
        k = k.replace(f"{pali}lm_head.", f"{pali}language_model.lm_head.")

    return f"model.{k}"


def load_pretrained_weights(policy: AffordanceVLAPolicy, path: str):
    import safetensors.torch
    from safetensors import safe_open

    path = str(path)
    if os.path.isdir(path):
        model_file = os.path.join(path, "model.safetensors")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"No model.safetensors in {path}")
    elif os.path.isfile(path) and path.endswith(".safetensors"):
        model_file = path
    else:
        raise FileNotFoundError(f"Pretrained path not found: {path}")

    logger.info(f"Loading pretrained weights from: {model_file}")

    policy_keys = set(policy.state_dict().keys())
    with safe_open(model_file, framework="pt") as f:
        ckpt_keys = set(f.keys())

    direct_overlap = ckpt_keys & policy_keys

    if len(direct_overlap) >= len(ckpt_keys) * 0.5:
        missing, unexpected = safetensors.torch.load_model(policy, model_file, strict=False)
        if missing:
            logger.info(f"  {len(missing)} missing keys, e.g.: {list(missing)[:3]}...")
        if unexpected:
            logger.info(f"  {len(unexpected)} unexpected keys, e.g.: {list(unexpected)[:3]}...")
    else:
        logger.info("  External checkpoint detected, applying key remapping...")
        with safe_open(model_file, framework="pt") as f:
            remapped = {}
            for key in f.keys():
                new_key = _remap_pi0_key(key)
                if new_key in policy_keys:
                    remapped[new_key] = f.get_tensor(key)
        policy.load_state_dict(remapped, strict=False)
        missing_keys = policy_keys - set(remapped.keys())
        logger.info(f"  Loaded {len(remapped)} / {len(ckpt_keys)} keys (after remapping)")
        if missing_keys:
            logger.info(f"  {len(missing_keys)} missing keys, e.g.: {sorted(list(missing_keys))[:3]}...")

    logger.info("Pretrained weights loaded successfully.")


# ═══════════════════════════════════════════════════════════════════════
# Warmup + Cosine Scheduler
# ═══════════════════════════════════════════════════════════════════════


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=1e-7):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            factor = (step + 1) / max(1, self.warmup_steps)
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.eta_min + (base_lr - self.eta_min) * factor for base_lr in self.base_lrs]


# ═══════════════════════════════════════════════════════════════════════
# Dataset Creation (ALL ranks — same datasets, different shards via sampler)
# ═══════════════════════════════════════════════════════════════════════


def create_datasets(cfg, mode: str = "debug", force_rescan: bool = False) -> dict:
    """Create training datasets from config.

    Args:
        cfg: data config (cfg.data from OmegaConf).
        mode: "debug" uses minimal data (1 task/suite per dataset);
              "train" uses full data (all tasks/suites/splits).
        force_rescan: If True, ignore cached frame indices and rebuild from scratch.
    """
    datasets = {}
    default_target = (cfg.get("dataset_target_h", 480), cfg.get("dataset_target_w", 640))
    chunk_size = cfg.get("chunk_size", 50)
    control_frame = cfg.get("control_frame", False)
    frame_num = cfg.get("frame_num", 10)
    control_speed = cfg.get("control_speed", False)
    pos_limit = cfg.get("pos_limit", 0.06)
    ori_limit = cfg.get("ori_limit", 0.14)

    logger.info(f"Dataset mode: {mode}")

    for ds_cfg in cfg.datasets:
        ds_type = ds_cfg.type
        logger.info(f"Loading dataset: {ds_type}")

        # ── Stage I (mode does not affect dataset assembly) ──
        if ds_type == "prism":
            datasets[ds_type] = PRISMDataset(
                data_dir=ds_cfg.path,
                min_bbox_ratio=ds_cfg.get("min_bbox_ratio", 0.0),
            )

        elif ds_type == "agd20k":
            target_size = tuple(ds_cfg.get("target_size", list(default_target)))
            datasets[ds_type] = AGD20KDataset(data_dir=ds_cfg.path, target_size=target_size)

        elif ds_type == "refspatial":
            target_size = tuple(ds_cfg.get("target_size", list(default_target)))
            datasets[ds_type] = RefSpatialDataset(
                data_dir=ds_cfg.data_dir, image_dir=ds_cfg.image_dir,
                target_size=target_size,
                min_bbox_ratio=ds_cfg.get("min_bbox_ratio", 0.0),
            )

        # ── Stage II (mode controls how much data to load) ──
        elif ds_type == "a1":
            datasets[ds_type] = A1Dataset(
                preprocessed_dir=ds_cfg.preprocessed_root,
                raw_dir=ds_cfg.raw_root,
                frames_dir=ds_cfg.frames_root,
                category_path=ds_cfg.get("debug_task", ""),
                chunk_size=chunk_size,
                control_frame=control_frame,
                frame_num=frame_num,
                control_speed=control_speed,
                pos_limit=pos_limit, ori_limit=ori_limit,
                scan_all_tasks=(mode == "train"),
                force_rescan=force_rescan,
            )

        elif ds_type == "libero":
            if mode == "train":
                suites = ["libero_spatial", "libero_object", "libero_goal",
                           "libero_90", "libero_10"]
            else:
                suites = [ds_cfg.get("debug_suite", "libero_spatial")]
            datasets[ds_type] = LiberoDataset(
                preprocessed_dir=ds_cfg.preprocessed_dir, raw_dir=ds_cfg.raw_dir,
                keyframe_results_dir=ds_cfg.keyframe_results_dir,
                suites=suites, chunk_size=chunk_size,
                control_frame=control_frame, frame_num=frame_num,
                control_speed=control_speed,
                pos_limit=pos_limit, ori_limit=ori_limit,
                filter_noops=ds_cfg.get("filter_noops", True),
                force_rescan=force_rescan)

        elif ds_type == "calvin":
            if mode == "train":
                for split in ["training", "validation"]:
                    key = f"calvin_{split}"
                    datasets[key] = CalvinDataset(
                        preprocessed_dir=ds_cfg.preprocessed_dir, raw_dir=ds_cfg.raw_dir,
                        keyframe_results_dir=ds_cfg.keyframe_results_dir,
                        split=split, chunk_size=chunk_size,
                        control_frame=control_frame, frame_num=frame_num,
                        control_speed=control_speed,
                        pos_limit=pos_limit, ori_limit=ori_limit,
                        force_rescan=force_rescan)
            else:
                datasets[ds_type] = CalvinDataset(
                    preprocessed_dir=ds_cfg.preprocessed_dir, raw_dir=ds_cfg.raw_dir,
                    keyframe_results_dir=ds_cfg.keyframe_results_dir,
                    split=ds_cfg.get("debug_split", "training"), chunk_size=chunk_size,
                    control_frame=control_frame, frame_num=frame_num,
                    control_speed=control_speed,
                    pos_limit=pos_limit, ori_limit=ori_limit,
                    force_rescan=force_rescan)

        else:
            raise ValueError(
                f"Unknown dataset type: '{ds_type}'. "
                f"Supported: prism, agd20k, refspatial, a1, libero, calvin"
            )

    if not datasets:
        raise RuntimeError("No datasets loaded!")

    total = sum(len(ds) for ds in datasets.values())
    logger.info(
        f"Loaded {len(datasets)} datasets, {total:,} total samples: "
        + ", ".join(f"{n}={len(d):,}" for n, d in datasets.items())
    )
    return datasets


# ═══════════════════════════════════════════════════════════════════════
# Freeze Control
# ═══════════════════════════════════════════════════════════════════════


def apply_freeze_strategy(policy: AffordanceVLAPolicy, stage: str):
    if stage == "stage1":
        logger.info("=== Stage I: Freeze Understanding + Action ===")
        for p in policy.model.paligemma_with_expert.paligemma.parameters():
            p.requires_grad = False
        policy.model.paligemma_with_expert.paligemma.eval()
        for p in policy.model.paligemma_with_expert.gemma_expert.parameters():
            p.requires_grad = False
        policy.model.paligemma_with_expert.gemma_expert.eval()
        for name in ["state_proj", "action_in_proj", "action_out_proj",
                      "action_time_mlp_in", "action_time_mlp_out"]:
            for p in getattr(policy.model, name).parameters():
                p.requires_grad = False
        for p in policy.how2act_decoder.parameters():
            p.requires_grad = STAGE1_TRAIN_HOW2ACT
        if policy.wrist_decoder is not None:
            for p in policy.wrist_decoder.parameters():
                p.requires_grad = False
        for p in policy.model.paligemma_with_expert.gemma_gen_expert.parameters():
            p.requires_grad = True
        for p in policy.model.affordance_query_module.parameters():
            p.requires_grad = True
        if policy.which2act_decoder is not None:
            if hasattr(policy.which2act_decoder, 'cls_head'):
                for p in policy.which2act_decoder.cls_head.parameters():
                    p.requires_grad = True
            if hasattr(policy.which2act_decoder, 'proj_head'):
                for p in policy.which2act_decoder.proj_head.parameters():
                    p.requires_grad = True
        for p in policy.where2act_decoder.parameters():
            p.requires_grad = True
        # Sync flags so PaliGemmaWithExpertModel.train() preserves eval()
        # on frozen paligemma and gemma_expert (action) after policy.train()
        policy.model.paligemma_with_expert.train_gen_expert_only = True
    elif stage == "stage2":
        logger.info("=== Stage II: Train all experts + decoders ===")
        for n, p in policy.named_parameters():
            if "vqvae" not in n and "flux_vae" not in n:
                p.requires_grad = True
        if STAGE2_FREEZE_VISION_ENCODER:
            for p in policy.model.paligemma_with_expert.paligemma.vision_tower.parameters():
                p.requires_grad = False
    else:
        raise ValueError(f"Unknown stage: {stage}")

    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"Params: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint Save / Load (rank 0 saves, all ranks load)
# ═══════════════════════════════════════════════════════════════════════


def save_checkpoint(policy, optimizer, scheduler, step, output_dir,
                    scaler=None, train_loader=None):
    """Only rank 0 saves model/optimizer/scheduler; ALL ranks save RNG states.

    Uses write-to-tmp-then-rename for atomicity on model files.
    If the process is killed mid-write, only the hidden .tmp directory is
    left incomplete — the real checkpoint-{step} directory is never corrupted.
    """
    ckpt_dir = output_dir / f"checkpoint-{step}"

    if is_main_process():
        tmp_dir = output_dir / f".checkpoint-{step}.tmp"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        model_to_save = policy.module if isinstance(policy, DDP) else policy
        model_to_save._save_pretrained(tmp_dir)

        state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": step,
        }
        if scaler is not None:
            state["scaler"] = scaler.state_dict()
        if train_loader is not None and hasattr(train_loader, "state_dict"):
            state["loader"] = train_loader.state_dict()
        torch.save(state, tmp_dir / "training_state.pt")

        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        tmp_dir.rename(ckpt_dir)
        logger.info(f"Saved checkpoint: {ckpt_dir}")
    barrier()

    # ALL ranks save their own RNG states for deterministic resume
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    torch.save(rng_state, ckpt_dir / f"rng_state_{get_rank()}.pt")
    barrier()


def _is_checkpoint_complete(ckpt_dir: Path) -> bool:
    """Verify that a checkpoint directory contains all required files."""
    required = ["config.json", "model.safetensors", "training_state.pt"]
    return all((ckpt_dir / f).exists() for f in required)


def load_checkpoint(policy, optimizer, scheduler, ckpt_dir, device,
                    scaler=None, train_loader=None):
    import safetensors.torch
    if not _is_checkpoint_complete(ckpt_dir):
        missing = [f for f in ["config.json", "model.safetensors", "training_state.pt"]
                    if not (ckpt_dir / f).exists()]
        raise RuntimeError(f"Incomplete checkpoint: {ckpt_dir}. Missing: {missing}")
    model = policy.module if isinstance(policy, DDP) else policy
    safetensors.torch.load_model(model, str(ckpt_dir / "model.safetensors"), device=str(device))
    state = torch.load(ckpt_dir / "training_state.pt", map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    if train_loader is not None and "loader" in state and hasattr(train_loader, "load_state_dict"):
        train_loader.load_state_dict(state["loader"])

    # Restore per-rank RNG states for deterministic resume
    rng_file = ckpt_dir / f"rng_state_{get_rank()}.pt"
    if rng_file.exists():
        rng_state = torch.load(rng_file, map_location="cpu")
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.random.set_rng_state(rng_state["torch"])
        if rng_state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_state["cuda"])
        logger.info(f"Restored RNG states for rank {get_rank()}")
    else:
        logger.warning(f"RNG state file not found: {rng_file}")

    return state["global_step"]


def find_latest_checkpoint(output_dir: Path):
    ckpt_dirs = sorted(
        [
            d for d in output_dir.glob("checkpoint-*")
            if not d.name.endswith(".tmp") and _is_checkpoint_complete(d)
        ],
        key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0,
    )
    if ckpt_dirs:
        return ckpt_dirs[-1]
    # Only rank 0 cleans up incomplete .tmp dirs to avoid DDP race conditions
    if is_main_process():
        for tmp in output_dir.glob(".checkpoint-*.tmp"):
            logger.warning(f"Removing incomplete temp checkpoint: {tmp}")
            try:
                shutil.rmtree(tmp)
            except OSError:
                pass
    return None


# ═══════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════


def train(cfg):
    # ── 0. Distributed setup ──
    setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    use_bf16 = cfg.training.get("bf16", False) and torch.cuda.is_available()

    # ── 0a. Logging (rank 0 only writes to file; other ranks suppress INFO) ──
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = setup_logging(str(output_dir), cfg.training.get("log_level", "INFO"))

    # ── 0b. W&B (rank 0 only) ──
    setup_wandb(cfg, str(output_dir))

    logger.info(f"Device: {device}, bf16: {use_bf16}, world_size: {get_world_size()}")
    logger.info(f"Log file: {log_file}")

    set_seed(cfg.training.seed)

    # ── 1. Create model (ALL ranks) ──
    logger.info("Creating model...")
    model_cfg = AffordanceVLAConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    policy = AffordanceVLAPolicy(model_cfg)

    # ── 2. Load pretrained weights (ALL ranks, same file) ──
    if model_cfg.pretrained_path:
        load_pretrained_weights(policy, model_cfg.pretrained_path)
    else:
        logger.warning("model.pretrained_path not set — random init!")

    # ── 3. Load stage checkpoint (ALL ranks) ──
    if model_cfg.load_ckpt:
        load_pretrained_weights(policy, model_cfg.load_ckpt)

    policy = policy.to(device)

    # ── 4. Apply freeze strategy (ALL ranks) ──
    stage = cfg.training.stage
    apply_freeze_strategy(policy, stage)

    # ── 5. Wrap with DDP ──
    if is_dist():
        # find_unused_parameters=True is REQUIRED — DO NOT change to static_graph=True.
        # The computation graph varies per batch: action_loss, how2act_loss, wrist_loss
        # are conditionally zero depending on which dataset the batch comes from,
        # so different trainable parameter subsets participate in each forward pass.
        # In Stage 2 with mixed datasets, this is essential to avoid DDP hangs.
        policy = DDP(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        logger.info(f"Model wrapped with DDP on device cuda:{local_rank}")

    # ── 6. Create datasets (ALL ranks — same datasets, different shards) ──
    # CRITICAL: This is OUTSIDE the training loop. Never re-create datasets during training!
    logger.info("Loading datasets...")
    force_rescan = cfg.data.get("force_rescan", False)
    if force_rescan:
        logger.info("force_rescan=True — ignoring cached frame indices, rebuilding from scratch")
    datasets_dict = create_datasets(
        cfg.data, mode=cfg.get("mode", "debug"), force_rescan=force_rescan,
    )

    ds_weights = OmegaConf.to_container(
        cfg.data.get("weights", OmegaConf.create({})), resolve=True
    ) or None
    ds_batch_sizes = OmegaConf.to_container(
        cfg.data.get("batch_sizes", OmegaConf.create({})), resolve=True
    ) or None

    train_loader = MultiDatasetLoader(
        datasets=datasets_dict,
        batch_size=cfg.training.batch_size,
        batch_sizes=ds_batch_sizes,
        weights=ds_weights,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        seed=cfg.training.seed,
        distributed=is_dist(),  # ← DDP: uses DistributedSampler per dataset
    )

    # ── 7. Optimizer + Scheduler ──
    # Optimizer operates on the unwrapped model parameters
    opt_model = policy.module if isinstance(policy, DDP) else policy
    optimizer = create_optimizer(opt_model, cfg.training)
    total_steps = cfg.training.max_steps
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=cfg.training.warmup_steps, total_steps=total_steps
    )

    # ── 8. Resume ──
    global_step = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_bf16)

    if is_main_process():
        OmegaConf.save(cfg, output_dir / "config.yaml")
        total_params = sum(p.numel() for p in opt_model.parameters())
        trainable_params = sum(p.numel() for p in opt_model.parameters() if p.requires_grad)
        logger.info(
            f"Model: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable, "
            f"{(total_params-trainable_params)/1e6:.1f}M frozen"
        )

    resume_path = cfg.training.get("resume_from_checkpoint")
    if not resume_path:
        latest = find_latest_checkpoint(output_dir)
        if latest is not None:
            resume_path = str(latest)
            logger.info(f"Auto-resume: found latest checkpoint {latest}")

    if resume_path:
        resume_dir = Path(resume_path)
        if not resume_dir.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_dir}"
            )
        global_step = load_checkpoint(
            policy, optimizer, scheduler, resume_dir, device,
            scaler=scaler, train_loader=train_loader,
        )
        logger.info(f"Resumed from step {global_step}")
    barrier()  # All ranks synced before training starts

    # ── 9. Training loop ──
    policy.train()

    logger.info(
        f"Training: stage={stage}, steps={total_steps}, "
        f"bs={cfg.training.batch_size}×{get_world_size()} GPUs, "
        f"effective_bs={cfg.training.batch_size * get_world_size()}, "
        f"datasets={list(datasets_dict.keys())}"
    )

    # ── Stop file: touch outputs/.../STOP to gracefully stop training ──
    # This avoids SIGTERM/torchrun timeout issues with large checkpoint saves.
    stop_file = output_dir / "STOP"
    if stop_file.exists():
        stop_file.unlink()  # Clean stale stop file from previous run

    global _SHUTDOWN_REQUESTED
    try:
        for batch in train_loader:
            if global_step >= total_steps or _SHUTDOWN_REQUESTED:
                break
            if stop_file.exists() and is_main_process():
                logger.info(f"Stop file detected ({stop_file}), saving checkpoint and exiting...")
                _SHUTDOWN_REQUESTED = True
                break

            batch = _batch_to_device(batch, device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bf16):
                loss_dict = policy(batch)
                loss = loss_dict["loss"]

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if cfg.training.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in policy.parameters() if p.requires_grad],
                    cfg.training.max_grad_norm,
                )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            # Logging — all ranks participate in all_reduce, rank 0 logs
            if global_step % cfg.training.log_every == 0:
                if is_dist():
                    for k in list(loss_dict.keys()):
                        v = loss_dict[k]
                        if isinstance(v, torch.Tensor) and v.ndim == 0:
                            avg = v.detach().clone()
                            dist.all_reduce(avg, op=dist.ReduceOp.SUM)
                            avg /= get_world_size()
                            loss_dict[k] = avg
                if is_main_process():
                    extra = {}
                    ds_names = batch.get("dataset_name")
                    if ds_names and isinstance(ds_names, list):
                        extra["dataset"] = ds_names[0] if len(set(ds_names)) == 1 else "mixed"
                    _log_metrics(loss_dict, global_step, optimizer, extra=extra)

            # Checkpoint (rank 0 saves, all ranks wait)
            if global_step % cfg.training.save_every == 0:
                save_checkpoint(policy, optimizer, scheduler, global_step, output_dir,
                                scaler=scaler, train_loader=train_loader)

    finally:
        save_checkpoint(policy, optimizer, scheduler, global_step, output_dir,
                        scaler=scaler, train_loader=train_loader)
        if _SHUTDOWN_REQUESTED:
            logger.info(f"Graceful shutdown complete at step {global_step}.")
        else:
            logger.info(f"Training complete: {global_step} steps.")
        wandb_finish()
        cleanup_distributed()


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {
                sk: sv.to(device, non_blocking=True) if isinstance(sv, torch.Tensor) else sv
                for sk, sv in v.items()
            }
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [t.to(device, non_blocking=True) for t in v]
        else:
            out[k] = v
    return out


def _log_metrics(loss_dict, step, optimizer, extra: dict = None):
    parts = [f"step={step}"]
    wb_metrics = {"train/step": step}

    if extra:
        for k, v in extra.items():
            parts.append(f"{k}={v}")

    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor) and v.ndim == 0:
            val = v.item()
            parts.append(f"{k}={val:.4f}")
            wb_metrics[f"train/{k}"] = val

    lr_strs = []
    for i, pg in enumerate(optimizer.param_groups):
        group_name = pg.get("group_name", f"g{i}")
        lr_strs.append(f"{group_name}={pg['lr']:.2e}")
        wb_metrics[f"lr/{group_name}"] = pg["lr"]
    parts.append(f"lr=[{', '.join(lr_strs)}]")

    if torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        parts.append(f"gpu_mem={mem_gb:.1f}GB")
        wb_metrics["system/gpu_mem_gb"] = mem_gb

    logger.info(" | ".join(parts))
    wandb_log(wb_metrics, step=step)


# ═══════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="AffordanceVLA Training")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument(
        "--mode", type=str, default="debug", choices=["debug", "train"],
        help="debug: minimal data (1 task/suite per dataset); "
             "train: full data (all tasks/suites/splits)",
    )
    parser.add_argument(
        "--force_rescan", action="store_true", default=False,
        help="Ignore cached frame indices and rebuild from scratch",
    )
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if unknown:
        overrides = OmegaConf.from_dotlist(unknown)
        cfg = OmegaConf.merge(cfg, overrides)

    if args.force_rescan:
        cfg.data.force_rescan = True

    # Inject mode into config so create_datasets can access it
    cfg.mode = args.mode

    if int(os.environ.get("RANK", 0)) == 0:
        print(f"Mode: {args.mode}")
        print(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    train(cfg)


if __name__ == "__main__":
    main()
