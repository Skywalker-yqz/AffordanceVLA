"""Resolve the paper's effective batch size and step/epoch training budgets."""

import math


def resolve_training_schedule(training, world_size, dataset_sizes=None, weights=None):
    """Return accumulation steps, global batch size, and optimizer-step budget.

    ``batch_size`` is the per-device microbatch. ``effective_batch_size`` stays
    fixed when the number of devices changes. Epoch schedules are defined by
    the active dataset size and the number of examples per optimizer update.
    """
    micro_batch = int(training.get("batch_size", 1))
    world_size = int(world_size)
    if micro_batch <= 0 or world_size <= 0:
        raise ValueError("batch_size and world_size must be positive")
    device_batch = micro_batch * world_size
    target_batch = training.get("effective_batch_size")
    if target_batch is not None:
        effective_batch = int(target_batch)
        if effective_batch <= 0 or effective_batch % device_batch:
            raise ValueError(
                "effective_batch_size must be a positive multiple of "
                "batch_size × world_size; reduce the per-device batch_size"
            )
        accumulation = effective_batch // device_batch
    else:
        accumulation = int(training.get("gradient_accumulation_steps", 1))
        if accumulation <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        effective_batch = device_batch * accumulation

    steps = training.get("max_steps")
    if steps is None:
        epochs = training.get("num_epochs")
        if epochs is None or float(epochs) <= 0:
            raise ValueError("Set a positive max_steps or num_epochs")
        active_sizes = [
            int(size) for name, size in (dataset_sizes or {}).items()
            if weights is None or float(weights.get(name, 0)) > 0
        ]
        if not active_sizes or sum(active_sizes) <= 0:
            raise ValueError("Epoch training requires a nonempty active dataset")
        steps = math.ceil(sum(active_sizes) * float(epochs) / effective_batch)
    if int(steps) <= 0:
        raise ValueError("max_steps must be positive")
    return accumulation, effective_batch, int(steps)
