"""
Optimizer with per-module learning rates for AffordanceVLA.

  - Uses transformers' get_parameter_names + ALL_LAYERNORM_LAYERS for decay/no-decay split
  - 4 parameter groups × 2 (decay/no-decay) = up to 8 groups
  - Per-group learning rates controlled via training config

Parameter Groups:
  Group 0 — Vision Encoder:  SigLIP vision tower + multi-modal projector
  Group 1 — Understanding Expert:  PaliGemma language model (excluding vision)
  Group 2 — Generation Expert:  Gemma gen expert + affordance queries + all decoders
  Group 3 — Action Expert:  Gemma action expert + state/action projection modules
"""

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names


def create_optimizer(opt_model, args):
    """Create AdamW optimizer with per-module LRs and decay/no-decay split.

    Uses transformers' get_parameter_names to identify LayerNorm/bias
    parameters and zero out their weight decay.

    Args:
        opt_model: AffordanceVLAPolicy instance.
        args: Training config (OmegaConf) with lr fields:
            vision_encoder_lr, und_expert_lr, gen_expert_lr, act_expert_lr,
            weight_decay, adam_beta1, adam_beta2, adam_eps.

    Returns:
        torch.optim.AdamW optimizer.
    """
    # Identify parameters that should NOT receive weight decay
    decay_parameters = get_parameter_names(
        opt_model, ALL_LAYERNORM_LAYERS, ["bias", "layernorm", "rmsnorm"]
    )
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    # ── Group 0: Vision Encoder (SigLIP + multi-modal projector) ──
    vision_encoder_parameters = [
        name for name, _ in opt_model.named_parameters()
        if any(x in name for x in [
            "paligemma_with_expert.paligemma.vision_tower",
            "paligemma_with_expert.paligemma.multi_modal_projector",
        ])
    ]

    # ── Group 1: Understanding Expert (PaliGemma LM, excluding vision) ──
    und_expert_parameters = [
        name for name, _ in opt_model.named_parameters()
        if "paligemma_with_expert.paligemma" in name
        and name not in vision_encoder_parameters
    ]

    # ── Group 2: Generation Expert + Affordance Queries + Decoders ──
    gen_expert_parameters = [
        name for name, _ in opt_model.named_parameters()
        if "paligemma_with_expert.gemma_gen_expert" in name
    ]
    gen_module_parameters = [
        name for name, _ in opt_model.named_parameters()
        if any(x in name for x in [
            "affordance_query_module",
            "which2act_decoder.cls_head",
            "which2act_decoder.proj_head",
            "wrist_decoder.cls_head",
            "wrist_decoder.proj_head",
            "where2act_decoder",
            "how2act_decoder",
        ])
    ]

    # ── Group 3: Action Expert + Action Modules ──
    act_expert_parameters = [
        name for name, _ in opt_model.named_parameters()
        if "paligemma_with_expert.gemma_expert" in name
    ]
    action_parameters = [
        name for name, _ in opt_model.named_parameters()
        if any(x in name for x in [
            "state_proj", "action_in_proj", "action_out_proj",
            "action_time_mlp_in", "action_time_mlp_out",
        ])
    ]

    # ── Build optimizer groups (decay + no-decay for each) ──
    optimizer_grouped_parameters = []

    def _add_group(param_names, lr):
        """Add decay + no-decay sub-groups for a set of parameter names."""
        decay_params = [
            p for n, p in opt_model.named_parameters()
            if n in decay_parameters and n in param_names and p.requires_grad
        ]
        nodecay_params = [
            p for n, p in opt_model.named_parameters()
            if n not in decay_parameters and n in param_names and p.requires_grad
        ]
        if decay_params:
            optimizer_grouped_parameters.append({
                "params": decay_params,
                "weight_decay": args.weight_decay,
                "lr": lr,
            })
        if nodecay_params:
            optimizer_grouped_parameters.append({
                "params": nodecay_params,
                "weight_decay": 0.0,
                "lr": lr,
            })

    # Group 0: Vision encoder
    _add_group(vision_encoder_parameters, args.vision_encoder_lr)

    # Group 1: Understanding expert
    _add_group(und_expert_parameters, args.und_expert_lr)

    # Group 2: Generation expert + affordance modules
    _add_group(
        gen_expert_parameters + gen_module_parameters,
        args.gen_expert_lr,
    )

    # Group 3: Action expert + action modules
    _add_group(
        act_expert_parameters + action_parameters,
        args.act_expert_lr,
    )

    import torch
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )

    return optimizer
