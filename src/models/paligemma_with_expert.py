"""
Mixture-of-Transformer (MoT) backbone for AffordanceVLA.

Runs multiple expert models with shared attention computation:
  - Understanding Expert (PaliGemma): SigLIP vision encoder + Gemma language model
  - Generation Expert (Gemma): processes affordance learnable queries
  - Action Expert (Gemma): processes state + action embeddings

In each transformer layer:
  1. Per-expert: LayerNorm → Q/K/V projection
  2. Shared: concatenate Q/K/V across all active experts → joint attention → split back
  3. Per-expert: O-projection → residual → post-attention LayerNorm → MLP → residual
"""

from typing import List, Optional, Union

import torch
from torch import nn
from transformers import (
    GemmaForCausalLM,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)


def apply_rope(x, positions, max_wavelength=10_000):
    """Apply Rotary Position Embeddings (RoPE) to input tensor.

    Args:
        x: [B, L, H, D] query or key states.
        positions: [B, L] position indices.
        max_wavelength: Maximum wavelength for the sinusoidal basis.

    Returns:
        [B, L, H, D] tensor with RoPE applied.
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(
        d_half, dtype=torch.float32, device=device
    )
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(
        torch.float32
    )
    radians = radians[..., None, :]

    sin = torch.sin(radians)
    cos = torch.cos(radians)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


class PaliGemmaWithExpertModel(PreTrainedModel):
    """MoT backbone with shared attention across expert transformer models.

    Supports 2 experts (understanding + action) or 3 experts
    (understanding + generation + action) depending on config.
    """

    config_class = PretrainedConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_flash_attn_2 = False
    _supports_sdpa = True

    def __init__(self, config: PretrainedConfig):
        super().__init__(config=config)
        self.config = config

        # Understanding Expert (PaliGemma: SigLIP + Gemma LM)
        self.paligemma = PaliGemmaForConditionalGeneration(
            config=config.und_expert_config
        )

        # Action Expert (Gemma)
        self.gemma_expert = GemmaForCausalLM(config=config.act_expert_config)
        self.gemma_expert.model.embed_tokens = None  # not needed

        # Generation Expert (Gemma) — for affordance queries
        if (
            hasattr(self.config, "gen_expert_config")
            and self.config.gen_expert_config is not None
        ):
            self.gemma_gen_expert = GemmaForCausalLM(config=config.gen_expert_config)
            self.gemma_gen_expert.model.embed_tokens = None  # not needed

        self.is_causal = False
        self.num_key_value_heads = (
            config.und_expert_config.text_config.num_key_value_heads
        )
        self.num_key_value_groups = (
            config.und_expert_config.text_config.num_attention_heads
            // self.num_key_value_heads
        )

        # Freeze control state
        self.freeze_vision_encoder = True
        self.train_act_expert_only = False
        self.train_gen_expert_only = False
        self.freeze_gen_expert = False

        self.to_bfloat16_like_physical_intelligence()

    # ─── Gradient / Freeze Control ───────────────────────────────────

    def set_requires_grad(
        self,
        freeze_vision_encoder=False,
        freeze_gen_expert=False,
        train_act_expert_only=False,
        train_gen_expert_only=False,
    ):
        """Configure which parts of the model are trainable."""
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_act_expert_only = train_act_expert_only
        self.train_gen_expert_only = train_gen_expert_only
        self.freeze_gen_expert = freeze_gen_expert

        if freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
            for param in self.paligemma.vision_tower.parameters():
                param.requires_grad = False

        if freeze_gen_expert and hasattr(self, "gemma_gen_expert"):
            self.gemma_gen_expert.eval()
            for param in self.gemma_gen_expert.parameters():
                param.requires_grad = False

        if train_act_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

        if hasattr(self, "gemma_gen_expert") and train_gen_expert_only:
            self.paligemma.eval()
            self.gemma_expert.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False
            for param in self.gemma_expert.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
        if self.train_act_expert_only:
            self.paligemma.eval()
        if self.train_gen_expert_only:
            self.paligemma.eval()
            self.gemma_expert.eval()
        if self.freeze_gen_expert and hasattr(self, "gemma_gen_expert"):
            self.gemma_gen_expert.eval()

    # ─── Dtype Management ────────────────────────────────────────────

    def to_bfloat16_like_physical_intelligence(self):
        """Cast transformer layers and vision tower to bfloat16."""
        self.paligemma = self.paligemma.to(dtype=torch.bfloat16)
        params_to_change_dtype = [
            "language_model.model.layers",
            "gemma_expert.model.layers",
            "gemma_gen_expert.model.layers",
            "vision_tower",
            "multi_modal",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    # ─── Embedding Helpers ───────────────────────────────────────────

    def embed_image(self, image: torch.Tensor):
        """Extract image features via SigLIP vision encoder."""
        return self.paligemma.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        """Embed language tokens via PaliGemma's token embedding layer."""
        return self.paligemma.language_model.model.embed_tokens(tokens)

    # ─── Shared Attention Forward ────────────────────────────────────

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], dict]] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
        cat_past_key_values: Optional[bool] = False,
        return_attention_weights: Optional[bool] = False,
    ):
        """Forward pass with shared attention across experts.

        Args:
            attention_mask: [B, L, L] 2D boolean attention mask.
            position_ids: [B, L] position indices for RoPE.
            past_key_values: Cached KV states (dict keyed by layer index).
            inputs_embeds: List of per-expert embeddings [und, gen, act].
                Each element can be None if that expert is inactive.
            use_cache: Whether to use/create KV cache.
            fill_kv_cache: If True, store KV to cache (first pass).
                If False, read from cache and optionally append.
            cat_past_key_values: If True, concatenate new KV to existing cache
                (for autoregressive generation).
            return_attention_weights: If True, collect per-layer attention
                weights (eager mode only). Returns list of (B, H, L_q, L_kv).

        Returns:
            (outputs_embeds_list, past_key_values) or
            (outputs_embeds_list, past_key_values, all_attention_weights).
        """
        # Resolve model list based on available experts
        if hasattr(self, "gemma_gen_expert"):
            models = [
                self.paligemma.language_model.model,
                self.gemma_gen_expert.model,
                self.gemma_expert.model,
            ]
        else:
            models = [
                self.paligemma.language_model.model,
                self.gemma_expert.model,
            ]

        # Determine batch size from first non-None input
        batch_size = None
        for hidden_states in inputs_embeds:
            if hidden_states is not None:
                batch_size = hidden_states.shape[0]
                break

        num_layers = self.paligemma.config.text_config.num_hidden_layers
        head_dim = self.paligemma.config.text_config.head_dim

        all_attention_weights = [] if return_attention_weights else None

        for layer_idx in range(num_layers):
            # ── Per-expert: LayerNorm + Q/K/V ──
            query_states = []
            key_states = []
            value_states = []

            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue
                layer = models[i].layers[layer_idx]
                hidden_states = layer.input_layernorm(hidden_states)

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

                hidden_states = hidden_states.to(dtype=torch.bfloat16)
                query_states.append(
                    layer.self_attn.q_proj(hidden_states).view(hidden_shape)
                )
                key_states.append(
                    layer.self_attn.k_proj(hidden_states).view(hidden_shape)
                )
                value_states.append(
                    layer.self_attn.v_proj(hidden_states).view(hidden_shape)
                )

            # ── Shared: Concatenate across experts ──
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)

            # Apply RoPE
            query_states = apply_rope(query_states, position_ids)
            key_states = apply_rope(key_states, position_ids)

            # KV cache management
            if use_cache and past_key_values is None:
                past_key_values = {}

            if use_cache:
                if fill_kv_cache:
                    past_key_values[layer_idx] = {
                        "key_states": key_states,
                        "value_states": value_states,
                    }
                else:
                    key_states = torch.cat(
                        [past_key_values[layer_idx]["key_states"], key_states],
                        dim=1,
                    )
                    value_states = torch.cat(
                        [past_key_values[layer_idx]["value_states"], value_states],
                        dim=1,
                    )
                    if cat_past_key_values:
                        past_key_values[layer_idx]["key_states"] = key_states
                        past_key_values[layer_idx]["value_states"] = value_states

            # ── Shared: Attention ──
            if self.config.attention_implementation == "eager":
                eager_result = self.eager_attention_forward(
                    attention_mask,
                    batch_size,
                    head_dim,
                    query_states,
                    key_states,
                    value_states,
                    return_attention_weights=return_attention_weights,
                )
                if return_attention_weights:
                    att_output, layer_attn_weights = eager_result
                    all_attention_weights.append(layer_attn_weights)
                else:
                    att_output = eager_result
            elif self.config.attention_implementation == "sdpa":
                # Expand K/V heads to match Q heads for SDPA compatibility
                # Q: (B, L, num_att_heads, D) → K/V: (B, L, num_kv_heads, D)
                # Need to expand K/V when num_kv_heads < num_att_heads (GQA)
                seq_len_kv = key_states.shape[1]
                if self.num_key_value_groups > 1:
                    key_states_expanded = (
                        key_states[:, :, :, None, :]
                        .expand(batch_size, seq_len_kv, self.num_key_value_heads,
                                self.num_key_value_groups, head_dim)
                        .reshape(batch_size, seq_len_kv,
                                 self.num_key_value_heads * self.num_key_value_groups, head_dim)
                    )
                    value_states_expanded = (
                        value_states[:, :, :, None, :]
                        .expand(batch_size, seq_len_kv, self.num_key_value_heads,
                                self.num_key_value_groups, head_dim)
                        .reshape(batch_size, seq_len_kv,
                                 self.num_key_value_heads * self.num_key_value_groups, head_dim)
                    )
                else:
                    key_states_expanded = key_states
                    value_states_expanded = value_states

                att_output = torch.nn.functional.scaled_dot_product_attention(
                    query=query_states.permute(0, 2, 1, 3),
                    key=key_states_expanded.permute(0, 2, 1, 3),
                    value=value_states_expanded.permute(0, 2, 1, 3),
                    attn_mask=attention_mask[:, None, :, :],
                    dropout_p=0.0,
                    is_causal=False,
                )
                att_output = att_output.permute(0, 2, 1, 3)
                att_output = att_output.reshape(
                    batch_size,
                    -1,
                    self.num_key_value_heads * self.num_key_value_groups * head_dim,
                )
            else:
                raise ValueError(
                    f"Unsupported attention implementation: "
                    f"{self.config.attention_implementation}"
                )

            # ── Per-expert: O-proj + residual + FFN + residual ──
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = models[i].layers[layer_idx]

                if hidden_states is not None:
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start:end])

                    # First residual connection
                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    # Post-attention norm + MLP
                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    # Second residual connection
                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)
                    start = end
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # ── Final LayerNorm per expert ──
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                outputs_embeds.append(models[i].norm(hidden_states))
            else:
                outputs_embeds.append(None)

        if return_attention_weights:
            return outputs_embeds, past_key_values, all_attention_weights
        return outputs_embeds, past_key_values

    # ─── Eager Attention (with GQA) ──────────────────────────────────

    def eager_attention_forward(
        self,
        attention_mask,
        batch_size,
        head_dim,
        query_states,
        key_states,
        value_states,
        return_attention_weights=False,
    ):
        """Manual attention computation with Grouped Query Attention support.

        Upcasts to float32 for numerical stability, applies GQA expansion
        for key/value heads, and uses explicit masking.

        When return_attention_weights=True, returns (att_output, probs) where
        probs is (B, num_heads, L_q, L_kv) softmax attention weights.
        """
        num_att_heads = (
            self.config.und_expert_config.text_config.num_attention_heads
        )
        num_key_value_heads = (
            self.config.und_expert_config.text_config.num_key_value_heads
        )
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]

        # GQA: expand key/value heads to match query heads
        key_states = (
            key_states[:, :, :, None, :]
            .expand(
                batch_size,
                sequence_length,
                num_key_value_heads,
                num_key_value_groups,
                head_dim,
            )
            .reshape(
                batch_size,
                sequence_length,
                num_key_value_heads * num_key_value_groups,
                head_dim,
            )
        )
        value_states = (
            value_states[:, :, :, None, :]
            .expand(
                batch_size,
                sequence_length,
                num_key_value_heads,
                num_key_value_groups,
                head_dim,
            )
            .reshape(
                batch_size,
                sequence_length,
                num_key_value_heads * num_key_value_groups,
                head_dim,
            )
        )

        # Attention weights
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5

        if att_weights.dtype == torch.float32:
            big_neg = -2.3819763e38  # Gemma-specific large negative
        elif att_weights.dtype in (torch.bfloat16, torch.float16):
            big_neg = -1e9
        else:
            raise ValueError(f"Unsupported dtype: {att_weights.dtype}")

        masked_att_weights = att_weights.masked_fill(
            ~attention_mask[:, None, :, :], big_neg
        )
        probs = nn.functional.softmax(masked_att_weights, dim=-1).to(
            dtype=value_states.dtype
        )

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
        att_output = att_output.permute(0, 2, 1, 3)
        att_output = att_output.reshape(
            batch_size,
            -1,
            num_key_value_heads * num_key_value_groups * head_dim,
        )

        if return_attention_weights:
            return att_output, probs.detach().float()
        return att_output
