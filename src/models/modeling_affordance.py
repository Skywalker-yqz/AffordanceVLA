"""
AffordanceVLA core model — AffordanceFlowMatching.

Architecture overview (Mixture-of-Transformer with shared attention):
  1. Understanding Expert (PaliGemma): embeds current-frame images + language
  2. Generation Expert (Gemma):        processes Learnable Affordance Queries
     → outputs three token groups: Which2Act, Where2Act, How2Act
  3. Action Expert (Gemma):            state + noisy actions → flow matching

Design highlights:
  - AffordanceQueryModule supplies a fixed set of learnable queries (instead
    of autoregressive multi-scale image-token generation) to drive the
    Generation Expert
  - Three-group affordance output (Which2Act / Where2Act / How2Act, plus an
    optional Wrist group) consumed by independent decoders
  - Single-pass training forward across the three experts via MoT shared
    attention; two-phase inference with KV cache for the action expert

Design references for AffordanceQueryModule:
  - BLIP-2 Q-Former: nn.Parameter + normal_(0, 0.02) initialization
  - Seer PerceiverResampler: learnable latents as queries in shared attention
  - DETR: object queries with learned position embeddings
  - AURORA Perception Tokens: multi-task token groups for visual reasoning
"""

from dataclasses import dataclass
from typing import List

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.models.paligemma_with_expert import PaliGemmaWithExpertModel
from src.utils.model_utils import create_sinusoidal_pos_embedding, make_att_2d_masks


# ═══════════════════════════════════════════════════════════════════════
# Output Dataclass
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class AffordanceOutput:
    """Output from AffordanceFlowMatching model.

    Attributes:
        action_loss: Per-element MSE loss for flow matching (training only).
        actions: Predicted/denoised action tensor.
        which2act_tokens: Generation expert output for Which2Act group.
        wrist_tokens: Generation expert output for Wrist group (only when
            control_wristtoken=True, else None).
        where2act_tokens: Generation expert output for Where2Act group.
        how2act_tokens: Generation expert output for How2Act group.
        gen_attention_weights: Per-layer attention weights from Phase 1
            (affordance query forward). List of (B, H, L_q, L_kv) tensors.
        act_attention_weights: Per-layer attention weights from the last
            denoise step (action forward). List of (B, H, L_q, L_kv) tensors.
    """

    action_loss: torch.Tensor | None = None
    actions: torch.Tensor | None = None
    which2act_tokens: torch.Tensor | None = None
    wrist_tokens: torch.Tensor | None = None
    where2act_tokens: torch.Tensor | None = None
    how2act_tokens: torch.Tensor | None = None
    gen_attention_weights: list | None = None
    act_attention_weights: list | None = None


# ═══════════════════════════════════════════════════════════════════════
# Learnable Affordance Queries
# ═══════════════════════════════════════════════════════════════════════


class AffordanceQueryModule(nn.Module):
    """Learnable Affordance Queries for the Generation Expert.

    These queries replace the VQ-VAE-encoded history-frame tokens used by
    prior autoregressive VLA designs. They are fed as input_embeds to the
    Generation Expert, attending to the Understanding Expert's image /
    language representations through the MoT shared attention mechanism.

    Token layout (control_wristtoken=False):
      [which2act | where2act | how2act]

    Token layout (control_wristtoken=True):
      [which2act | wrist | where2act | how2act]

    Args:
        config: AffordanceVLAConfig with token counts and init std.
    """

    def __init__(self, config):
        super().__init__()

        self.which2act_num_tokens = config.which2act_num_tokens
        self.where2act_num_tokens = config.where2act_num_tokens
        self.how2act_num_tokens = config.how2act_num_tokens
        self.control_wristtoken = config.control_wristtoken
        self.wrist_num_tokens = (
            config.which2act_num_tokens if self.control_wristtoken else 0
        )

        self.total_num_tokens = (
            self.which2act_num_tokens
            + self.wrist_num_tokens
            + self.where2act_num_tokens
            + self.how2act_num_tokens
        )

        gen_hidden_dim = config.gen_expert_config.hidden_size
        init_std = config.query_init_std

        self.affordance_queries = nn.Parameter(
            torch.empty(1, self.total_num_tokens, gen_hidden_dim)
        )
        nn.init.normal_(self.affordance_queries, mean=0.0, std=init_std)

        self.query_pos_embed = nn.Parameter(
            torch.empty(1, self.total_num_tokens, gen_hidden_dim)
        )
        nn.init.normal_(self.query_pos_embed, mean=0.0, std=init_std)

        self.which2act_type_embed = nn.Parameter(
            torch.zeros(1, 1, gen_hidden_dim)
        )
        if self.control_wristtoken:
            self.wrist_type_embed = nn.Parameter(
                torch.zeros(1, 1, gen_hidden_dim)
            )
        self.where2act_type_embed = nn.Parameter(
            torch.zeros(1, 1, gen_hidden_dim)
        )
        self.how2act_type_embed = nn.Parameter(
            torch.zeros(1, 1, gen_hidden_dim)
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        """Generate affordance query embeddings for a batch.

        Args:
            batch_size: Number of samples in the current batch.

        Returns:
            (batch_size, total_num_tokens, gen_hidden_dim) query embeddings
            with position and type information baked in.
        """
        queries = self.affordance_queries + self.query_pos_embed

        n_w2a_look = self.which2act_num_tokens
        n_wrist = self.wrist_num_tokens
        n_w2a = self.where2act_num_tokens

        offset = 0
        w2a_look = queries[:, offset : offset + n_w2a_look] + self.which2act_type_embed
        offset += n_w2a_look

        parts = [w2a_look]
        if self.control_wristtoken:
            wrist = queries[:, offset : offset + n_wrist] + self.wrist_type_embed
            offset += n_wrist
            parts.append(wrist)

        w2a = queries[:, offset : offset + n_w2a] + self.where2act_type_embed
        offset += n_w2a
        parts.append(w2a)

        h2a = queries[:, offset :] + self.how2act_type_embed
        parts.append(h2a)

        queries = torch.cat(parts, dim=1)
        return queries.expand(batch_size, -1, -1)


# ═══════════════════════════════════════════════════════════════════════
# Core Model
# ═══════════════════════════════════════════════════════════════════════


class AffordanceFlowMatching(nn.Module):
    """AffordanceVLA core model combining three experts via MoT.

    Training forward (single pass):
      1. embed_prefix  → Understanding Expert embeddings (images + language)
      2. AffordanceQueryModule → Generation Expert embeddings (learnable queries)
      3. embed_suffix  → Action Expert embeddings (state + noisy_actions + time)
      4. PaliGemmaWithExpertModel → shared attention across all three
      5. Split gen output → Which2Act, Where2Act, How2Act tokens
      6. Action loss via flow matching MSE

    Inference:
      1–2. Forward understanding + generation with KV cache
      3–4. Iterative Euler denoising for actions (using cached KV)
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        # MoT backbone with three experts
        self.paligemma_with_expert = PaliGemmaWithExpertModel(config)

        # Affordance query module (replaces VQ-VAE world model inputs)
        self.affordance_query_module = AffordanceQueryModule(config)

        # Action modules (flow matching)
        self.state_proj = nn.Linear(config.max_state_dim, config.proj_width)
        self.action_in_proj = nn.Linear(config.max_action_dim, config.proj_width)
        self.action_out_proj = nn.Linear(config.proj_width, config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(
            config.proj_width * 2, config.proj_width
        )
        self.action_time_mlp_out = nn.Linear(config.proj_width, config.proj_width)

        # Apply freeze control
        training_args = kwargs.get("training_args", None)
        self.set_requires_grad(training_args)

    # ─── Freeze Control ──────────────────────────────────────────────

    def set_requires_grad(self, training_args=None):
        """Configure which modules are trainable based on training args."""
        if training_args is None:
            self.freeze_vision_encoder = True
            self.freeze_gen_expert = False
            self.train_act_expert_only = False
            self.train_gen_expert_only = False
            self.train_state_proj = True
        else:
            self.freeze_vision_encoder = training_args.freeze_vision_encoder
            self.train_act_expert_only = training_args.train_act_expert_only
            self.train_gen_expert_only = training_args.train_gen_expert_only
            self.train_state_proj = training_args.train_state_proj
            self.freeze_gen_expert = training_args.freeze_gen_expert

        self.paligemma_with_expert.set_requires_grad(
            freeze_vision_encoder=self.freeze_vision_encoder,
            freeze_gen_expert=self.freeze_gen_expert,
            train_act_expert_only=self.train_act_expert_only,
            train_gen_expert_only=self.train_gen_expert_only,
        )

        for param in self.state_proj.parameters():
            param.requires_grad = self.train_state_proj

        if training_args is not None and training_args.train_gen_expert_only:
            freeze_modules = [
                "state_proj",
                "action_in_proj",
                "action_out_proj",
                "action_time_mlp_in",
                "action_time_mlp_out",
            ]
            for name, param in self.named_parameters():
                if any(x in name for x in freeze_modules):
                    param.requires_grad = False

    # ─── Noise / Time Sampling ───────────────────────────────────────

    def sample_noise(self, shape, device):
        """Sample Gaussian noise for flow matching."""
        return torch.normal(
            mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device
        )

    def sample_time(self, bsize, device):
        """Sample timesteps from Beta(1.5, 1.0) distribution in (0.001, 1)."""
        beta_dist = torch.distributions.Beta(
            concentration1=1.5, concentration0=1.0
        )
        time_beta = beta_dist.sample((bsize,)).to(
            device=device, dtype=torch.float32
        )
        return time_beta * 0.999 + 0.001

    # ─── Embedding Methods ───────────────────────────────────────────

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images + language for Understanding Expert.

        Args:
            images: List of image tensors [B, C, H, W] per camera.
            img_masks: List of boolean masks [B] per camera.
            lang_tokens: [B, L] tokenized language input.
            lang_masks: [B, L] boolean language attention mask.

        Returns:
            (embs, pad_masks, att_masks):
                embs: [B, N_und, D_und] concatenated embeddings.
                pad_masks: [B, N_und] boolean validity mask.
                att_masks: [B, N_und] block-group mask (all 0 = bidirectional).
        """
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=False):
            img_emb = self.paligemma_with_expert.embed_image(img)

            # Normalize image embeddings (match PaliGemma convention)
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(
                img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device
            )

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_masks += [0] * num_img_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(
        self, state, noisy_actions, timestep
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed state + noisy actions + timestep for Action Expert.

        Uses sinusoidal positional encoding for timestep, then fuses with
        action embeddings via MLP.

        Args:
            state: [B, max_state_dim] robot state.
            noisy_actions: [B, chunk_size, max_action_dim] noisy action targets.
            timestep: [B] flow matching timestep values.

        Returns:
            (embs, pad_masks, att_masks):
                embs: [B, 1 + chunk_size, proj_width] concatenated embeddings.
                pad_masks: [B, 1 + chunk_size] boolean validity mask.
                att_masks: [B, 1 + chunk_size] block-group mask.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # State embedding
        state_emb = self.state_proj(state)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        dtype = state_emb.dtype
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1]  # state starts a new causal block

        # Timestep embedding (sinusoidal, sensitive to [0, 1] range)
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.config.proj_width,
            min_period=4e-3,
            max_period=4.0,
            device=device,
        )
        time_emb = time_emb.to(dtype=dtype)

        # Fuse action + timestep via MLP
        action_emb = self.action_in_proj(noisy_actions)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=device
        )
        pad_masks.append(action_time_mask)
        # Action tokens: first starts new block, rest continue same block
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, device=device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    # ─── Affordance Mask / Split Helpers ─────────────────────────────

    def _build_gen_masks(self, bsize, device):
        """Build padding and attention masks for affordance queries.

        The queries start a new causal block (att_mask=1 for first token)
        so that Understanding Expert tokens cannot attend to queries,
        but queries CAN attend to Understanding Expert tokens.

        Returns:
            (gen_pad_masks, gen_att_masks): both [B, total_tokens].
        """
        total_tokens = self.affordance_query_module.total_num_tokens
        gen_pad_masks = torch.ones(
            bsize, total_tokens, dtype=torch.bool, device=device
        )
        gen_att_masks_list = [1] + [0] * (total_tokens - 1)
        gen_att_masks = torch.tensor(gen_att_masks_list, device=device)
        gen_att_masks = gen_att_masks[None, :].expand(bsize, -1)
        return gen_pad_masks, gen_att_masks

    def split_affordance_outputs(self, gen_out):
        """Split generation expert output into affordance token groups.

        Token layout matches AffordanceQueryModule:
          [which2act | (wrist) | where2act | how2act]

        Args:
            gen_out: [B, total_tokens, D] generation expert output.

        Returns:
            dict with keys 'which2act', 'wrist' (or None), 'where2act', 'how2act'.
        """
        qm = self.affordance_query_module
        offset = 0

        which2act = gen_out[:, offset : offset + qm.which2act_num_tokens]
        offset += qm.which2act_num_tokens

        if qm.control_wristtoken:
            wrist = gen_out[:, offset : offset + qm.wrist_num_tokens]
            offset += qm.wrist_num_tokens
        else:
            wrist = None

        where2act = gen_out[:, offset : offset + qm.where2act_num_tokens]
        offset += qm.where2act_num_tokens

        how2act = gen_out[:, offset :]

        return {
            "which2act": which2act,
            "wrist": wrist,
            "where2act": where2act,
            "how2act": how2act,
        }

    # ─── Training Forward ────────────────────────────────────────────

    def forward(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> AffordanceOutput:
        """Training forward pass (single-pass through all three experts).

        Block-causal attention structure:
          Block 0: Understanding tokens (images + language) — bidirectional
          Block 1: Affordance queries — attend to Block 0 + self
          Block 2: State — attends to Blocks 0-2
          Block 3: Action chunk — attends to everything

        Args:
            images: List of [B, C, H, W] observation images.
            img_masks: List of [B] per-camera validity masks.
            lang_tokens: [B, L] tokenized task text.
            lang_masks: [B, L] language attention mask.
            state: [B, max_state_dim] robot state.
            actions: [B, chunk_size, max_action_dim] ground-truth actions.
            noise: Optional pre-sampled noise.
            time: Optional pre-sampled timesteps.

        Returns:
            AffordanceOutput with action_loss and three affordance token groups.
        """
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        if time is None:
            time = self.sample_time(bsize, device)

        # Flow matching interpolation: x_t = t·noise + (1-t)·actions
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions  # target velocity field

        # 1. Understanding Expert embeddings
        und_embs, und_pad_masks, und_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        # 2. Generation Expert: Learnable Affordance Queries
        gen_embs = self.affordance_query_module(bsize)
        gen_pad_masks, gen_att_masks = self._build_gen_masks(bsize, device)

        # 3. Action Expert embeddings
        act_embs, act_pad_masks, act_att_masks = self.embed_suffix(
            state, x_t, time
        )

        # Combine masks across all experts (cast to long for concatenation)
        pad_masks = torch.cat(
            [und_pad_masks, gen_pad_masks, act_pad_masks], dim=1
        )
        att_masks = torch.cat(
            [
                und_att_masks.to(torch.long),
                gen_att_masks.to(torch.long),
                act_att_masks.to(torch.long),
            ],
            dim=1,
        )
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # 4. Single-pass MoT shared attention
        (und_out, gen_out, act_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[und_embs, gen_embs, act_embs],
            use_cache=False,
            fill_kv_cache=False,
        )

        # 5. Split generation expert output into affordance token groups
        affordance_outputs = self.split_affordance_outputs(gen_out)

        # 6. Action loss via flow matching
        act_out = act_out[:, -self.config.chunk_size :]
        act_out = act_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(act_out)
        action_loss = F.mse_loss(u_t, v_t, reduction="none")

        return AffordanceOutput(
            action_loss=action_loss,
            actions=v_t,
            which2act_tokens=affordance_outputs["which2act"],
            wrist_tokens=affordance_outputs["wrist"],
            where2act_tokens=affordance_outputs["where2act"],
            how2act_tokens=affordance_outputs["how2act"],
        )

    # ─── Inference ───────────────────────────────────────────────────

    @torch.no_grad()
    def sample_actions(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> AffordanceOutput:
        """Inference: generate affordance tokens + denoise actions.

        Two-phase forward:
          Phase 1: Understanding + Generation experts → KV cache + affordance tokens
          Phase 2: Iterative Euler denoising with Action expert (reads cached KV)

        Args:
            images, img_masks, lang_tokens, lang_masks, state: as in forward().
            noise: Optional pre-sampled noise for action denoising.
            return_attention_weights: If True, collect attention weights from
                Phase 1 (gen_attention_weights) and the last denoise step
                (act_attention_weights). Requires eager attention mode.

        Returns:
            AffordanceOutput with denoised actions and affordance token groups.
        """
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )
            noise = self.sample_noise(actions_shape, device)

        # Phase 1: Forward understanding + affordance queries
        und_embs, und_pad_masks, und_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        gen_embs = self.affordance_query_module(bsize)
        gen_pad_masks, gen_att_masks = self._build_gen_masks(bsize, device)

        prefix_pad_masks = torch.cat([und_pad_masks, gen_pad_masks], dim=1)
        prefix_att_masks = torch.cat(
            [und_att_masks.to(torch.long), gen_att_masks.to(torch.long)],
            dim=1,
        )
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        phase1_result = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[und_embs, gen_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
            return_attention_weights=return_attention_weights,
        )
        if return_attention_weights:
            (_, gen_out, _), past_key_values, gen_attn_weights = phase1_result
        else:
            (_, gen_out, _), past_key_values = phase1_result
            gen_attn_weights = None

        affordance_outputs = self.split_affordance_outputs(gen_out)

        # Phase 2: Euler denoising for actions
        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        t = torch.tensor(1.0, dtype=torch.float32, device=device)
        act_attn_weights = None
        step_count = 0
        total_steps = self.config.num_steps
        while t >= -dt / 2:
            expanded_t = t.expand(bsize)
            is_last_step = (step_count == total_steps - 1)
            collect_this_step = return_attention_weights and is_last_step
            v_t = self.denoise_step(
                state, prefix_pad_masks, past_key_values, x_t, expanded_t,
                return_attention_weights=collect_this_step,
            )
            if collect_this_step and isinstance(v_t, tuple):
                v_t, act_attn_weights = v_t
            x_t += dt * v_t
            t += dt
            step_count += 1

        return AffordanceOutput(
            actions=x_t,
            which2act_tokens=affordance_outputs["which2act"],
            wrist_tokens=affordance_outputs["wrist"],
            where2act_tokens=affordance_outputs["where2act"],
            how2act_tokens=affordance_outputs["how2act"],
            gen_attention_weights=gen_attn_weights,
            act_attention_weights=act_attn_weights,
        )

    def denoise_step(
        self, state, prefix_pad_masks, past_key_values, x_t, timestep,
        return_attention_weights=False,
    ):
        """Apply one Euler denoising step for action generation.

        Uses cached KV from prefix (understanding + affordance) to avoid
        recomputing those representations at each denoising step.

        Args:
            state: [B, max_state_dim] robot state.
            prefix_pad_masks: [B, N_prefix] padding masks from cached prefix.
            past_key_values: Cached KV states from prefix forward.
            x_t: [B, chunk_size, max_action_dim] current noisy actions.
            timestep: [B] current denoising timestep.
            return_attention_weights: If True, return (v_t, attn_weights).

        Returns:
            v_t or (v_t, attn_weights) depending on return_attention_weights.
        """
        act_embs, act_pad_masks, act_att_masks = self.embed_suffix(
            state, x_t, timestep
        )
        act_len = act_pad_masks.shape[1]

        # Build attention mask: action tokens attend to prefix (via cache)
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].repeat(
            1, act_len, 1
        )
        act_att_2d_masks = make_att_2d_masks(act_pad_masks, act_att_masks)
        full_att_2d_masks = torch.cat(
            [prefix_pad_2d_masks, act_att_2d_masks], dim=2
        )

        # Position IDs continue from prefix
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(act_pad_masks, dim=1) - 1

        fwd_result = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, None, act_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            return_attention_weights=return_attention_weights,
        )
        if return_attention_weights:
            (_, _, act_out), _, act_attn_weights = fwd_result
        else:
            (_, _, act_out), _ = fwd_result

        act_out = act_out[:, -self.config.chunk_size :]
        act_out = act_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(act_out)

        if return_attention_weights:
            return v_t, act_attn_weights
        return v_t
