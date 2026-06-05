"""
Where2Act decoder — supervises Where2Act tokens via pixel-level heatmap prediction.

Pipeline:
  1. GT heatmap (H_orig, W_orig) → bilinear resize to (H_target, W_target) — online normalization
  2. Learnable spatial position embeddings (H_target × W_target, D_dec) as decoder queries
  3. Where2Act tokens (N_w2a, D_gen) → projection → serve as memory (KV) in Transformer Decoder
  4. n-layer Transformer Decoder: spatial queries cross-attend to Where2Act memory
  5. Output head → (H_target, W_target) logits → BCE loss with GT heatmap

Architecture (DETR-style):
  - Queries:  H*W learnable spatial position embeddings (one per pixel)
  - Memory:   Where2Act tokens from generation expert (compressed affordance info)
  - Decoder:  n layers of [self-attention → cross-attention → FFN]
  - Output:   Linear(D_dec → 1) per spatial position → sigmoid → heatmap
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Where2ActDecoder(nn.Module):
    """Decoder that predicts affordance heatmaps from Where2Act tokens.

    The Where2Act tokens (from the generation expert) encode compressed
    spatial affordance information. This decoder "unpacks" them into a
    full-resolution heatmap via cross-attention with spatial queries.

    Args:
        config: AffordanceVLAConfig with where2act_* parameters.
    """

    def __init__(self, config):
        super().__init__()

        self.heatmap_h = config.where2act_heatmap_h
        self.heatmap_w = config.where2act_heatmap_w
        self.num_spatial_tokens = self.heatmap_h * self.heatmap_w
        self.loss_weight = config.where2act_loss_weight

        decoder_dim = config.where2act_decoder_dim
        n_heads = config.where2act_decoder_heads
        n_layers = config.where2act_decoder_layers
        gen_hidden_dim = config.gen_expert_config.hidden_size

        # ── Spatial position embeddings (decoder queries) ──
        # One learnable embedding per spatial position in the target heatmap.
        # Resolution matches the GT heatmap after resize: H_target × W_target.
        self.spatial_pos_embed = nn.Parameter(
            torch.empty(1, self.num_spatial_tokens, decoder_dim)
        )
        nn.init.normal_(self.spatial_pos_embed, mean=0.0, std=0.02)

        # ── Project Where2Act tokens to decoder dimension (memory KV) ──
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(gen_hidden_dim),
            nn.Linear(gen_hidden_dim, decoder_dim),
        )

        # ── Lightweight Transformer Decoder ──
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=n_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stable training
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers
        )

        # ── Output head: per-pixel heatmap logit ──
        self.output_head = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def decode_heatmap(
        self,
        where2act_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Decode Where2Act tokens into a spatial heatmap.

        Args:
            where2act_tokens: (B, N_w2a, D_gen) from generation expert.

        Returns:
            (B, 1, H_target, W_target) logits (before sigmoid).
        """
        B = where2act_tokens.shape[0]

        # Cast to float32: generation expert outputs bfloat16 but decoder heads are float32
        where2act_tokens = where2act_tokens.float()

        # Project Where2Act tokens to decoder dim → memory
        memory = self.memory_proj(where2act_tokens)  # (B, N_w2a, D_dec)

        # Spatial position queries
        queries = self.spatial_pos_embed.expand(B, -1, -1)  # (B, H*W, D_dec)

        # Transformer Decoder: queries cross-attend to Where2Act memory
        decoded = self.transformer_decoder(
            tgt=queries, memory=memory
        )  # (B, H*W, D_dec)

        # Per-pixel prediction
        logits = self.output_head(decoded)  # (B, H*W, 1)
        logits = logits.squeeze(-1)  # (B, H*W)
        logits = logits.view(B, 1, self.heatmap_h, self.heatmap_w)

        return logits

    def compute_loss(
        self,
        where2act_tokens: torch.Tensor,
        gt_heatmap: torch.Tensor,
    ) -> dict:
        """Compute pixel-level BCE loss and metrics.

        Args:
            where2act_tokens: (B, N_w2a, D_gen) from generation expert.
            gt_heatmap: (B, 1, H_orig, W_orig) float [0, 1] GT affordance heatmap.

        Returns:
            dict with 'where2act_loss' (scalar) and 'where2act_pred' (B, 1, H, W).
        """
        # 1. Online resize GT heatmap to target resolution
        gt_resized = F.interpolate(
            gt_heatmap,
            size=(self.heatmap_h, self.heatmap_w),
            mode="bilinear",
            align_corners=False,
        )  # (B, 1, H_target, W_target)

        # 2. Decode heatmap from Where2Act tokens
        logits = self.decode_heatmap(where2act_tokens)  # (B, 1, H, W)

        # 3. Pixel-level BCE loss
        loss = self.loss_fn(logits, gt_resized).mean()

        # 4. Prediction (sigmoid) for visualization / metrics
        with torch.no_grad():
            pred = logits.sigmoid()

        return {
            "where2act_loss": loss * self.loss_weight,
            "where2act_pred": pred,
        }

    def forward(
        self,
        where2act_tokens: torch.Tensor,
        gt_heatmap: torch.Tensor,
    ) -> dict:
        """Full forward: decode + compute loss.

        Args:
            where2act_tokens: (B, N_w2a, D_gen) from generation expert.
            gt_heatmap: (B, 1, H_orig, W_orig) GT heatmap in [0, 1].

        Returns:
            dict with 'where2act_loss' and 'where2act_pred'.
        """
        return self.compute_loss(where2act_tokens, gt_heatmap)
