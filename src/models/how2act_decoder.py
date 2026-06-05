"""
How2Act decoder — supervises How2Act tokens for 3D shape and layout prediction.

How2Act tokens are split into two groups:
  1. Shape tokens → diffusion denoiser → reconstruct SAM-3D shape latent (4096, 8)
  2. Layout tokens → MLP → regress SAM-3D layout values (rotation, scale, translation)

Shape decoder (diffusion-based, ref: ReconVLA):
  - GT: shape_token (B, 4096, 8) where 4096 = 16³ voxel grid, 8 latent channels
    → reshape to (B, 8, 16, 16, 16) as 3D diffusion target
  - Condition: How2Act Shape tokens from generation expert
  - Architecture: ViT denoiser with 3D patchify + AdaLN conditioning + Gaussian diffusion
  - Loss: MSE (epsilon prediction)

Layout decoder (regression-based):
  - GT: layout values — rotation(4) + scale(3) + translation(3) = 10D
  - Architecture: How2Act Layout tokens → LayerNorm → MLP → predicted layout
  - Loss: SmoothL1
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Minimal Gaussian Diffusion (cosine schedule, epsilon prediction)
# ═══════════════════════════════════════════════════════════════════════


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
    """Cosine beta schedule (Nichol & Dhariwal, 2021)."""
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps, dtype=np.float64)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0, 0.999)


class GaussianDiffusion(nn.Module):
    """Minimal Gaussian diffusion for training and sampling.

    Supports epsilon prediction with cosine beta schedule.
    Works with arbitrary spatial dimensions (4D or 5D tensors).
    """

    def __init__(self, timesteps: int = 1000):
        super().__init__()
        betas = _cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        self.num_timesteps = timesteps

        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.tensor(np.sqrt(alphas_cumprod), dtype=torch.float32),
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.tensor(np.sqrt(1.0 - alphas_cumprod), dtype=torch.float32),
        )
        # For sampling
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_variance",
            torch.tensor(posterior_variance, dtype=torch.float32),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            torch.tensor(
                betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            torch.tensor(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
                dtype=torch.float32,
            ),
        )

    def _broadcast(self, values, x):
        """Broadcast (B,) schedule values to match x's spatial dimensions."""
        shape = (-1,) + (1,) * (x.dim() - 1)
        return values.view(shape)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: x_t = sqrt(a_bar) * x_0 + sqrt(1-a_bar) * eps."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ab = self._broadcast(self.sqrt_alphas_cumprod[t], x_start)
        sqrt_1mab = self._broadcast(self.sqrt_one_minus_alphas_cumprod[t], x_start)
        return sqrt_ab * x_start + sqrt_1mab * noise

    def training_losses(self, model, x_start, t, model_kwargs=None):
        """Compute epsilon-prediction MSE loss."""
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        pred_noise = model(x_t, t, **(model_kwargs or {}))
        loss = F.mse_loss(pred_noise, noise, reduction="none")
        return loss.mean(dim=list(range(1, loss.dim())))  # per-sample loss

    @torch.no_grad()
    def p_sample(self, model, x_t, t_idx, model_kwargs=None, temperature=1.0):
        """Single reverse step."""
        B = x_t.shape[0]
        t = torch.full((B,), t_idx, device=x_t.device, dtype=torch.long)
        pred_noise = model(x_t, t, **(model_kwargs or {}))

        # Predict x_0
        sqrt_ab = self._broadcast(self.sqrt_alphas_cumprod[t], x_t)
        sqrt_1mab = self._broadcast(self.sqrt_one_minus_alphas_cumprod[t], x_t)
        pred_x0 = (x_t - sqrt_1mab * pred_noise) / sqrt_ab

        # Posterior mean
        coef1 = self._broadcast(self.posterior_mean_coef1[t], x_t)
        coef2 = self._broadcast(self.posterior_mean_coef2[t], x_t)
        mean = coef1 * pred_x0 + coef2 * x_t

        if t_idx > 0:
            var = self._broadcast(self.posterior_variance[t], x_t)
            noise = torch.randn_like(x_t) * temperature
            return mean + torch.sqrt(var) * noise
        return mean

    @torch.no_grad()
    def p_sample_loop(self, model, shape, model_kwargs=None, temperature=1.0):
        """Full reverse sampling loop T → 0."""
        x = torch.randn(shape, device=self.betas.device)
        for t_idx in reversed(range(self.num_timesteps)):
            x = self.p_sample(model, x, t_idx, model_kwargs, temperature)
        return x


# ═══════════════════════════════════════════════════════════════════════
# ViT Denoiser with Adaptive Conditioning (ref: ReconVLA ViTAdaLN)
# ═══════════════════════════════════════════════════════════════════════


class _TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding → MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class _DenoiserBlock(nn.Module):
    """Transformer block with additive conditioning (simplified AdaLN)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            dim, heads, batch_first=True, dropout=0.0
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, c):
        """x: (B, T, D), c: (B, T, D) conditioning."""
        h = self.norm1(x + c)
        h2, _ = self.attn(h, h, h, need_weights=False)
        x = x + h2
        x = x + self.mlp(self.norm2(x))
        return x


class ShapeDenoiserNet(nn.Module):
    """ViT-based denoiser for shape latent diffusion (3D voxel grid).

    Input:  noisy shape latent (B, C, S, S, S) + timestep + condition
            where S=16, C=8, and S³=4096 is the voxel grid.
    Output: predicted noise (B, C, S, S, S)

    Condition: How2Act Shape tokens → mean-pool → project → broadcast to all patches.
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = 512,
        depth: int = 3,
        heads: int = 8,
        patch_size: int = 4,
        cond_dim: int = 1024,
        shape_spatial: int = 16,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.shape_spatial = shape_spatial
        self.num_patches_per_dim = shape_spatial // patch_size  # 4 for default

        num_patches = self.num_patches_per_dim ** 3
        patch_dim = in_channels * patch_size ** 3

        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, hidden_dim)
        self.pos_embed = nn.Parameter(
            torch.empty(1, num_patches, hidden_dim)
        )
        nn.init.normal_(self.pos_embed, std=0.02)

        # Timestep embedding
        self.t_embed = _TimestepEmbedder(hidden_dim)

        # Condition projection: pool How2Act tokens → global vector
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [_DenoiserBlock(hidden_dim, heads) for _ in range(depth)]
        )

        # Output projection
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, patch_dim)

    def patchify(self, x):
        """(B, C, D, H, W) → (B, T, patch_dim)  where T = (S/p)³."""
        p = self.patch_size
        x = rearrange(
            x, "b c (d p1) (h p2) (w p3) -> b (d h w) (c p1 p2 p3)",
            p1=p, p2=p, p3=p,
        )
        return x

    def unpatchify(self, x):
        """(B, T, patch_dim) → (B, C, D, H, W)."""
        p = self.patch_size
        n = self.num_patches_per_dim
        x = rearrange(
            x, "b (d h w) (c p1 p2 p3) -> b c (d p1) (h p2) (w p3)",
            d=n, h=n, w=n, p1=p, p2=p, p3=p, c=self.in_channels,
        )
        return x

    def forward(self, x, t, context):
        """
        Args:
            x: (B, C, D, H, W) noisy shape latent (e.g. B, 8, 16, 16, 16).
            t: (B,) diffusion timesteps.
            context: (B, N_shape, D_gen) How2Act Shape tokens.

        Returns:
            (B, C, D, H, W) predicted noise.
        """
        B = x.shape[0]

        # Patchify + position embed
        x = self.patch_embed(self.patchify(x)) + self.pos_embed  # (B, T, D)
        T = x.shape[1]

        # Timestep embedding
        t_emb = self.t_embed(t)  # (B, D)

        # Condition: mean-pool shape tokens → project → broadcast
        cond = self.cond_proj(context.mean(dim=1))  # (B, D)

        # Combined conditioning: timestep + global condition → broadcast
        c = (t_emb + cond).unsqueeze(1).expand(B, T, -1)  # (B, T, D)

        # Transformer blocks with conditioning
        for block in self.blocks:
            x = block(x, c)

        # Output
        x = self.out_proj(self.norm_out(x))  # (B, T, patch_dim)
        x = self.unpatchify(x)  # (B, C, D, H, W)
        return x


# ═══════════════════════════════════════════════════════════════════════
# Shape Diffusion Decoder
# ═══════════════════════════════════════════════════════════════════════


class How2ActShapeDecoder(nn.Module):
    """Diffusion-based decoder for SAM-3D shape latent reconstruction.

    GT shape_token (B, 4096, 8) where 4096 = 16³ voxel grid
    → reshape to (B, 8, 16, 16, 16) as 3D diffusion target.
    How2Act Shape tokens condition the denoiser via AdaLN.

    Args:
        config: AffordanceVLAConfig.
    """

    def __init__(self, config):
        super().__init__()
        gen_hidden_dim = config.gen_expert_config.hidden_size
        self.loss_weight = config.how2act_shape_loss_weight
        self.shape_channels = 8   # SAM-3D shape latent channels
        self.shape_spatial = 16   # cbrt(4096) = 16, i.e. 16³ voxel grid

        self.denoiser = ShapeDenoiserNet(
            in_channels=self.shape_channels,
            hidden_dim=config.how2act_shape_denoiser_dim,
            depth=config.how2act_shape_denoiser_depth,
            heads=config.how2act_shape_denoiser_heads,
            patch_size=config.how2act_shape_denoiser_patch_size,
            cond_dim=gen_hidden_dim,
            shape_spatial=self.shape_spatial,
        )
        self.diffusion = GaussianDiffusion(
            timesteps=config.how2act_shape_diffusion_timesteps
        )

    def compute_loss(
        self,
        shape_tokens: torch.Tensor,
        gt_shape: torch.Tensor,
    ) -> dict:
        """Compute diffusion training loss.

        Args:
            shape_tokens: (B, N_shape, D_gen) How2Act Shape tokens.
            gt_shape: (B, 4096, 8) GT SAM-3D shape latent (4096 = 16³ voxels).

        Returns:
            dict with 'how2act_shape_loss' (scalar).
        """
        B = gt_shape.shape[0]
        S = self.shape_spatial

        # Reshape GT to 3D voxel grid for diffusion
        x_start = rearrange(
            gt_shape, "b (d h w) c -> b c d h w",
            d=S, h=S, w=S,
        )  # (B, 8, 16, 16, 16)

        # Sample random timesteps
        t = torch.randint(
            0, self.diffusion.num_timesteps, (B,), device=gt_shape.device
        ).long()

        # Diffusion training loss (epsilon prediction)
        loss_per_sample = self.diffusion.training_losses(
            model=self.denoiser,
            x_start=x_start,
            t=t,
            model_kwargs={"context": shape_tokens},
        )
        loss = loss_per_sample.mean()

        return {"how2act_shape_loss": loss * self.loss_weight}

    @torch.no_grad()
    def sample(self, shape_tokens: torch.Tensor) -> torch.Tensor:
        """Sample shape latent from noise conditioned on shape tokens.

        Args:
            shape_tokens: (B, N_shape, D_gen) How2Act Shape tokens.

        Returns:
            (B, 4096, 8) predicted shape latent (4096 = 16³ voxels).
        """
        B = shape_tokens.shape[0]
        S = self.shape_spatial
        shape = (B, self.shape_channels, S, S, S)

        sampled = self.diffusion.p_sample_loop(
            model=self.denoiser,
            shape=shape,
            model_kwargs={"context": shape_tokens},
        )  # (B, 8, 16, 16, 16)

        return rearrange(sampled, "b c d h w -> b (d h w) c")  # (B, 4096, 8)


# ═══════════════════════════════════════════════════════════════════════
# Layout MLP Decoder
# ═══════════════════════════════════════════════════════════════════════


class How2ActLayoutDecoder(nn.Module):
    """Simple MLP decoder for SAM-3D layout regression.

    How2Act Layout tokens → pool → MLP → predicted layout values.
    GT layout: rotation(4) + scale(3) + translation(3) = 10D.
    Loss: SmoothL1.

    Args:
        config: AffordanceVLAConfig.
    """

    LAYOUT_DIM = 10  # rotation(4) + scale(3) + translation(3)

    def __init__(self, config):
        super().__init__()
        gen_hidden_dim = config.gen_expert_config.hidden_size
        self.loss_weight = config.how2act_layout_loss_weight

        self.mlp = nn.Sequential(
            nn.LayerNorm(gen_hidden_dim),
            nn.Linear(gen_hidden_dim, gen_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(gen_hidden_dim // 2, self.LAYOUT_DIM),
        )
        self.loss_fn = nn.SmoothL1Loss(reduction="mean")

    def compute_loss(
        self,
        layout_tokens: torch.Tensor,
        gt_layout: dict,
    ) -> dict:
        """Compute layout regression loss.

        Args:
            layout_tokens: (B, N_layout, D_gen) How2Act Layout tokens.
            gt_layout: dict with 'rotation' (B, 4), 'scale' (B, 3),
                       'translation' (B, 3).

        Returns:
            dict with 'how2act_layout_loss' (scalar).
        """
        # Mean-pool layout tokens → single vector
        pooled = layout_tokens.mean(dim=1)  # (B, D_gen)

        # Predict layout
        pred = self.mlp(pooled)  # (B, 10)

        # Build GT vector: [rotation(4), scale(3), translation(3)]
        gt = torch.cat(
            [gt_layout["rotation"], gt_layout["scale"], gt_layout["translation"]],
            dim=-1,
        )  # (B, 10)

        loss = self.loss_fn(pred, gt)

        return {"how2act_layout_loss": loss * self.loss_weight}

    def forward_predict(self, layout_tokens: torch.Tensor) -> dict:
        """Predict layout values (inference).

        Returns:
            dict with 'rotation' (B, 4), 'scale' (B, 3), 'translation' (B, 3).
        """
        pooled = layout_tokens.mean(dim=1)
        pred = self.mlp(pooled)  # (B, 10)
        return {
            "rotation": pred[:, :4],
            "scale": pred[:, 4:7],
            "translation": pred[:, 7:10],
        }


# ═══════════════════════════════════════════════════════════════════════
# Combined How2Act Decoder
# ═══════════════════════════════════════════════════════════════════════


class How2ActDecoder(nn.Module):
    """Combined decoder for How2Act tokens (shape + layout).

    Splits How2Act tokens into:
      - Shape tokens (first how2act_shape_num_tokens) → diffusion decoder
      - Layout tokens (remaining) → MLP regression

    Args:
        config: AffordanceVLAConfig.
    """

    def __init__(self, config):
        super().__init__()

        self.shape_num_tokens = config.how2act_shape_num_tokens
        self.layout_num_tokens = config.how2act_layout_num_tokens

        # Validate split
        expected_total = config.how2act_num_tokens
        actual_total = self.shape_num_tokens + self.layout_num_tokens
        assert actual_total == expected_total, (
            f"how2act_shape_num_tokens ({self.shape_num_tokens}) + "
            f"how2act_layout_num_tokens ({self.layout_num_tokens}) = {actual_total} "
            f"!= how2act_num_tokens ({expected_total})"
        )

        self.shape_decoder = How2ActShapeDecoder(config)
        self.layout_decoder = How2ActLayoutDecoder(config)

    def forward(
        self,
        how2act_tokens: torch.Tensor,
        gt_shape: Optional[torch.Tensor] = None,
        gt_layout: Optional[dict] = None,
    ) -> dict:
        """Compute shape diffusion loss + layout regression loss.

        Args:
            how2act_tokens: (B, N_how2act, D_gen) full How2Act tokens.
            gt_shape: (B, 4096, 8) optional GT shape latent.
            gt_layout: optional dict with rotation/scale/translation.

        Returns:
            dict with 'how2act_shape_loss', 'how2act_layout_loss'.
        """
        # Cast to float32: generation expert outputs bfloat16 but decoder heads are float32
        how2act_tokens = how2act_tokens.float()

        # Split tokens
        shape_tokens = how2act_tokens[:, : self.shape_num_tokens]
        layout_tokens = how2act_tokens[:, self.shape_num_tokens :]

        results = {}

        # Shape diffusion loss
        if gt_shape is not None:
            results.update(self.shape_decoder.compute_loss(shape_tokens, gt_shape))
        else:
            results["how2act_shape_loss"] = torch.tensor(
                0.0, device=how2act_tokens.device
            )

        # Layout regression loss
        if gt_layout is not None:
            results.update(self.layout_decoder.compute_loss(layout_tokens, gt_layout))
        else:
            results["how2act_layout_loss"] = torch.tensor(
                0.0, device=how2act_tokens.device
            )

        return results
