"""
Which2Act decoder — supervises Which2Act tokens via VQ-VAE or Flux VAE.

Two backends controlled by ``which2act_use_flux_vae``:

  A. VQ-VAE mode (default, backward-compatible):
     1. Crop observation image by bounding box → resize to 256×256
     2. Frozen VQ-VAE encode → multi-scale codebook indices → extract target scale
     3. Which2Act tokens → LayerNorm + Linear classification head → logits
     4. CrossEntropyLoss vs ground-truth codebook indices

  B. Flux VAE mode:
     1. Crop observation image by bounding box → resize to 256×256
     2. Frozen Flux VAE (AutoencoderKL) encode → continuous latent z_q
     3. Which2Act tokens → MLP proj_head → Spatial Unfold → z_pred
     4. MSE or Smooth-L1 loss (WHICH2ACT_FLUX_LOSS_TYPE) vs z_q in continuous latent space
     5. Optional: decode z_pred back to image via Flux VAE decoder

Pipeline (wrist full-image path — used by wrist decoder):
  1. Resize full wrist image to 256×256 (no bbox cropping)
  2–4. Same as above (for either backend).
"""

from __future__ import annotations

import math
import os
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vqvae import VQVAE

logger = logging.getLogger(__name__)

VQVAE_INPUT_SIZE = 256
FLUX_VAE_INPUT_SIZE = 256
FLUX_VAE_DOWNSAMPLE = 8
FLUX_VAE_LATENT_CHANNELS = 16

# Flux VAE latent regression loss: "mse" | "smooth_l1"
WHICH2ACT_FLUX_LOSS_TYPE = "mse"


# ═══════════════════════════════════════════════════════════════════════
# VQ-VAE Loading Utility
# ═══════════════════════════════════════════════════════════════════════


def load_vqvae(config) -> VQVAE:
    """Load a frozen VQ-VAE for Which2Act ground-truth encoding.

    Args:
        config: AffordanceVLAConfig with which2act_vae_* fields.

    Returns:
        Frozen VQVAE module.
    """
    target_scale = int(math.isqrt(config.which2act_num_tokens))
    assert target_scale * target_scale == config.which2act_num_tokens, (
        f"which2act_num_tokens ({config.which2act_num_tokens}) must be a perfect square"
    )
    v_patch_nums = tuple(range(1, target_scale + 1))

    vae = VQVAE(
        vocab_size=config.which2act_vae_vocab_size,
        z_channels=config.which2act_vae_z_channels,
        ch=config.which2act_vae_ch,
        test_mode=True,
        share_quant_resi=config.which2act_vae_share_quant_resi,
        v_patch_nums=v_patch_nums,
    )

    if config.which2act_vae_ckpt and os.path.exists(config.which2act_vae_ckpt):
        ckpt = torch.load(
            config.which2act_vae_ckpt, map_location="cpu", weights_only=False
        )
        vae.load_state_dict(ckpt, strict=True)
        logger.info(f"Loaded VQ-VAE checkpoint: {config.which2act_vae_ckpt}")
    else:
        logger.warning(
            "No VQ-VAE checkpoint loaded. which2act_vae_ckpt not set or not found."
        )

    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    return vae


# ═══════════════════════════════════════════════════════════════════════
# Flux VAE Loading Utility
# ═══════════════════════════════════════════════════════════════════════


def load_flux_vae(config):
    """Load a frozen Flux VAE (AutoencoderKL) for Which2Act latent regression.

    Args:
        config: AffordanceVLAConfig with which2act_flux_vae_path.

    Returns:
        Frozen AutoencoderKL module.
    """
    from diffusers import AutoencoderKL

    path = config.which2act_flux_vae_path
    flux_vae = AutoencoderKL.from_pretrained(path)
    flux_vae.requires_grad_(False)
    flux_vae.float()
    flux_vae.eval()
    logger.info(f"Loaded Flux VAE from: {path}")
    return flux_vae


# ═══════════════════════════════════════════════════════════════════════
# Crop Utilities
# ═══════════════════════════════════════════════════════════════════════


def crop_and_resize(
    image,
    bbox: torch.Tensor,
    target_size: int,
    expand_ratio: float = 0.0,
) -> torch.Tensor:
    """Crop each image in a batch by its bounding box and resize.

    Two cropping modes:
      - expand_ratio > 0 (context-expanded): expand bbox by ratio around
        center, enforce square, clamp to image bounds, resize to target_size.
      - expand_ratio <= 0 (tight crop): crop exactly at bbox, resize.

    Args:
        image: (B, 3, H, W) float tensor, OR list of (3, H_i, W_i) tensors.
        bbox: (B, 4) float tensor [x_min, y_min, x_max, y_max] in pixel coords.
        target_size: Output spatial size (square).
        expand_ratio: >0 enables context-expanded cropping.

    Returns:
        (B, 3, target_size, target_size) cropped and resized images.
    """
    is_list = isinstance(image, (list, tuple))
    B = len(image) if is_list else image.shape[0]

    crops = []
    for i in range(B):
        img_i = image[i] if is_list else image[i]
        _, H, W = img_i.shape

        x1, y1, x2, y2 = bbox[i].float().tolist()

        if expand_ratio > 0:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = (x2 - x1) * expand_ratio
            bh = (y2 - y1) * expand_ratio
            side = max(bw, bh)
            x1 = cx - side / 2.0
            y1 = cy - side / 2.0
            x2 = cx + side / 2.0
            y2 = cy + side / 2.0

        x1_i = max(0, min(int(x1), W - 1))
        y1_i = max(0, min(int(y1), H - 1))
        x2_i = max(x1_i + 1, min(int(x2 + 0.5), W))
        y2_i = max(y1_i + 1, min(int(y2 + 0.5), H))

        crop = img_i[:, y1_i:y2_i, x1_i:x2_i].unsqueeze(0)
        crop = F.interpolate(
            crop,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        crops.append(crop)
    return torch.cat(crops, dim=0)


def resize_full_image(
    image: torch.Tensor,
    target_size: int,
) -> torch.Tensor:
    """Resize full images to target_size without any cropping.

    Used for wrist camera path where no bbox is available.

    Args:
        image: (B, 3, H, W) float tensor.
        target_size: Output spatial size (square).

    Returns:
        (B, 3, target_size, target_size) resized images.
    """
    return F.interpolate(
        image,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )


# ═══════════════════════════════════════════════════════════════════════
# Which2Act Decoder (VQ-VAE backend — backward-compatible)
# ═══════════════════════════════════════════════════════════════════════


class Which2ActDecoder(nn.Module):
    """Decoder that supervises Which2Act tokens via VQ-VAE token prediction.

    Two operating modes controlled by ``is_wrist_mode``:
      - bbox mode (default): crops observation image by bbox, then VQ-VAE encode.
      - wrist mode: resizes full wrist image to 256×256, then VQ-VAE encode.

    Args:
        config: AffordanceVLAConfig.
        vqvae: Pre-loaded frozen VQVAE module.
        is_wrist_mode: If True, skip bbox cropping; resize full image directly.
    """

    def __init__(self, config, vqvae: VQVAE, is_wrist_mode: bool = False):
        super().__init__()

        self.vqvae = vqvae
        self.is_wrist_mode = is_wrist_mode

        self.use_expanded_crop = config.control_contextexpandedcropping
        self.expand_ratio = config.which2act_bbox_expand_ratio
        self.concat_wrist = config.which2act_concat_wrist
        self.target_image_size = config.which2act_target_image_size
        self.loss_weight = config.which2act_loss_weight
        self.label_smoothing = config.which2act_label_smoothing

        self.num_tokens = config.which2act_num_tokens
        self.target_scale = int(math.isqrt(self.num_tokens))

        _FULL_SCALES = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        if self.is_wrist_mode or self.use_expanded_crop:
            vae_input = VQVAE_INPUT_SIZE
        else:
            vae_input = self.target_image_size
        max_spatial = vae_input // 16
        self.encode_scales = tuple(s for s in _FULL_SCALES if s <= max_spatial)
        assert self.target_scale in self.encode_scales, (
            f"target_scale {self.target_scale} exceeds max spatial {max_spatial} "
            f"(vae_input={vae_input}). Increase target_image_size or use expanded crop."
        )
        self.target_scale_idx = list(self.encode_scales).index(self.target_scale)

        self.vocab_size = config.which2act_vae_vocab_size

        gen_hidden_dim = config.gen_expert_config.hidden_size
        self.cls_head = nn.Sequential(
            nn.LayerNorm(gen_hidden_dim),
            nn.Linear(gen_hidden_dim, self.vocab_size),
        )

        self.loss_fn = nn.CrossEntropyLoss(
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.vqvae.eval()
        return self

    @torch.no_grad()
    def encode_target(
        self,
        image: torch.Tensor,
        bbox: Optional[torch.Tensor] = None,
        wrist_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Construct ground-truth VQ-VAE codebook indices.

        Returns:
            (B, num_tokens) long tensor of ground-truth codebook indices.
        """
        if self.is_wrist_mode:
            prepared = resize_full_image(image, VQVAE_INPUT_SIZE)
        else:
            assert bbox is not None, "bbox is required for non-wrist Which2Act"
            if self.use_expanded_crop:
                prepared = crop_and_resize(
                    image, bbox, VQVAE_INPUT_SIZE, expand_ratio=self.expand_ratio
                )
            else:
                prepared = crop_and_resize(image, bbox, self.target_image_size)

            if self.concat_wrist and wrist_image is not None:
                wrist_resized = F.interpolate(
                    wrist_image,
                    size=(prepared.shape[2], prepared.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
                combined = torch.cat([prepared, wrist_resized], dim=3)
                prepared = F.interpolate(
                    combined,
                    size=(prepared.shape[2], prepared.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )

        if prepared.min() >= 0:
            prepared = prepared * 2.0 - 1.0

        idxBl_list = self.vqvae.img_to_idxBl(
            prepared, v_patch_nums=self.encode_scales
        )
        gt_indices = idxBl_list[self.target_scale_idx]
        return gt_indices

    def compute_loss(
        self,
        which2act_tokens: torch.Tensor,
        gt_indices: torch.Tensor,
    ) -> dict:
        """Compute cross-entropy loss and accuracy.

        Args:
            which2act_tokens: (B, num_tokens, D_gen) from generation expert.
            gt_indices: (B, num_tokens) long, from encode_target().

        Returns:
            dict with 'which2act_loss' and 'which2act_acc'.
        """
        logits = self.cls_head(which2act_tokens.float())

        loss = self.loss_fn(
            logits.reshape(-1, self.vocab_size),
            gt_indices.reshape(-1),
        ).mean()

        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == gt_indices).float().mean()

        return {
            "which2act_loss": loss * self.loss_weight,
            "which2act_acc": acc,
        }

    def forward(
        self,
        which2act_tokens: torch.Tensor,
        image: torch.Tensor,
        bbox: Optional[torch.Tensor] = None,
        wrist_image: Optional[torch.Tensor] = None,
    ) -> dict:
        """Full forward: encode target + compute loss.

        Returns:
            dict with 'which2act_loss' and 'which2act_acc'.
        """
        gt_indices = self.encode_target(image, bbox, wrist_image)
        return self.compute_loss(which2act_tokens, gt_indices)


# ═══════════════════════════════════════════════════════════════════════
# Which2Act Flux Decoder (Flux VAE backend — continuous latent regression)
# ═══════════════════════════════════════════════════════════════════════


class Which2ActFluxDecoder(nn.Module):
    """Decoder that supervises Which2Act tokens via Flux VAE latent regression.

    Instead of predicting discrete VQ-VAE codebook indices, this decoder
    predicts continuous Flux VAE latent features via an MLP + Spatial Unfold,
    supervised with MSE or Smooth-L1 loss (set by WHICH2ACT_FLUX_LOSS_TYPE).

    Spatial mapping:
      - 16 Which2Act tokens form a 4×4 grid.
      - Flux VAE encodes 256×256 → (B, 16, 32, 32) latent.
      - Each token (1024 dim) predicts one 8×8 patch across all 16 channels:
        1024 = 16 channels × 8 rows × 8 cols.
      - Spatial Unfold reassembles the 4×4 grid of 8×8 patches into (B, 16, 32, 32).

    Args:
        config: AffordanceVLAConfig.
        flux_vae: Pre-loaded frozen AutoencoderKL module.
        is_wrist_mode: If True, skip bbox cropping; resize full image directly.
    """

    def __init__(self, config, flux_vae, is_wrist_mode: bool = False):
        super().__init__()

        self.flux_vae = flux_vae
        self.is_wrist_mode = is_wrist_mode

        self.use_expanded_crop = config.control_contextexpandedcropping
        self.expand_ratio = config.which2act_bbox_expand_ratio
        self.loss_weight = config.which2act_loss_weight

        self.num_tokens = config.which2act_num_tokens
        self.grid_size = int(math.isqrt(self.num_tokens))
        assert self.grid_size * self.grid_size == self.num_tokens

        self.latent_spatial = FLUX_VAE_INPUT_SIZE // FLUX_VAE_DOWNSAMPLE
        self.patch_size = self.latent_spatial // self.grid_size

        self.scaling_factor = flux_vae.config.scaling_factor
        self.shift_factor = flux_vae.config.shift_factor

        gen_hidden_dim = config.gen_expert_config.hidden_size
        latent_patch_dim = FLUX_VAE_LATENT_CHANNELS * self.patch_size * self.patch_size
        assert gen_hidden_dim == latent_patch_dim, (
            f"gen_hidden_dim ({gen_hidden_dim}) must equal "
            f"latent_channels * patch_size^2 ({latent_patch_dim})"
        )

        self.proj_head = nn.Sequential(
            nn.LayerNorm(gen_hidden_dim),
            nn.Linear(gen_hidden_dim, gen_hidden_dim * 2),
            nn.GELU(),
            nn.Linear(gen_hidden_dim * 2, gen_hidden_dim),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.flux_vae.eval()
        return self

    def _prepare_image(
        self,
        image: torch.Tensor,
        bbox: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Crop/resize image to FLUX_VAE_INPUT_SIZE and normalize to [-1, 1]."""
        if self.is_wrist_mode:
            prepared = resize_full_image(image, FLUX_VAE_INPUT_SIZE)
        else:
            assert bbox is not None, "bbox is required for non-wrist Which2Act"
            if self.use_expanded_crop:
                prepared = crop_and_resize(
                    image, bbox, FLUX_VAE_INPUT_SIZE,
                    expand_ratio=self.expand_ratio,
                )
            else:
                prepared = crop_and_resize(image, bbox, FLUX_VAE_INPUT_SIZE)

        if prepared.min() >= 0:
            prepared = prepared * 2.0 - 1.0

        return prepared

    @torch.no_grad()
    def encode_target(
        self,
        image: torch.Tensor,
        bbox: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode image to Flux VAE continuous latent.

        Returns:
            (B, 16, 32, 32) float tensor — scaled VAE latent.
        """
        prepared = self._prepare_image(image, bbox)
        prepared = prepared.float()

        posterior = self.flux_vae.encode(prepared).latent_dist
        z_q = (posterior.sample() - self.shift_factor) * self.scaling_factor
        return z_q

    def _tokens_to_latent(self, which2act_tokens: torch.Tensor) -> torch.Tensor:
        """Map Which2Act tokens to predicted latent via MLP + Spatial Unfold.

        Args:
            which2act_tokens: (B, num_tokens, D_gen) from generation expert.

        Returns:
            (B, C, H, W) predicted latent matching Flux VAE shape.
        """
        z_pred = self.proj_head(which2act_tokens.float())
        B = z_pred.shape[0]
        g = self.grid_size
        p = self.patch_size
        c = FLUX_VAE_LATENT_CHANNELS

        z_pred = z_pred.view(B, g, g, c, p, p)
        z_pred = z_pred.permute(0, 3, 1, 4, 2, 5).contiguous()
        z_pred = z_pred.view(B, c, g * p, g * p)
        return z_pred

    def compute_loss(
        self,
        which2act_tokens: torch.Tensor,
        z_q: torch.Tensor,
    ) -> dict:
        """Compute latent regression loss and cosine similarity.

        Args:
            which2act_tokens: (B, num_tokens, D_gen) from generation expert.
            z_q: (B, C, H, W) ground-truth Flux VAE latent.

        Returns:
            dict with 'which2act_loss' and 'which2act_cos_sim'.
        """
        z_pred = self._tokens_to_latent(which2act_tokens)
        if WHICH2ACT_FLUX_LOSS_TYPE == "smooth_l1":
            loss = F.smooth_l1_loss(z_pred, z_q.float())
        else:
            loss = F.mse_loss(z_pred, z_q.float())

        with torch.no_grad():
            cos_sim = F.cosine_similarity(
                z_pred.flatten(1), z_q.float().flatten(1), dim=1
            ).mean()

        return {
            "which2act_loss": loss * self.loss_weight,
            "which2act_cos_sim": cos_sim,
        }

    def forward(
        self,
        which2act_tokens: torch.Tensor,
        image: torch.Tensor,
        bbox: Optional[torch.Tensor] = None,
        wrist_image: Optional[torch.Tensor] = None,
    ) -> dict:
        """Full forward: encode target + compute loss.

        Returns:
            dict with 'which2act_loss' and 'which2act_cos_sim'.
        """
        z_q = self.encode_target(image, bbox)
        return self.compute_loss(which2act_tokens, z_q)

    @torch.no_grad()
    def reconstruct(self, which2act_tokens: torch.Tensor) -> torch.Tensor:
        """Decode Which2Act tokens back to image via Flux VAE decoder.

        Args:
            which2act_tokens: (B, num_tokens, D_gen) from generation expert.

        Returns:
            (B, 3, 256, 256) reconstructed image in [0, 1].
        """
        z_pred = self._tokens_to_latent(which2act_tokens)
        z_decoded = (z_pred.float() / self.scaling_factor) + self.shift_factor
        img = self.flux_vae.decode(z_decoded).sample
        return (img / 2 + 0.5).clamp(0, 1)
