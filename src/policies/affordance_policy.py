"""
AffordanceVLA policy wrapper.

High-level interface that wraps AffordanceFlowMatching, handling:
  - Data preparation (images, language, state, actions)
  - Training forward with loss computation
  - Inference with action caching
  - Model loading/saving (safetensors + HuggingFace Hub)

Supports two Which2Act backends controlled by config.which2act_use_flux_vae:
  - VQ-VAE mode: CrossEntropy on discrete codebook indices
  - Flux VAE mode: MSE or Smooth-L1 on continuous Flux VAE latent
"""

import ast
import os
import logging
from pathlib import Path
from collections import deque

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE

import packaging
import safetensors
from safetensors.torch import load_model as load_model_as_safetensor
from safetensors.torch import save_model as save_model_as_safetensor

import torch
import torch.nn as nn
from torch import Tensor

from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from src.models.modeling_affordance import AffordanceFlowMatching, AffordanceOutput
from src.models.configuration_affordance import AffordanceVLAConfig
from src.models.which2act_decoder import (
    Which2ActDecoder, Which2ActFluxDecoder, load_vqvae, load_flux_vae,
)
from src.models.where2act_decoder import Where2ActDecoder
from src.models.how2act_decoder import How2ActDecoder
from src.utils.image_tools import resize_with_pad
from src.utils.model_utils import pad_vector

logger = hf_logging.get_logger(__name__)

# Constants (inlined from lerobot to avoid hard dependency)
OBS_STATE = "observation.state"
ACTION = "action"


class AffordanceVLAPolicy(nn.Module):
    """High-level policy for AffordanceVLA.

    Wraps AffordanceFlowMatching with data preparation, loss computation,
    action caching for inference, and model serialization.

    Includes Which2Act decoder for supervising Which2Act tokens via either
    VQ-VAE discrete token prediction or Flux VAE continuous latent regression,
    and optionally a Wrist decoder (same architecture) for the wrist token group.

    Attributes:
        config: AffordanceVLAConfig instance.
        language_tokenizer: PaliGemma tokenizer for task text.
        model: AffordanceFlowMatching core model.
        which2act_decoder: Which2Act token supervisor.
        wrist_decoder: Wrist token supervisor (when control_wristtoken=True).
    """

    config_class = AffordanceVLAConfig
    cache_action_steps = 5

    def __init__(self, config: AffordanceVLAConfig, **kwargs):
        super().__init__()
        self.config = config

        # Language tokenizer (PaliGemma)
        self.language_tokenizer = AutoTokenizer.from_pretrained(
            config.language_tokenizer_path
        )

        # Core model (MoT with three experts)
        self.model = AffordanceFlowMatching(config, **kwargs)

        # Which2Act decoder: choose backend based on config
        if config.which2act_use_flux_vae:
            if config.which2act_flux_vae_path is not None:
                flux_vae = load_flux_vae(config)
                self.which2act_decoder = Which2ActFluxDecoder(
                    config, flux_vae, is_wrist_mode=False
                )
                if config.control_wristtoken:
                    self.wrist_decoder = Which2ActFluxDecoder(
                        config, flux_vae, is_wrist_mode=True
                    )
                else:
                    self.wrist_decoder = None
            else:
                logger.warning(
                    "which2act_use_flux_vae=True but which2act_flux_vae_path not set. "
                    "Which2Act decoder disabled."
                )
                self.which2act_decoder = None
                self.wrist_decoder = None
        else:
            if config.which2act_vae_ckpt is not None:
                vqvae = load_vqvae(config)
                self.which2act_decoder = Which2ActDecoder(
                    config, vqvae, is_wrist_mode=False
                )
                if config.control_wristtoken:
                    self.wrist_decoder = Which2ActDecoder(
                        config, vqvae, is_wrist_mode=True
                    )
                else:
                    self.wrist_decoder = None
            else:
                vqvae = None
                self.which2act_decoder = None
                if config.control_wristtoken:
                    self.wrist_decoder = None
                else:
                    self.wrist_decoder = None

        # Where2Act decoder: Transformer Decoder → pixel-level heatmap
        self.where2act_decoder = Where2ActDecoder(config)

        # How2Act decoder: Shape diffusion + Layout MLP
        self.how2act_decoder = How2ActDecoder(config)

        self.reset()

    def reset(self):
        """Reset action queue. Should be called on every environment reset."""
        self._action_queue = deque([], maxlen=self.cache_action_steps)

    def get_optim_params(self):
        """Return all trainable parameters."""
        return self.parameters()

    # ─── Inference ───────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
    ) -> Tensor:
        """Generate actions with caching."""
        self.eval()

        if len(self._action_queue) == 0:
            images, image_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            lang_tokens, lang_masks = self.prepare_language(batch)

            output = self.model.sample_actions(
                images=images,
                img_masks=image_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                state=state,
                noise=noise,
            )

            original_action_dim = 7
            actions = output.actions[:, :, :original_action_dim]
            for step_idx in range(self.cache_action_steps):
                self._action_queue.append(actions[:, step_idx])

        return self._action_queue.popleft()

    @torch.no_grad()
    def inference_with_attention(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
    ):
        """Run inference and return full output including attention weights."""
        self.eval()
        images, image_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)

        return self.model.sample_actions(
            images=images,
            img_masks=image_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            noise=noise,
            return_attention_weights=True,
        )

    # ─── Training Forward ────────────────────────────────────────────

    def forward(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Training forward pass.

        Prepares data from batch, runs the model, and computes losses:
          - action_loss: flow matching MSE on action expert output
          - which2act_loss: VQ-VAE CE or Flux VAE MSE/Smooth-L1 on Which2Act tokens
          - where2act / how2act losses

        Returns:
            Loss dictionary with loss, action_loss, which2act_loss, etc.
        """
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        action_is_pad = batch.get("action_is_pad")

        output = self.model(
            images, img_masks, lang_tokens, lang_masks,
            state, actions, noise, time,
        )

        # ── Action Loss ──
        loss_dict = {}
        action_losses = output.action_loss

        has_real_actions = batch.get("action") is not None and batch.get(
            "action_mask", torch.tensor(False)
        ).any()

        if has_real_actions:
            if action_is_pad is not None:
                in_episode_bound = ~action_is_pad
                if action_losses.shape == in_episode_bound.shape:
                    action_losses = action_losses * in_episode_bound
                else:
                    action_losses = action_losses * in_episode_bound.unsqueeze(-1)

            action_losses = action_losses[:, :, : self.config.max_action_dim]

            loss_dict["action_loss"] = action_losses.mean()
        else:
            loss_dict["action_loss"] = torch.tensor(0.0, device=images[0].device)

        # ── Which2Act Loss ──
        if self.which2act_decoder is not None and "bbox" in batch:
            crop_image, crop_bbox = self._get_raw_image_for_crop(batch)
            wrist_image = batch.get("wrist_image")

            w2a_look_results = self.which2act_decoder(
                which2act_tokens=output.which2act_tokens,
                image=crop_image,
                bbox=crop_bbox,
                wrist_image=wrist_image,
            )
            loss_dict["which2act_loss"] = w2a_look_results["which2act_loss"]
            # VQ-VAE mode returns 'which2act_acc', Flux VAE returns 'which2act_cos_sim'
            if "which2act_acc" in w2a_look_results:
                loss_dict["which2act_acc"] = w2a_look_results["which2act_acc"]
            if "which2act_cos_sim" in w2a_look_results:
                loss_dict["which2act_cos_sim"] = w2a_look_results["which2act_cos_sim"]
        else:
            loss_dict["which2act_loss"] = torch.tensor(
                0.0, device=state.device
            )
            loss_dict["which2act_acc"] = torch.tensor(
                0.0, device=state.device
            )

        # ── Where2Act Loss ──
        if "heatmap" in batch:
            gt_heatmap = batch["heatmap"]
            w2a_results = self.where2act_decoder(
                where2act_tokens=output.where2act_tokens,
                gt_heatmap=gt_heatmap,
            )
            loss_dict["where2act_loss"] = w2a_results["where2act_loss"]
            loss_dict["where2act_pred"] = w2a_results["where2act_pred"]
        else:
            loss_dict["where2act_loss"] = torch.tensor(
                0.0, device=state.device
            )

        # ── How2Act Loss ──
        gt_shape = batch.get("shape_token")
        gt_layout = batch.get("layout_token")
        h2a_results = self.how2act_decoder(
            how2act_tokens=output.how2act_tokens,
            gt_shape=gt_shape,
            gt_layout=gt_layout,
        )
        loss_dict["how2act_shape_loss"] = h2a_results["how2act_shape_loss"]
        loss_dict["how2act_layout_loss"] = h2a_results["how2act_layout_loss"]

        # ── Wrist Loss ──
        if (
            self.wrist_decoder is not None
            and output.wrist_tokens is not None
            and batch.get("wrist_image") is not None
        ):
            wrist_results = self.wrist_decoder(
                which2act_tokens=output.wrist_tokens,
                image=batch["wrist_image"],
                bbox=None,
            )
            loss_dict["wrist_loss"] = wrist_results["which2act_loss"]
            if "which2act_acc" in wrist_results:
                loss_dict["wrist_acc"] = wrist_results["which2act_acc"]
            if "which2act_cos_sim" in wrist_results:
                loss_dict["wrist_cos_sim"] = wrist_results["which2act_cos_sim"]
        else:
            loss_dict["wrist_loss"] = torch.tensor(0.0, device=state.device)
            loss_dict["wrist_acc"] = torch.tensor(0.0, device=state.device)

        # ── Total Loss ──
        loss_dict["loss"] = (
            loss_dict["action_loss"]
            + loss_dict["which2act_loss"]
            + loss_dict["wrist_loss"]
            + loss_dict["where2act_loss"]
            + loss_dict["how2act_shape_loss"]
            + loss_dict["how2act_layout_loss"]
        )

        return loss_dict

    def _get_raw_image_for_crop(self, batch) -> tuple:
        """Get the original-resolution image and bbox for Which2Act cropping.

        When DataLoader resize is applied (AGD20K, RefSpatial), the batch
        contains raw_image (list of variable-size tensors at original
        resolution) and raw_bbox (original pixel coords). This ensures
        Which2Act crops from the full-resolution image for higher-quality
        encoding, avoiding double-resize artifacts.
        """
        raw_image = batch.get("raw_image")
        if raw_image is not None:
            return raw_image, batch["raw_bbox"]
        return batch["image"], batch["bbox"]

    # ─── Data Preparation ────────────────────────────────────────────

    def prepare_images(self, batch):
        """Prepare observation images for Understanding Expert."""
        images = []
        image_masks = []

        resize_dims = self.config.resize_imgs_with_padding
        if isinstance(resize_dims, str):
            resize_dims = ast.literal_eval(resize_dims)

        img = batch["image"]
        if resize_dims is not None:
            img = resize_with_pad(img, *resize_dims, pad_value=0)
        img = img * 2.0 - 1.0
        B = img.shape[0]
        images.append(img)
        image_masks.append(torch.ones(B, dtype=torch.bool, device=img.device))

        wrist = batch.get("wrist_image")
        if wrist is not None:
            wrist_mask = batch.get("wrist_image_mask")
            if wrist_mask is not None and wrist_mask.any():
                if resize_dims is not None:
                    wrist = resize_with_pad(wrist, *resize_dims, pad_value=0)
                wrist = wrist * 2.0 - 1.0
                images.append(wrist)
                image_masks.append(wrist_mask)

        return images, image_masks

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize task text using PaliGemma tokenizer."""
        device = batch["image"].device
        tasks = batch["instruction"]

        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        self.language_tokenizer.padding_side = "right"
        tokenized = self.language_tokenizer(
            tasks,
            padding="max_length",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
            truncation=True,
        )
        lang_tokens = tokenized["input_ids"].to(device=device)
        lang_masks = tokenized["attention_mask"].to(
            device=device, dtype=torch.bool
        )
        return lang_tokens, lang_masks

    def prepare_state(self, batch) -> Tensor:
        """Pad state to max_state_dim. Returns zeros if state absent."""
        state = batch.get("state")
        if state is None:
            B = batch["image"].shape[0]
            device = batch["image"].device
            return torch.zeros(B, self.config.max_state_dim, device=device)
        return pad_vector(state, self.config.max_state_dim)

    def prepare_action(self, batch) -> Tensor:
        """Pad action's last dim to max_action_dim. Returns zeros if absent."""
        action = batch.get("action")
        if action is None:
            B = batch["image"].shape[0]
            device = batch["image"].device
            return torch.zeros(
                B, self.config.chunk_size, self.config.max_action_dim,
                device=device,
            )
        return pad_vector(action, self.config.max_action_dim)

    # ─── Model Loading / Saving ──────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: AffordanceVLAConfig | None = None,
        strict: bool = False,
        **kwargs,
    ):
        """Load a pretrained AffordanceVLA policy."""
        if config is None:
            config = AffordanceVLAConfig.from_pretrained(
                pretrained_name_or_path,
                **kwargs,
            )
        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)

        if model_id.endswith(".json"):
            model_id = "/".join(model_id.split("/")[:-1])

        if os.path.isdir(model_id):
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            policy = cls._load_as_safetensor(instance, model_file, "cpu", strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                )
                policy = cls._load_as_safetensor(
                    instance, model_file, "cpu", strict
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found in {model_id}"
                ) from e

        policy.eval()
        return policy

    @classmethod
    def _load_as_safetensor(cls, model, model_file, map_location, strict):
        """Load model weights from safetensors file."""
        if packaging.version.parse(
            safetensors.__version__
        ) < packaging.version.parse("0.4.3"):
            load_model_as_safetensor(model, model_file, strict=strict)
            if map_location != "cpu":
                model.to(map_location)
        else:
            safetensors.torch.load_model(
                model, model_file, strict=strict, device=map_location
            )
        return model

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save model config and weights to directory."""
        save_directory = Path(save_directory)
        self.config.save_pretrained(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(
            model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE)
        )
