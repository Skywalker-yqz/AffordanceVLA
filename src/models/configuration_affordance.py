"""
AffordanceVLA model configuration.

Defines AffordanceVLAConfig with three expert sub-configs:
  - Understanding Expert (PaliGemma: SigLIP + Gemma)
  - Generation Expert (Gemma backbone for affordance queries)
  - Action Expert (Gemma for flow matching)

Key difference from F1Config: removes VQ-VAE / world model temporal configs,
adds affordance query parameters (which2act, where2act, how2act token counts).
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import CONFIG_MAPPING, AutoConfig

logger = logging.get_logger(__name__)


class AffordanceVLAConfig(PretrainedConfig):
    """Configuration for AffordanceVLA model.

    Attributes:
        und_expert_config: PaliGemma config for Understanding Expert.
        gen_expert_config: Gemma config for Generation Expert (affordance queries).
        act_expert_config: Gemma config for Action Expert (flow matching).
        proj_width: Projection dimension shared across action modules.
        chunk_size: Number of action steps predicted per chunk.
        max_action_dim: Maximum action dimension (padded).
        max_state_dim: Maximum state dimension (padded).
        num_steps: Number of Euler denoising steps for flow matching.
        tokenizer_max_length: Max length for language tokenizer.
        which2act_num_tokens: Number of Which2Act output tokens. Must be a perfect
            square whose sqrt is in VQ-VAE scale list (1,2,3,4,5,6,8,10,13,16).
            Valid values: 1, 4, 9, 16, 25, 36, 64, 100, 169, 256.
        where2act_num_tokens: Number of Where2Act output tokens.
        how2act_num_tokens: Number of How2Act output tokens.
        query_init_std: Standard deviation for query parameter initialization.
        which2act_use_flux_vae: When True, use Flux VAE (continuous latent
            regression) instead of VQ-VAE (discrete token classification).
        which2act_flux_vae_path: Path to pretrained Flux VAE checkpoint
            (AutoencoderKL). Required when which2act_use_flux_vae=True.
        control_contextexpandedcropping: When True, expand bbox by
            which2act_bbox_expand_ratio (square, center-aligned) before crop.
            which2act_target_image_size is ignored (fixed 256 for VQ-VAE/Flux VAE).
            When False, tight bbox crop + resize to which2act_target_image_size.
        which2act_bbox_expand_ratio: Expand ratio for context-expanded cropping.
        which2act_label_smoothing: Label smoothing for Which2Act CrossEntropy
            (only used in VQ-VAE mode).
        control_wristtoken: When True, add an independent wrist token group
            (same count/decoder/loss_weight as Which2Act) supervised by the
            full wrist image. Affects model architecture (total query tokens).
    """

    model_type = "affordance_vla"
    sub_configs = {
        "und_expert_config": AutoConfig,
        "gen_expert_config": AutoConfig,
        "act_expert_config": AutoConfig,
    }

    def __init__(
        self,
        # Expert sub-configs
        und_expert_config=None,
        gen_expert_config=None,
        act_expert_config=None,
        # Projection
        proj_width=1024,
        # Action
        chunk_size=50,
        max_action_dim=32,
        max_state_dim=32,
        num_steps=10,
        # Language
        tokenizer_max_length=48,
        language_tokenizer_path=None,
        # Affordance query configuration
        which2act_num_tokens=16,  # valid: 1,4,9,16,25,36,64,100,169,256
        where2act_num_tokens=16,
        how2act_num_tokens=16,
        query_init_std=0.02,
        # Which2Act backend switch
        which2act_use_flux_vae=False,
        which2act_flux_vae_path=None,
        # Which2Act decoder (VQ-VAE discrete token prediction — default)
        which2act_vae_ckpt=None,
        which2act_vae_vocab_size=4096,
        which2act_vae_z_channels=32,
        which2act_vae_ch=160,
        which2act_vae_share_quant_resi=4,
        which2act_target_image_size=256,
        which2act_concat_wrist=False,
        which2act_loss_weight=1.0,
        # Context-Expanded Cropping
        control_contextexpandedcropping=True,
        which2act_bbox_expand_ratio=1.5,
        # Label Smoothing (VQ-VAE mode only)
        which2act_label_smoothing=0.1,
        # Wrist Token — independent token group with its own Which2Act decoder
        control_wristtoken=True,
        # Where2Act decoder (Transformer Decoder → pixel-level heatmap)
        where2act_heatmap_h=24,
        where2act_heatmap_w=24,
        where2act_decoder_layers=4,
        where2act_decoder_heads=8,
        where2act_decoder_dim=256,
        where2act_loss_weight=1.0,
        # How2Act decoder (Shape diffusion + Layout MLP)
        how2act_shape_num_tokens=12,
        how2act_layout_num_tokens=4,
        # Shape denoiser
        how2act_shape_denoiser_dim=512,
        how2act_shape_denoiser_depth=3,
        how2act_shape_denoiser_heads=8,
        how2act_shape_denoiser_patch_size=4,
        how2act_shape_diffusion_timesteps=1000,
        how2act_shape_loss_weight=1.0,
        # Layout MLP
        how2act_layout_loss_weight=1.0,
        # Pretrained weights
        pretrained_path=None,
        load_ckpt=None,
        # Data loading — dataset-level resize for variable-resolution datasets
        dataset_target_h=480,
        dataset_target_w=640,
        # General
        use_cache=True,
        attention_implementation="eager",
        resize_imgs_with_padding="(224, 224)",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.is_encoder_decoder = False

        # Pretrained weights loading
        self.pretrained_path = pretrained_path
        self.load_ckpt = load_ckpt

        # Projection and action
        self.proj_width = proj_width
        self.chunk_size = chunk_size
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.num_steps = num_steps

        # Language
        self.tokenizer_max_length = tokenizer_max_length
        self.language_tokenizer_path = language_tokenizer_path

        # Affordance query configuration
        self.which2act_num_tokens = which2act_num_tokens
        self.where2act_num_tokens = where2act_num_tokens
        self.how2act_num_tokens = how2act_num_tokens
        self.query_init_std = query_init_std

        # Which2Act backend switch
        self.which2act_use_flux_vae = which2act_use_flux_vae
        self.which2act_flux_vae_path = which2act_flux_vae_path

        # Which2Act decoder — frozen VQ-VAE for ground-truth encoding (default)
        self.which2act_vae_ckpt = which2act_vae_ckpt
        self.which2act_vae_vocab_size = which2act_vae_vocab_size
        self.which2act_vae_z_channels = which2act_vae_z_channels
        self.which2act_vae_ch = which2act_vae_ch
        self.which2act_vae_share_quant_resi = which2act_vae_share_quant_resi
        self.which2act_target_image_size = which2act_target_image_size
        self.which2act_concat_wrist = which2act_concat_wrist
        self.which2act_loss_weight = which2act_loss_weight
        self.control_contextexpandedcropping = control_contextexpandedcropping
        self.which2act_bbox_expand_ratio = which2act_bbox_expand_ratio
        self.which2act_label_smoothing = which2act_label_smoothing
        self.control_wristtoken = control_wristtoken

        # Where2Act decoder
        self.where2act_heatmap_h = where2act_heatmap_h
        self.where2act_heatmap_w = where2act_heatmap_w
        self.where2act_decoder_layers = where2act_decoder_layers
        self.where2act_decoder_heads = where2act_decoder_heads
        self.where2act_decoder_dim = where2act_decoder_dim
        self.where2act_loss_weight = where2act_loss_weight

        # How2Act decoder
        self.how2act_shape_num_tokens = how2act_shape_num_tokens
        self.how2act_layout_num_tokens = how2act_layout_num_tokens
        self.how2act_shape_denoiser_dim = how2act_shape_denoiser_dim
        self.how2act_shape_denoiser_depth = how2act_shape_denoiser_depth
        self.how2act_shape_denoiser_heads = how2act_shape_denoiser_heads
        self.how2act_shape_denoiser_patch_size = how2act_shape_denoiser_patch_size
        self.how2act_shape_diffusion_timesteps = how2act_shape_diffusion_timesteps
        self.how2act_shape_loss_weight = how2act_shape_loss_weight
        self.how2act_layout_loss_weight = how2act_layout_loss_weight

        # Data loading
        self.dataset_target_h = dataset_target_h
        self.dataset_target_w = dataset_target_w

        # General
        self.use_cache = use_cache
        self.attention_implementation = attention_implementation
        self.resize_imgs_with_padding = resize_imgs_with_padding

        # ── Understanding Expert (PaliGemma: SigLIP vision + Gemma LM) ──
        self.und_expert_config = und_expert_config
        if isinstance(self.und_expert_config, dict):
            und_expert_config["model_type"] = und_expert_config.get(
                "model_type", "paligemma"
            )
            self.und_expert_config = CONFIG_MAPPING[
                und_expert_config["model_type"]
            ](**und_expert_config)
        elif und_expert_config is None:
            self.und_expert_config = CONFIG_MAPPING["paligemma"](
                transformers_version="4.48.1",
                _vocab_size=257152,
                bos_token_id=2,
                eos_token_id=1,
                hidden_size=2048,
                image_token_index=257152,
                model_type="paligemma",
                pad_token_id=0,
                projection_dim=2048,
                text_config={
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 2048,
                    "intermediate_size": 16384,
                    "model_type": "gemma",
                    "num_attention_heads": 8,
                    "num_hidden_layers": 18,
                    "num_image_tokens": 256,
                    "num_key_value_heads": 1,
                    "torch_dtype": "float32",
                    "vocab_size": 257152,
                },
                vision_config={
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "model_type": "siglip_vision_model",
                    "num_attention_heads": 16,
                    "num_hidden_layers": 27,
                    "num_image_tokens": 256,
                    "patch_size": 14,
                    "projection_dim": 2048,
                    "projector_hidden_act": "gelu_fast",
                    "torch_dtype": "float32",
                    "vision_use_head": False,
                },
            )

        # ── Generation Expert (Gemma backbone for affordance queries) ──
        self.gen_expert_config = gen_expert_config
        if isinstance(self.gen_expert_config, dict):
            gen_expert_config["model_type"] = gen_expert_config.get(
                "model_type", "gemma"
            )
            self.gen_expert_config = CONFIG_MAPPING[
                gen_expert_config["model_type"]
            ](**gen_expert_config)
        elif gen_expert_config is None:
            self.gen_expert_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=4096,
                max_position_embeddings=8192,
                model_type="gemma",
                num_attention_heads=8,
                num_hidden_layers=18,
                num_key_value_heads=1,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype="float32",
                transformers_version="4.48.1",
                use_cache=True,
                vocab_size=257152,
            )

        # ── Action Expert (Gemma for flow matching) ──
        self.act_expert_config = act_expert_config
        if isinstance(self.act_expert_config, dict):
            act_expert_config["model_type"] = act_expert_config.get(
                "model_type", "gemma"
            )
            self.act_expert_config = CONFIG_MAPPING[
                act_expert_config["model_type"]
            ](**act_expert_config)
        elif act_expert_config is None:
            self.act_expert_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=4096,
                max_position_embeddings=8192,
                model_type="gemma",
                num_attention_heads=8,
                num_hidden_layers=18,
                num_key_value_heads=1,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype="float32",
                transformers_version="4.48.1",
                use_cache=True,
                vocab_size=257152,
            )

    @property
    def total_affordance_tokens(self) -> int:
        """Total number of affordance query tokens across all groups.

        Includes the optional wrist token group (sharing the Which2Act count)
        when ``control_wristtoken`` is enabled, so the value matches
        ``AffordanceQueryModule.total_num_tokens`` exactly.
        """
        wrist_num_tokens = (
            self.which2act_num_tokens if self.control_wristtoken else 0
        )
        return (
            self.which2act_num_tokens
            + wrist_num_tokens
            + self.where2act_num_tokens
            + self.how2act_num_tokens
        )

    def to_dict(self):
        output = super().to_dict()
        output.pop("_ignore_index", None)
        return output

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        if pretrained_model_name_or_path.endswith(".json"):
            config = super().from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        else:
            config = super().from_pretrained(
                f"{pretrained_model_name_or_path}/config.json", **kwargs
            )
        return config
