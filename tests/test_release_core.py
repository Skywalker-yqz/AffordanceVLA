"""CPU checks for affordance decoding and the published training recipe."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from src.models.which2act_decoder import Which2ActFluxDecoder
from src.processors.training_schedule import resolve_training_schedule
from src.utils.model_utils import make_att_2d_masks


class FrozenVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)


def decoder_config(tokens):
    return SimpleNamespace(
        control_contextexpandedcropping=True, which2act_bbox_expand_ratio=1.5,
        which2act_loss_weight=0.1, which2act_num_tokens=tokens,
        gen_expert_config=SimpleNamespace(hidden_size=1024),
    )


class ReleaseCoreTests(unittest.TestCase):
    def test_flux_fast_and_full_shapes_and_gradients(self):
        for token_count in (16, 64):
            with self.subTest(token_count=token_count):
                decoder = Which2ActFluxDecoder(decoder_config(token_count), FrozenVAE())
                features = torch.randn(2, token_count, 1024, requires_grad=True)
                target = torch.randn(2, 16, 32, 32)
                self.assertEqual(decoder._tokens_to_latent(features).shape, target.shape)
                loss = decoder.compute_loss(features, target)['which2act_loss']
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertGreater(features.grad.abs().sum().item(), 0)

    def test_wrist_shares_flux_projection(self):
        from unittest.mock import patch
        import src.policies.affordance_policy as policy_module

        config = decoder_config(64)
        config.language_tokenizer_path = "unused"
        config.which2act_use_flux_vae = True
        config.which2act_flux_vae_path = "unused"
        config.control_wristtoken = True
        with patch.object(policy_module.AutoTokenizer, "from_pretrained", return_value=object()), \
             patch.object(policy_module, "AffordanceFlowMatching", return_value=nn.Identity()), \
             patch.object(policy_module, "load_flux_vae", return_value=FrozenVAE()), \
             patch.object(policy_module, "Where2ActDecoder", return_value=nn.Identity()), \
             patch.object(policy_module, "How2ActDecoder", return_value=nn.Identity()):
            policy = policy_module.AffordanceVLAPolicy(config)
        self.assertIs(policy.wrist_decoder.proj_head, policy.which2act_decoder.proj_head)
        self.assertIs(policy.wrist_decoder.flux_vae, policy.which2act_decoder.flux_vae)

    def test_flux_rejects_invalid_token_grid(self):
        for token_count in (0, 17, 36):
            with self.subTest(token_count=token_count), self.assertRaises(ValueError):
                Which2ActFluxDecoder(decoder_config(token_count), FrozenVAE())

    def test_uaa_information_flow(self):
        # Two tokens per expert, with one padded understanding token.
        padding = torch.tensor([[True, False, True, True, True, True]])
        blocks = torch.tensor([[0, 0, 1, 0, 1, 0]])
        mask = make_att_2d_masks(padding, blocks)
        self.assertFalse(mask[0, :2, 2:].any())
        self.assertTrue(mask[0, 2:4, 0].all())
        self.assertTrue(mask[0, 2:4, 2:4].all())
        self.assertFalse(mask[0, 2:4, 4:].any())
        self.assertTrue(mask[0, 4:, 2:4].all())
        self.assertFalse(mask[0, :, 1].any())

    def test_effective_batch_across_device_counts(self):
        for devices in (8, 16):
            for batch, steps in ((3072, 300000), (1024, 230000)):
                acc, effective, budget = resolve_training_schedule(
                    {'batch_size': 16, 'effective_batch_size': batch, 'max_steps': steps}, devices
                )
                self.assertEqual(acc * 16 * devices, batch)
                self.assertEqual((effective, budget), (batch, steps))

    def test_accumulated_training_matches_full_batch(self):
        import itertools
        import tempfile
        from unittest.mock import patch
        from omegaconf import OmegaConf
        import src.train as trainer

        class ToyPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))

            def forward(self, batch):
                return {"loss": ((self.weight * batch["x"] - 2 * batch["x"]) ** 2).mean()}

        trained, reference = ToyPolicy(), ToyPolicy()
        full_x = torch.arange(1, 5, dtype=torch.float32)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        for _ in range(2):
            reference_optimizer.zero_grad()
            reference({"x": full_x})["loss"].backward()
            reference_optimizer.step()
        microbatches = [{"x": full_x[:2]}, {"x": full_x[2:]}]
        with tempfile.TemporaryDirectory() as output_dir:
            cfg = OmegaConf.create({
                "model": {}, "data": {}, "training": {
                    "output_dir": output_dir, "stage": "stage3", "batch_size": 2,
                    "effective_batch_size": 4, "max_steps": 2, "num_workers": 0,
                    "seed": 0, "bf16": False, "lr_scheduler": "constant",
                    "max_grad_norm": 0, "log_every": 100, "save_every": 100,
                    "wandb_enabled": False,
                },
            })
            with patch.object(trainer, "AffordanceVLAConfig", return_value=SimpleNamespace(pretrained_path=None, load_ckpt=None)), \
                 patch.object(trainer, "AffordanceVLAPolicy", return_value=trained), \
                 patch.object(trainer, "apply_freeze_strategy"), \
                 patch.object(trainer, "setup_logging"), \
                 patch.object(trainer, "create_datasets", return_value={"toy": range(4)}), \
                 patch.object(trainer, "MultiDatasetLoader", return_value=itertools.cycle(microbatches)), \
                 patch.object(trainer, "create_optimizer", side_effect=lambda model, _: torch.optim.SGD(model.parameters(), lr=0.01)), \
                 patch.object(trainer, "save_checkpoint"), \
                 patch.object(torch.cuda, "is_available", return_value=False):
                trainer.train(cfg)
        torch.testing.assert_close(trained.weight, reference.weight)

    def test_epoch_budget_and_invalid_batch(self):
        schedule = resolve_training_schedule(
            {'batch_size': 16, 'effective_batch_size': 256, 'num_epochs': 20},
            8, {'train': 2560, 'validation': 1000}, {'train': 1, 'validation': 0},
        )
        self.assertEqual(schedule, (2, 256, 200))
        with self.assertRaises(ValueError):
            resolve_training_schedule({'batch_size': 15, 'effective_batch_size': 256, 'max_steps': 1}, 8)


if __name__ == '__main__':
    unittest.main()
