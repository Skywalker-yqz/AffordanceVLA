"""
Multi-scale Vector Quantizer for VQ-VAE (VAR framework).

Implements residual multi-scale quantization where each scale quantizes the
residual from all previous scales, from coarse (1×1) to fine (N×N).

References:
    https://github.com/CompVis/taming-transformers/blob/main/taming/models/vqvae.py
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["VectorQuantizer2"]


class VectorQuantizer2(nn.Module):
    def __init__(
        self,
        vocab_size,
        Cvae,
        using_znorm,
        beta: float = 0.25,
        default_qresi_counts=0,
        v_patch_nums=None,
        quant_resi=0.5,
        share_quant_resi=4,
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        self.v_patch_nums: Tuple[int] = v_patch_nums

        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:
            self.quant_resi = PhiNonShared(
                [
                    (Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
                    for _ in range(default_qresi_counts or len(self.v_patch_nums))
                ]
            )
        elif share_quant_resi == 1:
            self.quant_resi = PhiShared(
                Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()
            )
        else:
            self.quant_resi = PhiPartiallyShared(
                nn.ModuleList(
                    [
                        (Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
                        for _ in range(share_quant_resi)
                    ]
                )
            )

        self.register_buffer(
            "ema_vocab_hit_SV",
            torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0),
        )
        self.record_hit = 0

        self.beta: float = beta
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)

        self.prog_si = -1

    def eini(self, eini):
        if eini > 0:
            nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0:
            self.embedding.weight.data.uniform_(
                -abs(eini) / self.vocab_size, abs(eini) / self.vocab_size
            )

    def extra_repr(self) -> str:
        return (
            f"{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  "
            f"|  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}"
        )

    def embed_to_fhat(
        self,
        ms_h_BChw: List[torch.Tensor],
        all_to_max_scale=True,
        last_one=False,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums):
                h_BChw = ms_h_BChw[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode="bicubic")
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
                f_hat.add_(h_BChw)
                if last_one:
                    ls_f_hat_BChw = f_hat
                else:
                    ls_f_hat_BChw.append(f_hat.clone())
        else:
            f_hat = ms_h_BChw[0].new_zeros(
                B, self.Cvae, self.v_patch_nums[0], self.v_patch_nums[0], dtype=torch.float32
            )
            for si, pn in enumerate(self.v_patch_nums):
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode="bicubic")
                h_BChw = self.quant_resi[si / (SN - 1)](ms_h_BChw[si])
                f_hat.add_(h_BChw)
                if last_one:
                    ls_f_hat_BChw = f_hat
                else:
                    ls_f_hat_BChw.append(f_hat)
        return ls_f_hat_BChw

    def f_to_idxBl_or_fhat(
        self,
        f_BChw: torch.Tensor,
        to_fhat: bool,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
    ) -> List[Union[torch.Tensor, torch.LongTensor]]:
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        f_hat_or_idx_Bl: List[torch.Tensor] = []

        patch_hws = [
            (pn, pn) if isinstance(pn, int) else (pn[0], pn[1])
            for pn in (v_patch_nums or self.v_patch_nums)
        ]

        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws):
            if 0 <= self.prog_si < si:
                break
            z_NC = (
                F.interpolate(f_rest, size=(ph, pw), mode="area")
                .permute(0, 2, 3, 1)
                .reshape(-1, C)
                if (si != SN - 1)
                else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            )
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(
                    z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1
                )
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(
                    self.embedding.weight.data.square(), dim=1, keepdim=False
                )
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                idx_N = torch.argmin(d_no_grad, dim=1)

            idx_Bhw = idx_N.view(B, ph, pw)
            h_BChw = (
                F.interpolate(
                    self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode="bicubic"
                ).contiguous()
                if (si != SN - 1)
                else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            )
            h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_Bl.append(
                f_hat.clone() if to_fhat else idx_N.reshape(B, ph * pw)
            )

        return f_hat_or_idx_Bl

    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)

        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = self.v_patch_nums[0]
        for si in range(SN - 1):
            if self.prog_si == 0 or (0 <= self.prog_si - 1 < si):
                break
            h_BChw = F.interpolate(
                self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, pn_next),
                size=(H, W),
                mode="bicubic",
            )
            f_hat.add_(self.quant_resi[si / (SN - 1)](h_BChw))
            pn_next = self.v_patch_nums[si + 1]
            next_scales.append(
                F.interpolate(f_hat, size=(pn_next, pn_next), mode="area")
                .view(B, C, -1)
                .transpose(1, 2)
            )
        return torch.cat(next_scales, dim=1) if len(next_scales) else None

    def get_next_autoregressive_input(
        self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        HW = self.v_patch_nums[-1]
        if si != SN - 1:
            h = self.quant_resi[si / (SN - 1)](
                F.interpolate(h_BChw, size=(HW, HW), mode="bicubic")
            )
            f_hat.add_(h)
            return f_hat, F.interpolate(
                f_hat, size=(self.v_patch_nums[si + 1], self.v_patch_nums[si + 1]), mode="area"
            )
        else:
            h = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat


class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(
            in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks // 2
        )
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BChw):
        return h_BChw.mul(1 - self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi

    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = (
            np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K)
            if K == 4
            else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)
        )

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) -> str:
        return f"ticks={self.ticks}"


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        K = len(qresi)
        self.ticks = (
            np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K)
            if K == 4
            else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)
        )

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())

    def extra_repr(self) -> str:
        return f"ticks={self.ticks}"
