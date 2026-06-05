"""
Core utility functions for AffordanceVLA model.

Includes attention mask construction, sinusoidal embeddings for flow matching,
vector padding, and top-k/top-p sampling.
"""

import math
import torch
import torch.nn.functional as F


def create_sinusoidal_pos_embedding(
    timesteps: torch.Tensor,
    dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
    device=None,
) -> torch.Tensor:
    """Create sinusoidal position embedding for flow matching timesteps.

    Maps scalar timesteps to high-dimensional embeddings using log-spaced
    sinusoidal frequencies, providing fine-grained discrimination across [0, 1].

    Args:
        timesteps: [B] tensor of timestep values (typically in [0, 1]).
        dim: Embedding dimension.
        min_period: Minimum period of sinusoidal functions.
        max_period: Maximum period of sinusoidal functions.
        device: Target device.

    Returns:
        [B, dim] embedding tensor.
    """
    half_dim = dim // 2
    freqs = torch.exp(
        torch.arange(half_dim, device=device, dtype=torch.float32)
        * (-math.log(max_period / min_period) / (half_dim - 1))
    ) / min_period

    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )

    return embedding


def make_att_2d_masks(
    pad_masks: torch.Tensor, att_masks: torch.Tensor
) -> torch.Tensor:
    """Create 2D attention masks from 1D padding and block-group masks.

    Uses block-causal attention: tokens are grouped into blocks by cumulative
    sum of att_masks. A token in block k can attend to all tokens in blocks
    0, 1, ..., k (i.e., same or earlier blocks).

    Block assignment:
        - att_masks value 0: continues the current block
        - att_masks value >0: starts a new block

    Args:
        pad_masks: [B, L] boolean mask (True = valid token).
        att_masks: [B, L] mask where non-zero values start new causal blocks.

    Returns:
        [B, L, L] boolean mask where True means attention is allowed.
    """
    block_ids = torch.cumsum(att_masks.to(dtype=torch.long), dim=1)

    # Token i (block b_i) can attend to token j (block b_j) iff b_j <= b_i
    causal_mask = block_ids[:, None, :] <= block_ids[:, :, None]  # [B, L, L]

    # Also respect padding: cannot attend to padded positions
    padding_mask = pad_masks[:, None, :]  # [B, 1, L]

    return causal_mask & padding_mask


def pad_vector(vector: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Pad or truncate the last dimension of a vector to target_dim.

    Args:
        vector: Tensor of shape [..., D].
        target_dim: Target dimension for the last axis.

    Returns:
        Tensor of shape [..., target_dim].
    """
    if vector.shape[-1] >= target_dim:
        return vector[..., :target_dim]
    pad_size = target_dim - vector.shape[-1]
    return F.pad(vector, (0, pad_size), value=0)


def sample_with_top_k_top_p_(
    logits_BlV: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    rng=None,
    num_samples: int = 1,
) -> torch.Tensor:
    """Sample from logits with top-k and top-p (nucleus) filtering.

    Modifies logits_BlV in-place by masking out low-probability tokens.

    Args:
        logits_BlV: [B, l, V] logits tensor (modified in-place).
        top_k: Keep only top-k tokens with highest probability.
        top_p: Keep tokens with cumulative probability <= top_p.
        rng: Optional random number generator.
        num_samples: Number of samples to draw.

    Returns:
        [B, l, num_samples] sampled token indices.
    """
    B, l, V = logits_BlV.shape
    if top_k > 0:
        idx_to_remove = logits_BlV < logits_BlV.topk(
            top_k, largest=True, sorted=False, dim=-1
        )[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = (
            sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        )
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(
            sorted_idx_to_remove.scatter(
                sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove
            ),
            -torch.inf,
        )
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(
        logits_BlV.softmax(dim=-1).view(-1, V),
        num_samples=num_samples,
        replacement=replacement,
        generator=rng,
    ).view(B, l, num_samples)
