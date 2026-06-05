"""
Image utility functions for AffordanceVLA.

Provides normalization, format conversion, and aspect-ratio-preserving resize
with padding for observation images.
"""

import numpy as np
from PIL import Image
import torch.nn.functional as F


def normalize_01_into_pm1(x):
    """Convert [0, 1] range to [-1, 1] range."""
    return x.add(x).add_(-1)


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert float image to uint8 (0-255)."""
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def _resize_with_pad_pil(
    image: Image.Image, height: int, width: int, method: int
) -> Image.Image:
    """Resize a PIL image preserving aspect ratio with zero padding."""
    cur_height, cur_width = image.height, image.width
    if cur_width == width and cur_height == height:
        return image

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    return zero_image


def resize_with_pad(img, width, height, pad_value=-1):
    """Resize 4D tensor (B,C,H,W) preserving aspect ratio with padding.

    Padding is applied to the left and top of the image.

    Args:
        img: 4D tensor of shape (B, C, H, W).
        width: Target width.
        height: Target height.
        pad_value: Value used for padding.

    Returns:
        Resized and padded tensor of shape (B, C, height, width).
    """
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]
    if cur_height == height and cur_width == width:
        return img

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def resize_with_pad_eval(img, width, height, pad_value=-1):
    """Resize 3D or 4D tensor preserving aspect ratio with padding.

    Supports both single images (C,H,W) and batches (B,C,H,W).

    Args:
        img: 3D or 4D tensor.
        width: Target width.
        height: Target height.
        pad_value: Value used for padding.

    Returns:
        Resized and padded tensor.
    """
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
        squeeze_output = True
    elif len(img.shape) == 4:
        squeeze_output = False
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {img.shape}")

    cur_height, cur_width = img.shape[2:]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)

    if squeeze_output:
        padded_img = padded_img.squeeze(0)
    return padded_img
