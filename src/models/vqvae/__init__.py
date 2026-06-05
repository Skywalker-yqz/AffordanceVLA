"""
Multi-scale VQ-VAE for discrete visual tokenization.

Used as a frozen encoder in AffordanceVLA's Which2Act decoder to produce
ground-truth codebook indices.
"""

from .vqvae import VQVAE
from .quant import VectorQuantizer2
from .basic_vae import Encoder, Decoder
