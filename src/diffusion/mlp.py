"""
mlp.py
─────────────────────────────────────────────────────────────────────────────
Denoising neural network architecture for the diffusion model.

Single responsibility:
    Define the MLP that learns to predict the noise added at timestep t.
    This module knows nothing about the diffusion process itself.
    It simply receives (x_t, t) and returns noise_pred.

Classes:
    SinusoidalTimestepEmbedding  — encodes integer timestep t as a dense vector
    ResidualBlock                — Linear + LayerNorm + ReLU with skip connection
    DenoisingMLP                 — the full denoising network

Functions:
    auto_hidden_dims()           — infers MLP hidden layer sizes from n_features

Architecture overview:
    x_t ──► input_proj ──► h ──(+ t_emb)──► residual blocks ──► output_proj ──► noise_pred
    t   ──► time_embed ──► time_proj ──► t_emb

Dependencies:
    torch, math

Part of:
    Synthetic Tabular Data Engine — diffusion/
"""

import math
import torch
import torch.nn as nn


def auto_hidden_dims(n_features: int) -> list[int]:
    """
    Infers suitable hidden layer dimensions based on the number of features.

    Rationale: larger datasets have more complex joint distributions and
    require bigger networks to capture them accurately. The user can always
    override this by passing hidden_dims explicitly to TabDDPM.

    Args:
        n_features : number of tensor features (PreprocessorOutput.n_features)

    Returns:
        list of hidden dimensions, e.g. [256, 256] or [512, 512, 512]
    """
    if n_features <= 20:
        return [256, 256]
    elif n_features <= 50:
        return [256, 256, 256]
    elif n_features <= 100:
        return [512, 512, 256]
    else:
        return [512, 512, 512, 256]


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Encodes an integer timestep t as a sinusoidal embedding vector.

    Why not pass t directly as a number:
        A single integer 0..999 is too sparse for a network to learn smooth
        patterns over. The sinusoidal embedding produces a unique dense vector
        per timestep using multiple frequency components — the same idea as
        positional encoding in transformers (Vaswani et al., 2017).

    Property: nearby timesteps produce similar embeddings;
    distant timesteps produce clearly different ones.

    Args:
        dim : embedding dimension (must be even — split equally into sin / cos)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t : integer timesteps, shape (batch,)

        Returns:
            embedding : shape (batch, dim)
        """
        device = t.device
        half   = self.dim // 2

        # Geometric frequency sequence: low freqs capture global structure,
        # high freqs capture fine-grained timestep differences
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / half
        )

        # Outer product: (batch, 1) * (1, half) → (batch, half)
        args = t[:, None].float() * freqs[None, :]

        # Concatenate sin and cos halves → (batch, dim)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    """
    Residual block: Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm → (+ input) → ReLU

    The skip connection (output = block(input) + input) has two benefits:
        1. The block only needs to learn the residual difference — easier to optimise.
        2. Gradients flow directly backward without vanishing in deep networks.

    LayerNorm normalises within each individual sample rather than across the
    batch, which is more stable with variable batch sizes.

    Args:
        dim     : input and output dimension (must match to allow the element-wise sum)
        dropout : fraction of neurons zeroed during training (0.0 = disabled)
    """

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skip connection: add input to the transformed output before activation
        return self.relu(self.block(x) + x)


class DenoisingMLP(nn.Module):
    """
    Neural network that predicts the noise added to data at timestep t.

    Data flow:
        x_t  ──► input_proj  ──► h
        t    ──► time_embed  ──► time_proj  ──► t_emb  ──► h = h + t_emb
                                                            h ──► residual blocks
                                                                  ──► output_proj
                                                                        ──► noise_pred

    Args:
        n_features  : number of input / output features
        hidden_dims : list of hidden layer sizes, e.g. [256, 256, 256]
        dropout     : regularisation rate (0.0 recommended for small datasets)
    """

    def __init__(
        self,
        n_features:  int,
        hidden_dims: list[int],
        dropout:     float = 0.0,
    ):
        super().__init__()

        # Timestep embedding size matches the first hidden dimension
        dim_t = hidden_dims[0]

        self.time_embed = SinusoidalTimestepEmbedding(dim_t)

        # Input projection: map n_features into the first hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dims[0]),
            nn.ReLU(),
        )

        # Timestep projection: bring the timestep embedding into the same space
        self.time_proj = nn.Sequential(
            nn.Linear(dim_t, hidden_dims[0]),
            nn.ReLU(),
        )

        # Residual blocks — with linear transitions when dimension changes
        blocks = []
        for i, dim in enumerate(hidden_dims):
            blocks.append(ResidualBlock(dim, dropout))
            if i < len(hidden_dims) - 1 and hidden_dims[i + 1] != dim:
                blocks.append(nn.Linear(dim, hidden_dims[i + 1]))
                blocks.append(nn.ReLU())
        self.blocks = nn.Sequential(*blocks)

        # Output projection: map back to n_features (the predicted noise)
        self.output_proj = nn.Linear(hidden_dims[-1], n_features)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t : noisy data,    shape (batch, n_features)
            t   : timestep,      shape (batch,)  integer values 0..T-1

        Returns:
            noise_pred : predicted noise, shape (batch, n_features)
        """
        h     = self.input_proj(x_t)               # project data into hidden space
        t_emb = self.time_proj(self.time_embed(t))  # project timestep embedding
        h     = h + t_emb                          # inject timestep information
        h     = self.blocks(h)                     # pass through residual blocks
        return self.output_proj(h)                 # project back to feature space