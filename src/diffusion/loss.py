"""
loss.py
─────────────────────────────────────────────────────────────────────────────
Mixed loss function adapted to the column types of the dataset.

Single responsibility:
    Compute the weighted error between the predicted noise and the real noise,
    giving higher weight to binary and categorical columns.

Why not use plain MSE:
    Standard MSE treats all features equally. Predicting noise on a binary
    column (constrained to 0 or 1) is harder than on a continuous one.
    Upweighting those columns pushes the model to respect their discrete nature.

Weight scheme:
    continuous  → 1.0   (baseline)
    ordinal     → 1.2   (discrete, slightly more constrained)
    categorical → 1.5   (must predict correct class index)
    binary      → 2.0   (only 0 or 1, no values in between)

Classes:
    MixedLoss

Dependencies:
    torch

Part of:
    Synthetic Tabular Data Engine — diffusion/
"""

import torch
import torch.nn as nn


class MixedLoss(nn.Module):
    """
    Weighted MSE loss over noise predictions, scaled per column type.

    Args:
        column_types  : dict {feature_name: type} from PreprocessorOutput
        feature_names : ordered list of feature names (same order as tensor columns)
    """

    _WEIGHTS: dict[str, float] = {
        "continuous":  1.0,
        "ordinal":     1.2,
        "categorical": 1.5,
        "binary":      2.0,
    }

    def __init__(self, column_types: dict[str, str], feature_names: list[str]):
        super().__init__()

        weights = [
            self._WEIGHTS.get(column_types.get(name, "continuous"), 1.0)
            for name in feature_names
        ]

        # register_buffer: not a trainable parameter, but moves to the correct
        # device automatically when the module is moved with .to(device)
        self.register_buffer("weights", torch.tensor(weights))

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_real: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the weighted MSE between predicted and real noise.

        Args:
            noise_pred : shape (batch, n_features) — output of DenoisingMLP
            noise_real : shape (batch, n_features) — noise added in add_noise()

        Returns:
            loss : scalar — weighted mean squared error over the batch
        """
        squared_error  = (noise_pred - noise_real) ** 2   # (batch, n_features)
        weighted_error = squared_error * self.weights      # per-column weighting
        return weighted_error.mean()                       # scalar