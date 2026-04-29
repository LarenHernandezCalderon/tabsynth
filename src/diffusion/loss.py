"""
loss.py
─────────────────────────────────────────────────────────────────────────────
Loss functions for the TabDDPM diffusion model.

Classes:
    MixedLoss       — weighted MSE on noise predictions, scaled by column type
    CorrelationLoss — penalizes deviation from real pairwise correlations
                      in the reconstructed x_0, using critical_pairs from Phase 1

MixedLoss weight scheme:
    continuous  → 1.0
    ordinal     → 1.2
    categorical → 1.5
    binary      → 2.0

CorrelationLoss:
    During training, reconstructs x_0_pred from (x_t, noise_pred, alpha_bar[t]),
    then for each critical pair (col_a, col_b) computes the Pearson correlation
    and penalizes the squared difference from the real correlation.
    Only applied to numeric pairs (pearson / spearman method).

Dependencies:
    torch

Part of:
    Synthetic Tabular Data Engine — diffusion/
"""

import torch
import torch.nn as nn


# ── MixedLoss ─────────────────────────────────────────────────────────────────

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

        self.register_buffer("weights", torch.tensor(weights))

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_real: torch.Tensor,
    ) -> torch.Tensor:
        squared_error  = (noise_pred - noise_real) ** 2
        weighted_error = squared_error * self.weights
        return weighted_error.mean()


# ── CorrelationLoss ───────────────────────────────────────────────────────────

class CorrelationLoss(nn.Module):
    """
    Penalizes deviation from real pairwise correlations in x_0_pred.

    Why this matters:
        The MixedLoss works in noise space — it has no signal about whether
        the joint distribution between columns is correct. A model can have
        low noise-prediction loss but still produce Pclass=1 with Fare=5,
        because it never sees the pair jointly penalized.

        CorrelationLoss reconstructs the clean data estimate x_0_pred from
        (x_t, noise_pred) and measures the Pearson correlation between each
        critical pair. If Pclass and Fare are uncorrelated in x_0_pred but
        should be correlated at -0.55, this loss pushes the model to fix it.

    Args:
        pairs : list of (idx_a, idx_b, target_corr) where:
                    idx_a, idx_b    : column indices in the tensor
                    target_corr     : real Pearson correlation (signed, -1 to 1)

    Usage (in training loop):
        # Reconstruct x_0_pred
        ab       = scheduler.alpha_bar[t.cpu()].unsqueeze(1).to(device)
        x_0_pred = (x_t - torch.sqrt(1.0 - ab) * noise_pred) / (torch.sqrt(ab) + 1e-8)
        x_0_pred = torch.clamp(x_0_pred, 0.0, 1.0)

        # Compute losses
        loss = mixed_loss(noise_pred, noise) + weight * corr_loss(x_0_pred)
    """

    def __init__(self, pairs: list[tuple[int, int, float]]):
        super().__init__()
        self.pairs = pairs  # [(idx_a, idx_b, target_corr), ...]

    def forward(self, x_0_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_0_pred : reconstructed clean data, shape (batch, n_features)
                       values in [0, 1]

        Returns:
            scalar loss — mean squared correlation error across all critical pairs
        """
        if not self.pairs or x_0_pred.shape[0] < 4:
            return torch.tensor(0.0, device=x_0_pred.device)

        total = torch.tensor(0.0, device=x_0_pred.device)

        for idx_a, idx_b, target_corr in self.pairs:
            pred_corr = self._pearson(
                x_0_pred[:, idx_a],
                x_0_pred[:, idx_b],
            )
            total = total + (pred_corr - target_corr) ** 2

        return total / len(self.pairs)

    @staticmethod
    def _pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Differentiable Pearson correlation between two vectors.
        Returns a scalar in [-1, 1].
        """
        a = a - a.mean()
        b = b - b.mean()
        denom = (a.norm() * b.norm()) + 1e-8
        return (a * b).sum() / denom