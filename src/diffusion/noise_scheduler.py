"""
noise_scheduler.py
─────────────────────────────────────────────────────────────────────────────
Manages the noise schedule for the forward and reverse diffusion processes.

Single responsibility:
    Know how much noise exists at each timestep t, and apply / undo that noise.
    This module does NOT train anything and does NOT generate anything.
    It is pure mathematics of the diffusion process.

Core concept — alpha_bar:
    alpha_bar[t] = how much of the original signal remains at timestep t.
        alpha_bar[0]   ≈ 1.0  →  almost all signal, almost no noise
        alpha_bar[500] ≈ 0.5  →  half signal, half noise
        alpha_bar[999] ≈ 0.0  →  almost pure noise

    Forward process formula (adding noise):
        x_t = sqrt(alpha_bar[t]) * x_0  +  sqrt(1 - alpha_bar[t]) * noise
        where noise ~ N(0, 1)

Classes:
    NoiseScheduler

Dependencies:
    torch, math

Part of:
    Synthetic Tabular Data Engine — diffusion/
"""

import math
import torch


class NoiseScheduler:
    """
    Manages the noise calendar (alpha_bar) for the diffusion process.

    Args:
        T        : number of diffusion timesteps (default: 1000)
        schedule : noise schedule type — 'cosine' (recommended) or 'linear'
    """

    def __init__(self, T: int = 1000, schedule: str = "cosine"):
        self.T        = T
        self.schedule = schedule

        self.alpha_bar = self._build_schedule()

        # alpha_bar_prev: alpha_bar shifted one step back.
        # Required in reverse_step to compute how much noise to remove.
        # We prepend 1.0 because before step 0 everything was pure signal.
        self.alpha_bar_prev = torch.cat([
            torch.tensor([1.0]),
            self.alpha_bar[:-1]
        ])

    def _build_schedule(self) -> torch.Tensor:
        """
        Builds the alpha_bar tensor for timesteps 0 .. T-1.

        Cosine schedule (Nichol & Dhariwal, 2021):
            Adds very little noise at the start and end of the process.
            Preserves data structure better than the linear variant.
            Recommended for tabular data.

        Linear schedule (Ho et al., 2020):
            Betas (noise amount per step) grow linearly from beta_start to beta_end.
            Simpler, but introduces too much noise in the early steps.
        """
        if self.schedule == "cosine":
            steps     = torch.linspace(0, self.T, self.T + 1)
            f         = torch.cos(((steps / self.T) + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = f / f[0]
            alpha_bar = alpha_bar[:-1]
            return torch.clamp(alpha_bar, min=1e-4, max=1 - 1e-4)

        elif self.schedule == "linear":
            betas     = torch.linspace(1e-4, 0.02, self.T)
            alphas    = 1.0 - betas
            alpha_bar = torch.cumprod(alphas, dim=0)
            return torch.clamp(alpha_bar, min=1e-4, max=1 - 1e-4)

        else:
            raise ValueError(
                f"Unknown schedule: '{self.schedule}'. Use 'cosine' or 'linear'."
            )

    def add_noise(
        self,
        x_0: torch.Tensor,
        t:   torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward process: mixes x_0 with Gaussian noise at timestep t.

        Formula: x_t = sqrt(ab[t]) * x_0 + sqrt(1 - ab[t]) * noise

        Args:
            x_0 : original clean data,         shape (batch, n_features)
            t   : timestep per sample,          shape (batch,)

        Returns:
            x_t   : noisy data,                 shape (batch, n_features)
            noise : actual noise added,         shape (batch, n_features)
                    Stored so MixedLoss can compare it against the predicted noise.
        """
        # Unsqueeze to (batch, 1) so it broadcasts correctly over (batch, n_features)
        ab    = self.alpha_bar[t].unsqueeze(1)
        noise = torch.randn_like(x_0)
        x_t   = torch.sqrt(ab) * x_0 + torch.sqrt(1.0 - ab) * noise
        return x_t, noise

    def reverse_step(
        self,
        x_t:        torch.Tensor,
        noise_pred: torch.Tensor,
        t:          int,
    ) -> torch.Tensor:
        """
        One step of the reverse process: moves x_t one step closer to x_0.

        Applied T times during generation, starting from pure Gaussian noise
        and ending at the synthetic data sample.

        Formula (DDPM, Ho et al. 2020):
            mean    = (1 / sqrt(1 - beta)) * (x_t - beta / sqrt(1 - ab) * noise_pred)
            x_{t-1} = mean + sqrt(beta) * z      if t > 0  (stochastic)
            x_{t-1} = mean                        if t == 0 (final output, no extra noise)

        Args:
            x_t        : noisy data at step t,       shape (batch, n_features)
            noise_pred : noise predicted by the MLP,  shape (batch, n_features)
            t          : current timestep (plain int, not a tensor)

        Returns:
            x_{t-1} : slightly denoised data,        shape (batch, n_features)
        """
        ab_t    = self.alpha_bar[t]
        ab_prev = self.alpha_bar_prev[t]
        beta_t  = 1.0 - ab_t / ab_prev

        coef1 = 1.0 / torch.sqrt(1.0 - beta_t)
        coef2 = beta_t / torch.sqrt(1.0 - ab_t)
        mean  = coef1 * (x_t - coef2 * noise_pred)

        if t > 0:
            # Add a small amount of noise to keep the generation stochastic
            sigma = torch.sqrt(beta_t)
            return mean + sigma * torch.randn_like(x_t)
        else:
            # Last step: return the clean mean directly
            return mean