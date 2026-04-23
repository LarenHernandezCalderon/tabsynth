"""
tuner.py
─────────────────────────────────────────────────────────────────────────────
Automatic hyperparameter tuning and early stopping for TabDDPM.

Single responsibility:
    1. Generate candidate configurations from dataset statistics
    2. Run a quick probe for each candidate and pick the best
    3. Provide an EarlyStopping utility so training stops automatically
       when the loss stops improving — no manual n_epochs needed

Strategy — Level 2 grid search + early stopping:
    Phase 1 — Grid search:
        Run 5 candidates for probe_epochs each, pick the lowest probe loss.
    Phase 2 — Full training with early stopping:
        Train the winner until the loss does not improve for `patience` epochs.
        A safety ceiling `max_epochs` prevents infinite training.

Early stopping logic:
    - Track the best loss seen so far (best_loss)
    - If current epoch loss < best_loss - min_delta → improvement, reset counter
    - If no improvement for `patience` epochs → stop training
    - min_delta: minimum change that counts as a real improvement (default: 1e-4)
      Avoids stopping due to tiny random fluctuations in the loss.

Classes:
    TunerConfig   — one candidate hyperparameter configuration
    EarlyStopping — stateful tracker that signals when to stop training
    HyperTuner    — runs the grid search and returns the best config

Dependencies:
    torch, dataclasses

Part of:
    Synthetic Tabular Data Engine — diffusion/
"""

import torch
from dataclasses import dataclass, field
from torch.utils.data import DataLoader

from .mlp             import DenoisingMLP, auto_hidden_dims
from .loss            import MixedLoss
from .noise_scheduler import NoiseScheduler


# ── TunerConfig ──────────────────────────────────────────────────────────────

@dataclass
class TunerConfig:
    """
    One candidate hyperparameter configuration.

    Fields:
        hidden_dims : MLP hidden layer sizes
        lr          : Adam learning rate
        batch_size  : samples per gradient step
        label       : human-readable name for logging
    """
    hidden_dims: list[int]
    lr:          float
    batch_size:  int
    label:       str


# ── EarlyStopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stateful tracker that signals when training should stop.

    How it works:
        Call .step(loss) after every epoch. It returns True when training
        should stop (no improvement for `patience` epochs), False otherwise.

        An improvement is defined as: new_loss < best_loss - min_delta
        The min_delta threshold filters out tiny random fluctuations that
        would otherwise reset the patience counter unnecessarily.

    Args:
        patience  : epochs to wait without improvement before stopping (default: 50)
        min_delta : minimum loss reduction that counts as improvement (default: 1e-4)
        max_epochs: hard ceiling — always stop here even if still improving (default: 2000)

    Usage:
        stopper = EarlyStopping(patience=50, max_epochs=2000)
        for epoch in range(stopper.max_epochs):
            loss = train_one_epoch(...)
            if stopper.step(loss):
                print(f"Early stop at epoch {epoch+1}")
                break
    """

    def __init__(
        self,
        patience:   int   = 50,
        min_delta:  float = 1e-4,
        max_epochs: int   = 2000,
    ):
        self.patience   = patience
        self.min_delta  = min_delta
        self.max_epochs = max_epochs

        self._best_loss    = float("inf")
        self._counter      = 0          # epochs since last improvement
        self.best_epoch    = 0          # epoch index of the best loss seen
        self.stopped_epoch = 0          # epoch index where training stopped

    def step(self, loss: float, epoch: int) -> bool:
        """
        Evaluates whether training should stop after this epoch.

        Args:
            loss  : mean training loss of the current epoch
            epoch : current epoch index (0-based)

        Returns:
            True  → stop training now
            False → keep going
        """
        if loss < self._best_loss - self.min_delta:
            # Genuine improvement — reset the patience counter
            self._best_loss    = loss
            self._counter      = 0
            self.best_epoch    = epoch
        else:
            # No meaningful improvement — increment counter
            self._counter += 1

        if self._counter >= self.patience:
            self.stopped_epoch = epoch
            return True   # stop

        return False      # keep going

    @property
    def best_loss(self) -> float:
        return self._best_loss

    def status(self) -> str:
        """Returns a short status string for logging."""
        return f"patience {self._counter}/{self.patience} | best {self._best_loss:.6f}"


# ── HyperTuner ───────────────────────────────────────────────────────────────

class HyperTuner:
    """
    Runs a quick grid search over candidate configurations and returns the best.

    Usage:
        tuner = HyperTuner(n_rows, n_features, column_types, feature_names)
        best  = tuner.run(tensor_dataset, T=1000, probe_epochs=50)
    """

    def __init__(
        self,
        n_rows:        int,
        n_features:    int,
        column_types:  dict[str, str],
        feature_names: list[str],
    ):
        self.n_rows        = n_rows
        self.n_features    = n_features
        self.column_types  = column_types
        self.feature_names = feature_names
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_candidates(self) -> list[TunerConfig]:
        """
        Generates 5 candidate configurations based on dataset statistics.

        Rules applied:
            n_rows < 500        → small batches (32 / 64)
            n_rows 500-5000     → medium batches (64 / 128)
            n_rows > 5000       → large batches (256 / 512)
            n_features drives hidden layer width via auto_hidden_dims()

        Candidates:
            A — small / fast baseline
            B — standard  (auto_hidden_dims default)
            C — one extra hidden layer (deeper)
            D — wider first layer + lower lr (slower but more precise)
            E — large network for complex distributions
        """
        base = auto_hidden_dims(self.n_features)

        if self.n_rows < 500:
            bs_small, bs_large = 32, 64
        elif self.n_rows < 5000:
            bs_small, bs_large = 64, 128
        else:
            bs_small, bs_large = 256, 512

        return [
            TunerConfig(
                hidden_dims = [max(64, base[0] // 2)] * 2,
                lr          = 1e-3,
                batch_size  = bs_small,
                label       = "A — small/fast",
            ),
            TunerConfig(
                hidden_dims = base,
                lr          = 1e-3,
                batch_size  = bs_small,
                label       = "B — standard",
            ),
            TunerConfig(
                hidden_dims = base + [base[-1]],
                lr          = 1e-3,
                batch_size  = bs_small,
                label       = "C — deeper",
            ),
            TunerConfig(
                hidden_dims = [base[0] * 2] + base,
                lr          = 5e-4,
                batch_size  = bs_large,
                label       = "D — wide + low lr",
            ),
            TunerConfig(
                hidden_dims = [512, 512, 256],
                lr          = 1e-3,
                batch_size  = bs_small,
                label       = "E — large network",
            ),
        ]

    def _probe(
        self,
        candidate:     TunerConfig,
        tensor_dataset,
        T:             int,
        probe_epochs:  int,
    ) -> float:
        """
        Trains a fresh model with candidate config for probe_epochs.
        Returns the final mean loss (lower = better candidate).
        """
        model     = DenoisingMLP(self.n_features, candidate.hidden_dims).to(self.device)
        scheduler = NoiseScheduler(T=T, schedule="cosine")
        loss_fn   = MixedLoss(self.column_types, self.feature_names).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=candidate.lr)
        loader    = DataLoader(tensor_dataset, batch_size=candidate.batch_size, shuffle=True)

        model.train()
        last_loss = float("inf")

        for _ in range(probe_epochs):
            epoch_losses = []
            for batch in loader:
                x_real = batch[0].to(self.device)
                t      = torch.randint(0, T, (len(x_real),), device=self.device)

                x_t, noise = scheduler.add_noise(x_real.cpu(), t.cpu())
                x_t   = x_t.to(self.device)
                noise = noise.to(self.device)

                noise_pred = model(x_t, t)
                loss       = loss_fn(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            last_loss = sum(epoch_losses) / len(epoch_losses)

        # Free memory before next candidate
        del model, optimizer, loader
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return last_loss

    def run(
        self,
        tensor_dataset,
        T:            int = 1000,
        probe_epochs: int = 50,
    ) -> TunerConfig:
        """
        Runs quick trials for all candidates and returns the best config.

        Args:
            tensor_dataset : TensorDataset from PreprocessorOutput
            T              : diffusion timesteps
            probe_epochs   : epochs per candidate (default: 50)

        Returns:
            TunerConfig with the lowest probe loss
        """
        candidates  = self._build_candidates()
        best_config = None
        best_loss   = float("inf")

        print("=" * 52)
        print("Auto-tuning — grid search")
        print(f"  Dataset      : {self.n_rows} rows x {self.n_features} features")
        print(f"  Candidates   : {len(candidates)}")
        print(f"  Probe epochs : {probe_epochs} each")
        print("=" * 52)

        for i, candidate in enumerate(candidates, 1):
            print(f"  [{i}/{len(candidates)}] {candidate.label:<26}", end="", flush=True)
            probe_loss = self._probe(candidate, tensor_dataset, T, probe_epochs)
            marker     = " ← best so far" if probe_loss < best_loss else ""
            print(f"loss: {probe_loss:.6f}{marker}")

            if probe_loss < best_loss:
                best_loss   = probe_loss
                best_config = candidate

        print()
        print(f"  Winner       : {best_config.label}")
        print(f"  hidden_dims  : {best_config.hidden_dims}")
        print(f"  lr           : {best_config.lr}")
        print(f"  batch_size   : {best_config.batch_size}")
        print(f"  probe loss   : {best_loss:.6f}")
        print("=" * 52)
        print()

        return best_config