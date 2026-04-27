"""
tabddpm.py
─────────────────────────────────────────────────────────────────────────────
Main orchestrator of the TabDDPM diffusion model (Phase 3).

Single responsibility:
    Coordinate NoiseScheduler, DenoisingMLP, and MixedLoss to train the model
    and generate synthetic tabular data.

Public interface:
    model = TabDDPM.from_preprocessor(preprocessor_output)

    # Fully automatic — no hyperparameters needed
    model.auto_tune(output.tensor_dataset)

    # Manual — full control
    model.fit(output.tensor_dataset, n_epochs=1000, batch_size=128, lr=1e-3)

    synthetic_tensor = model.generate(n_samples=len(df))
    model.save("models/titanic_tabddpm.pt")
    model2 = TabDDPM.load("models/titanic_tabddpm.pt")

Dependencies:
    torch, collections

Part of:
    Synthetic Tabular Data Engine — diffusion/
"""

import torch
from torch.utils.data import DataLoader
from collections import Counter

from .noise_scheduler import NoiseScheduler
from .mlp             import DenoisingMLP, auto_hidden_dims
from .loss            import MixedLoss
from .tuner           import HyperTuner, EarlyStopping


class TabDDPM:
    """
    Denoising Diffusion Probabilistic Model for mixed tabular data.

    Coordinates three internal components:
        NoiseScheduler — mathematical noise calendar (forward + reverse)
        DenoisingMLP   — neural network that learns to predict and undo noise
        MixedLoss      — error function weighted by column type

    Args:
        n_features    : number of tensor features   (PreprocessorOutput.n_features)
        column_types  : dict feature → type         (PreprocessorOutput.column_types)
        feature_names : ordered list of names       (PreprocessorOutput.feature_names)
        n_rows        : number of dataset rows      (used by auto_tune)
        T             : number of diffusion steps    (default: 1000)
        schedule      : noise schedule type          ('cosine' or 'linear')
        hidden_dims   : MLP hidden layer sizes       (None = inferred from n_features)
        dropout       : MLP regularisation rate      (default: 0.0)
    """

    def __init__(
        self,
        n_features:    int,
        column_types:  dict[str, str],
        feature_names: list[str],
        n_rows:        int              = 0,
        T:             int              = 1000,
        schedule:      str              = "cosine",
        hidden_dims:   list[int] | None = None,
        dropout:       float            = 0.0,
    ):
        self.n_features    = n_features
        self.column_types  = column_types
        self.feature_names = feature_names
        self.n_rows        = n_rows
        self.T             = T
        self.dropout       = dropout

        if hidden_dims is None:
            hidden_dims = auto_hidden_dims(n_features)
        self.hidden_dims = hidden_dims

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._build_components()
        self._is_fitted = False

    def _build_components(self) -> None:
        """Builds or rebuilds the three internal components."""
        self.scheduler = NoiseScheduler(T=self.T, schedule="cosine")

        self.model = DenoisingMLP(
            n_features  = self.n_features,
            hidden_dims = self.hidden_dims,
            dropout     = self.dropout,
        ).to(self.device)

        self.loss_fn = MixedLoss(
            column_types  = self.column_types,
            feature_names = self.feature_names,
        ).to(self.device)

    # ── Alternative constructor ──────────────────────────────────────────────

    @classmethod
    def from_preprocessor(cls, preprocessor_output, **kwargs) -> "TabDDPM":
        """
        Builds a TabDDPM instance directly from a Phase 2 PreprocessorOutput.

        Usage:
            model = TabDDPM.from_preprocessor(output)
            model = TabDDPM.from_preprocessor(output, T=1000, hidden_dims=[512,256])
        """
        return cls(
            n_features    = preprocessor_output.n_features,
            column_types  = preprocessor_output.column_types,
            feature_names = preprocessor_output.feature_names,
            n_rows        = len(preprocessor_output.tensor_dataset),
            **kwargs,
        )

    # ── Manual training ──────────────────────────────────────────────────────

    def fit(
        self,
        tensor_dataset,
        n_epochs:   int   = 1000,
        batch_size: int   = 512,
        lr:         float = 1e-3,
        log_every:  int   = 100,
        patience:   int   = 0,       # 0 = early stopping disabled
        min_delta:  float = 1e-4,
    ) -> list[float]:
        """
        Trains the diffusion model with manually specified hyperparameters.

        Optionally uses early stopping if patience > 0.

        Per batch, per epoch:
            1. Sample a random timestep t per data point
            2. Add noise:        x_t, noise = scheduler.add_noise(x, t)
            3. Predict noise:    noise_pred = model(x_t, t)
            4. Compute loss:     loss = loss_fn(noise_pred, noise)
            5. Update weights:   zero_grad → backward → step

        Args:
            tensor_dataset : PyTorch TensorDataset (from PreprocessorOutput)
            n_epochs       : maximum number of training epochs
            batch_size     : samples per gradient update
            lr             : Adam learning rate
            log_every      : print progress every N epochs
            patience       : early stopping patience (0 = disabled)
            min_delta      : minimum loss improvement to reset patience counter

        Returns:
            loss_history : mean loss per epoch
        """
        loader    = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        stopper   = EarlyStopping(patience=patience, min_delta=min_delta) if patience > 0 else None
        self.model.train()

        self._print_training_header(n_epochs, batch_size, lr, patience)

        loss_history: list[float] = []

        for epoch in range(n_epochs):
            epoch_losses: list[float] = []

            for batch in loader:
                x_real = batch[0].to(self.device)
                t      = torch.randint(0, self.T, (len(x_real),), device=self.device)

                x_t, noise = self.scheduler.add_noise(x_real.cpu(), t.cpu())
                x_t   = x_t.to(self.device)
                noise = noise.to(self.device)

                noise_pred = self.model(x_t, t)
                loss       = self.loss_fn(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            mean_loss = sum(epoch_losses) / len(epoch_losses)
            loss_history.append(mean_loss)

            if (epoch + 1) % log_every == 0 or epoch == 0:
                status = f" | {stopper.status()}" if stopper else ""
                print(f"  Epoch {epoch + 1:4d}/{n_epochs} | Loss: {mean_loss:.6f}{status}")

            # Early stopping check
            if stopper and stopper.step(mean_loss, epoch):
                print(f"\n  Early stop at epoch {epoch + 1} "
                      f"(no improvement for {stopper.patience} epochs)")
                print(f"  Best loss: {stopper.best_loss:.6f} at epoch {stopper.best_epoch + 1}")
                break

        print(f"\nTraining complete. Final loss: {loss_history[-1]:.6f}")
        self._is_fitted = True
        return loss_history

    # ── Fully automatic training ─────────────────────────────────────────────

    def auto_tune(
        self,
        tensor_dataset,
        probe_epochs: int   = 50,
        patience:     int   = 50,
        min_delta:    float = 1e-4,
        max_epochs:   int   = 2000,
        log_every:    int   = 100,
    ) -> list[float]:
        """
        Fully automatic training — no hyperparameters needed from the user.

        Two-phase process:
            Phase 1 — Grid search (quick):
                Probes 5 candidate configurations for probe_epochs each.
                Picks the winner based on lowest probe loss.
                Automatically selects: hidden_dims, batch_size, lr.

            Phase 2 — Full training with early stopping:
                Rebuilds the model with the winning configuration.
                Trains until the loss stops improving for `patience` epochs.
                Never trains more than `max_epochs` epochs.

        This is the recommended entry point for any new dataset.
        The system adapts entirely to your data — just call auto_tune() and wait.

        Args:
            tensor_dataset : TensorDataset from PreprocessorOutput
            probe_epochs   : epochs per candidate in the grid search (default: 50)
            patience       : early stopping — stop if no improvement for N epochs (default: 50)
            min_delta      : minimum loss change that counts as improvement (default: 1e-4)
            max_epochs     : hard ceiling on training epochs (default: 2000)
            log_every      : print interval during full training (default: 100)

        Returns:
            loss_history : mean loss per epoch of the full training run

        Usage:
            model = TabDDPM.from_preprocessor(output)
            model.auto_tune(output.tensor_dataset)   # ← that's it
            synthetic_tensor = model.generate(n_samples=len(df))
        """
        # ── Phase 1: find best config ────────────────────────────────────────
        tuner = HyperTuner(
            n_rows        = self.n_rows,
            n_features    = self.n_features,
            column_types  = self.column_types,
            feature_names = self.feature_names,
        )
        best = tuner.run(tensor_dataset, T=self.T, probe_epochs=probe_epochs)

        # ── Phase 2: rebuild model with winner and train with early stopping ──
        self.hidden_dims = best.hidden_dims
        self._build_components()

        print(f"Starting full training — winner: {best.label}")
        print(f"  Early stopping: patience={patience}, max_epochs={max_epochs}")
        print()

        return self.fit(
            tensor_dataset,
            n_epochs   = max_epochs,
            batch_size = best.batch_size,
            lr         = best.lr,
            log_every  = log_every,
            patience   = patience,
            min_delta  = min_delta,
        )

    # ── Generation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, n_samples: int) -> torch.Tensor:
        """
        Generates n_samples rows of synthetic data via the full reverse process.

        Args:
            n_samples : number of synthetic rows to generate

        Returns:
            tensor of shape (n_samples, n_features) with values in [0, 1]
            ready for PreprocessorOutput.inverse_transform()
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model has not been trained yet. Call fit() or auto_tune() first."
            )

        self.model.eval()

        # Fix the generation seed for reproducibility.
        # The training seed is intentionally NOT fixed (we want stochastic training).
        # Only the generation is fixed so results are reproducible across runs.
        torch.manual_seed(42)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(42)

        x = torch.randn(n_samples, self.n_features, device=self.device)

        print(f"Generating {n_samples} synthetic samples ({self.T} reverse steps)...")

        for t_val in reversed(range(self.T)):
            t_tensor   = torch.full((n_samples,), t_val, dtype=torch.long, device=self.device)
            noise_pred = self.model(x, t_tensor)
            x          = self.scheduler.reverse_step(x, noise_pred, t_val)

        x = torch.clamp(x, 0.0, 1.0)
        print(f"Generation complete. Output shape: {x.shape}")

        self.model.train()
        return x.cpu()

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Saves the trained model and hyperparameters to disk."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "n_features":       self.n_features,
            "column_types":     self.column_types,
            "feature_names":    self.feature_names,
            "n_rows":           self.n_rows,
            "T":                self.T,
            "hidden_dims":      self.hidden_dims,
            "is_fitted":        self._is_fitted,
        }, path)
        print(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "TabDDPM":
        """Loads a model previously saved with save()."""
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            n_features    = ckpt["n_features"],
            column_types  = ckpt["column_types"],
            feature_names = ckpt["feature_names"],
            n_rows        = ckpt.get("n_rows", 0),
            T             = ckpt["T"],
            hidden_dims   = ckpt["hidden_dims"],
        )
        model.model.load_state_dict(ckpt["model_state_dict"])
        model._is_fitted = ckpt["is_fitted"]
        print(f"Model loaded from: {path}")
        return model

    # ── Utilities ────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Prints a summary of the model architecture and training state."""
        n_params    = sum(p.numel() for p in self.model.parameters())
        type_counts = Counter(self.column_types.values())

        print("=" * 50)
        print("TabDDPM — Model Summary")
        print("=" * 50)
        print(f"  n_features   : {self.n_features}")
        print(f"  n_rows       : {self.n_rows}")
        print(f"  hidden_dims  : {self.hidden_dims}")
        print(f"  timesteps T  : {self.T}")
        print(f"  device       : {self.device}")
        print(f"  parameters   : {n_params:,}")
        print(f"  trained      : {self._is_fitted}")
        print()
        for col_type, count in sorted(type_counts.items()):
            print(f"  {col_type:<15}: {count} features")
        print("=" * 50)

    def _print_training_header(
        self, n_epochs: int, batch_size: int, lr: float, patience: int
    ) -> None:
        n_params = sum(p.numel() for p in self.model.parameters())
        es_info  = f"patience={patience}" if patience > 0 else "disabled"
        print("Starting TabDDPM training")
        print(f"  device         : {self.device}")
        print(f"  n_features     : {self.n_features}")
        print(f"  hidden_dims    : {self.hidden_dims}")
        print(f"  timesteps T    : {self.T}")
        print(f"  max epochs     : {n_epochs}")
        print(f"  batch_size     : {batch_size}")
        print(f"  lr             : {lr}")
        print(f"  parameters     : {n_params:,}")
        print(f"  early stopping : {es_info}")
        print()