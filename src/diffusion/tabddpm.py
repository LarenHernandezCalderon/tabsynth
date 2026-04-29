"""
tabddpm.py
─────────────────────────────────────────────────────────────────────────────
Main orchestrator of the TabDDPM diffusion model (Phase 3).

Changes vs previous version:
    - from_preprocessor() now accepts an optional `profile` (DatasetProfile)
    - When profile is provided, critical_pairs are converted to tensor indices
      and a CorrelationLoss is added to the training signal
    - fit() reconstructs x_0_pred per batch and adds corr_loss * corr_weight
      to the main MixedLoss — the model now has a direct incentive to preserve
      the joint distribution between strongly correlated columns

Public interface (unchanged):
    model = TabDDPM.from_preprocessor(preprocessor_output)              # no profile
    model = TabDDPM.from_preprocessor(preprocessor_output, profile=p)   # with profile

    model.auto_tune(output.tensor_dataset)
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
from .loss            import MixedLoss, CorrelationLoss
from .tuner           import HyperTuner, EarlyStopping


# Methods from Phase 1 that represent numeric pairwise correlations.
# Only these produce a signed Pearson-compatible score that makes sense
# to enforce in the diffusion model's continuous feature space.
_NUMERIC_METHODS = {"pearson", "spearman"}


def _build_corr_pairs(
    profile,
    feature_names: list[str],
    zero_inflated_map: dict[str, str],
) -> list[tuple[int, int, float]]:
    """
    Converts DatasetProfile.critical_pairs into (idx_a, idx_b, target_corr)
    tuples that reference tensor column indices.

    Rules:
        - Only numeric pairs (method == pearson or spearman) are included.
          Cramér's V and Eta-squared use different scales and the correlation
          signal would be misleading in the continuous feature space.
        - Columns excluded from the tensor (PII, IDs, free text) are skipped.
        - Zero-inflated columns use their _value feature for correlation.
        - Direction is used to recover the sign: score is always positive in
          CorrelationPair, but the model must know if the relation is inverse.
    """
    if profile is None or not profile.critical_pairs:
        return []

    # Build reverse map: original_col → _value feature name (zero-inflated)
    orig_to_value: dict[str, str] = {}
    for feat, orig in zero_inflated_map.items():
        if feat.endswith("_value"):
            orig_to_value[orig] = feat

    # Build name → index map for fast lookup
    name_to_idx: dict[str, int] = {name: i for i, name in enumerate(feature_names)}

    def resolve(col_name: str) -> int | None:
        if col_name in name_to_idx:
            return name_to_idx[col_name]
        # Zero-inflated: col not in tensor directly, use _value proxy
        if col_name in orig_to_value:
            proxy = orig_to_value[col_name]
            return name_to_idx.get(proxy)
        return None  # excluded from tensor

    pairs: list[tuple[int, int, float]] = []

    for cp in profile.critical_pairs:
        # Only numeric methods carry a meaningful signed correlation
        if cp.method not in _NUMERIC_METHODS:
            continue

        idx_a = resolve(cp.col_a)
        idx_b = resolve(cp.col_b)

        if idx_a is None or idx_b is None:
            continue  # at least one column not in the tensor

        # Recover the sign — CorrelationPair.score is always positive
        sign        = -1.0 if cp.direction == "negative" else 1.0
        target_corr = float(sign * cp.score)

        pairs.append((idx_a, idx_b, target_corr))

    return pairs


class TabDDPM:
    """
    Denoising Diffusion Probabilistic Model for mixed tabular data.

    Coordinates four internal components:
        NoiseScheduler  — mathematical noise calendar (forward + reverse)
        DenoisingMLP    — neural network that learns to predict and undo noise
        MixedLoss       — error function weighted by column type
        CorrelationLoss — penalizes joint correlation errors (when profile given)

    Args:
        n_features    : number of tensor features   (PreprocessorOutput.n_features)
        column_types  : dict feature → type         (PreprocessorOutput.column_types)
        feature_names : ordered list of names       (PreprocessorOutput.feature_names)
        n_rows        : number of dataset rows      (used by auto_tune)
        T             : number of diffusion steps    (default: 1000)
        schedule      : noise schedule type          ('cosine' or 'linear')
        hidden_dims   : MLP hidden layer sizes       (None = inferred from n_features)
        dropout       : MLP regularisation rate      (default: 0.0)
        corr_pairs    : list of (idx_a, idx_b, target_corr) from _build_corr_pairs
        corr_weight   : weight of CorrelationLoss relative to MixedLoss (default: 0.1)
    """

    def __init__(
        self,
        n_features:    int,
        column_types:  dict[str, str],
        feature_names: list[str],
        n_rows:        int                          = 0,
        T:             int                          = 1000,
        schedule:      str                          = "cosine",
        hidden_dims:   list[int] | None             = None,
        dropout:       float                        = 0.0,
        corr_pairs:    list[tuple[int,int,float]]   = None,
        corr_weight:   float                        = 0.1,
    ):
        self.n_features    = n_features
        self.column_types  = column_types
        self.feature_names = feature_names
        self.n_rows        = n_rows
        self.T             = T
        self.dropout       = dropout
        self.corr_pairs    = corr_pairs or []
        self.corr_weight   = corr_weight

        if hidden_dims is None:
            hidden_dims = auto_hidden_dims(n_features)
        self.hidden_dims = hidden_dims

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._build_components()
        self._is_fitted = False

    def _build_components(self) -> None:
        """Builds or rebuilds all internal components."""
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

        # CorrelationLoss — only active when critical pairs are provided
        if self.corr_pairs:
            self.corr_loss_fn = CorrelationLoss(self.corr_pairs).to(self.device)
        else:
            self.corr_loss_fn = None

    # ── Alternative constructor ──────────────────────────────────────────────

    @classmethod
    def from_preprocessor(
        cls,
        preprocessor_output,
        profile=None,          # DatasetProfile from Phase 1 (optional)
        corr_weight: float = 0.1,
        **kwargs,
    ) -> "TabDDPM":
        """
        Builds a TabDDPM instance directly from a Phase 2 PreprocessorOutput.

        Args:
            preprocessor_output : PreprocessorOutput from Phase 2
            profile             : DatasetProfile from Phase 1. When provided,
                                  critical_pairs are used to build CorrelationLoss.
            corr_weight         : weight of CorrelationLoss (default: 0.1)

        Usage:
            # Without correlation loss (original behaviour)
            model = TabDDPM.from_preprocessor(output)

            # With correlation loss — recommended
            model = TabDDPM.from_preprocessor(output, profile=profile)
        """
        corr_pairs = _build_corr_pairs(
            profile           = profile,
            feature_names     = preprocessor_output.feature_names,
            zero_inflated_map = preprocessor_output.zero_inflated_map,
        )

        if corr_pairs:
            print(f"  CorrelationLoss active: {len(corr_pairs)} critical pairs")
            for idx_a, idx_b, target in corr_pairs:
                name_a = preprocessor_output.feature_names[idx_a]
                name_b = preprocessor_output.feature_names[idx_b]
                print(f"    {name_a} ↔ {name_b}  target={target:+.3f}")
            print()

        return cls(
            n_features    = preprocessor_output.n_features,
            column_types  = preprocessor_output.column_types,
            feature_names = preprocessor_output.feature_names,
            n_rows        = len(preprocessor_output.tensor_dataset),
            corr_pairs    = corr_pairs,
            corr_weight   = corr_weight,
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
        patience:   int   = 0,
        min_delta:  float = 1e-4,
    ) -> list[float]:
        """
        Trains the diffusion model.

        Per batch:
            1. Sample random timestep t per sample
            2. Add noise:         x_t, noise = scheduler.add_noise(x, t)
            3. Predict noise:     noise_pred = model(x_t, t)
            4. Main loss:         mixed_loss(noise_pred, noise)
            5. Corr loss:         reconstruct x_0_pred, compute corr penalty
            6. Total loss:        mixed_loss + corr_weight * corr_loss
            7. Update weights
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

                # ── Main loss (noise prediction) ──────────────────────────────
                main_loss = self.loss_fn(noise_pred, noise)

                # ── Correlation loss (joint distribution) ─────────────────────
                if self.corr_loss_fn is not None:
                    ab       = self.scheduler.alpha_bar[t.cpu()].unsqueeze(1).to(self.device)
                    x_0_pred = (x_t - torch.sqrt(1.0 - ab) * noise_pred) / (torch.sqrt(ab) + 1e-8)
                    x_0_pred = torch.clamp(x_0_pred, 0.0, 1.0)
                    corr_loss = self.corr_loss_fn(x_0_pred)
                    loss = main_loss + self.corr_weight * corr_loss
                else:
                    loss = main_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            mean_loss = sum(epoch_losses) / len(epoch_losses)
            loss_history.append(mean_loss)

            if (epoch + 1) % log_every == 0 or epoch == 0:
                status = f" | {stopper.status()}" if stopper else ""
                print(f"  Epoch {epoch + 1:4d}/{n_epochs} | Loss: {mean_loss:.6f}{status}")

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
        Fully automatic training — no hyperparameters needed.

        Grid search phase uses MixedLoss only (fast architecture selection).
        Full training phase uses MixedLoss + CorrelationLoss if pairs exist.
        """
        tuner = HyperTuner(
            n_rows        = self.n_rows,
            n_features    = self.n_features,
            column_types  = self.column_types,
            feature_names = self.feature_names,
        )
        best = tuner.run(tensor_dataset, T=self.T, probe_epochs=probe_epochs)

        self.hidden_dims = best.hidden_dims
        self._build_components()

        print(f"Starting full training — winner: {best.label}")
        print(f"  Early stopping: patience={patience}, max_epochs={max_epochs}")
        if self.corr_loss_fn is not None:
            print(f"  CorrelationLoss: weight={self.corr_weight}, "
                  f"pairs={len(self.corr_pairs)}")
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
        if not self._is_fitted:
            raise RuntimeError(
                "Model has not been trained yet. Call fit() or auto_tune() first."
            )

        self.model.eval()

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
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "n_features":       self.n_features,
            "column_types":     self.column_types,
            "feature_names":    self.feature_names,
            "n_rows":           self.n_rows,
            "T":                self.T,
            "hidden_dims":      self.hidden_dims,
            "is_fitted":        self._is_fitted,
            "corr_pairs":       self.corr_pairs,
            "corr_weight":      self.corr_weight,
        }, path)
        print(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "TabDDPM":
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            n_features    = ckpt["n_features"],
            column_types  = ckpt["column_types"],
            feature_names = ckpt["feature_names"],
            n_rows        = ckpt.get("n_rows", 0),
            T             = ckpt["T"],
            hidden_dims   = ckpt["hidden_dims"],
            corr_pairs    = ckpt.get("corr_pairs", []),
            corr_weight   = ckpt.get("corr_weight", 0.1),
        )
        model.model.load_state_dict(ckpt["model_state_dict"])
        model._is_fitted = ckpt["is_fitted"]
        print(f"Model loaded from: {path}")
        return model

    # ── Utilities ────────────────────────────────────────────────────────────

    def summary(self) -> None:
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
        print(f"  corr_pairs   : {len(self.corr_pairs)}")
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
        if self.corr_loss_fn is not None:
            print(f"  corr_loss      : weight={self.corr_weight}, "
                  f"pairs={len(self.corr_pairs)}")
        print()