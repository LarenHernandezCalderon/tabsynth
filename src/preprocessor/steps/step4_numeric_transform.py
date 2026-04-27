"""
step4_numeric_transform.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 — Step 4: Numeric transforms.

Applies the transform recommended by the DistributionProfiler (Phase 1)
to each numeric column, making distributions more Gaussian before the model.

    log      → np.log1p(x)           inverse: np.expm1(x)
    sqrt     → np.sqrt(x)            inverse: x²
    quantile → QuantileTransformer   inverse: qt.inverse_transform(x)
    none     → no change

_missing and _flag columns are always skipped (they are binary).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from analyzer.dataset_analyzer import DatasetProfile


class NumericTransformer:
    """
    Applies distribution-normalising transforms to numeric columns.

    Usage:
        transformer = NumericTransformer(profile, zero_inflated_map, verbose=True)
        df_out      = transformer.fit_transform(df)
        # transformer.quantile_transformers → {col: QuantileTransformer}
    """

    def __init__(
        self,
        profile           : DatasetProfile,
        zero_inflated_map : dict[str, str],
        verbose           : bool = True,
    ):
        self.profile              = profile
        self.zero_inflated_map    = zero_inflated_map
        self.verbose              = verbose
        self.quantile_transformers: dict[str, QuantileTransformer] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.verbose:
            print("\nStep 4 — Numeric transforms")
            print("-" * 40)

        for col in list(df.columns):
            # Skip binary indicator columns
            if col.endswith("_missing") or col.endswith("_flag"):
                continue

            # Only numeric columns
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            # Resolve the base column name for profile lookup
            base_col = self._resolve_base(col)
            col_profile = self.profile.get_column(base_col)
            if col_profile is None:
                continue

            transform = col_profile.recommended_transform or "none"
            series    = df[col].copy()

            if transform == "log":
                df[col] = np.log1p(series.clip(lower=0))
                if self.verbose:
                    print(f"  {col:<30} → log1p")

            elif transform == "sqrt":
                df[col] = np.sqrt(series.clip(lower=0))
                if self.verbose:
                    print(f"  {col:<30} → sqrt")

            elif transform == "quantile":
                qt = QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=min(1000, len(series)),
                    random_state=42,
                )
                df[col] = qt.fit_transform(
                    series.values.reshape(-1, 1)
                ).flatten()
                self.quantile_transformers[col] = qt
                if self.verbose:
                    print(f"  {col:<30} → quantile (normal output)")

        return df

    def _resolve_base(self, col: str) -> str:
        """Resolve _flag/_value/_missing suffixes to original column name."""
        base = self.zero_inflated_map.get(col, col)
        for suffix in ("_missing", "_flag", "_value"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        return base