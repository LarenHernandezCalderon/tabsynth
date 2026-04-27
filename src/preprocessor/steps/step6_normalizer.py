"""
step6_normalizer.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 — Step 6: MinMax normalisation.

Scales all numeric columns to [0, 1].
The diffusion model operates in [0, 1]^d space — without normalisation,
columns with large ranges would dominate the gradient.

MinMaxScaler parameters are saved per column for inversion:
    x_original = x_scaled * (max - min) + min
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class MinMaxNormalizer:
    """
    Scales all numeric columns to [0, 1].

    Usage:
        normalizer = MinMaxNormalizer(verbose=True)
        df_out     = normalizer.fit_transform(df)
        # normalizer.minmax_scalers → {col: MinMaxScaler}
    """

    def __init__(self, verbose: bool = True):
        self.verbose        = verbose
        self.minmax_scalers : dict[str, MinMaxScaler] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.verbose:
            print("\nStep 6 — MinMax normalisation [0, 1]")
            print("-" * 40)

        for col in df.columns:
            series = df[col]

            if not pd.api.types.is_numeric_dtype(series):
                continue

            scaler  = MinMaxScaler(feature_range=(0, 1))
            df[col] = scaler.fit_transform(
                series.values.reshape(-1, 1)
            ).flatten()
            self.minmax_scalers[col] = scaler

        if self.verbose:
            print(f"  {len(self.minmax_scalers)} columns scaled to [0, 1]")

        return df