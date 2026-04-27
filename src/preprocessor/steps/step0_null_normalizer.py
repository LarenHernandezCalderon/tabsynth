"""
step0_null_normalizer.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 — Step 0: Masked null normalisation.

Replicates the MetadataExtractor (Phase 1) sentinel fix on the real DataFrame.
Replaces known sentinel values ("?", "N/A", "none", etc.) with real NaN
so all downstream steps see a consistent null representation.
"""

import pandas as pd
from preprocessor.contracts import MASKED_NULL_VALUES


class NullNormalizer:
    """
    Replaces sentinel null values with NaN across all string columns.

    Usage:
        normalizer = NullNormalizer(verbose=True)
        df_clean   = normalizer.fit_transform(df)
    """

    def __init__(self, verbose: bool = True):
        self.verbose        = verbose
        self.replaced_cols  : list[tuple[str, int]] = []
        self.total_replaced : int = 0

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.verbose:
            print("\nStep 0 — Masked null normalisation")
            print("-" * 40)

        for col in df.columns:
            series = df[col]
            if series.dtype != object and not pd.api.types.is_string_dtype(series):
                continue

            mask  = series.isin(MASKED_NULL_VALUES)
            count = int(mask.sum())
            if count > 0:
                df[col] = series.where(~mask, other=float("nan"))
                self.total_replaced += count
                self.replaced_cols.append((col, count))

        if self.verbose:
            if self.total_replaced == 0:
                print("  No sentinel values detected — step skipped")
            else:
                for col, count in self.replaced_cols:
                    ratio = count / len(df)
                    print(f"  {col:<25} → {count} sentinels replaced ({ratio:.1%})")
                print(f"  Total replaced: {self.total_replaced}")

        return df