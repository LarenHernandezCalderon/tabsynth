"""
step3_zero_inflated.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 — Step 3: Zero-inflated column split.

Splits zero-inflated columns into two simpler features:
    <col>_flag   : 1 if original value was > 0, else 0  (binary)
    <col>_value  : the numeric value (0 when flag=0)     (continuous)

This separates two distinct learning problems:
    1. How often does a non-zero value occur?     (flag)
    2. What is the non-zero value when it occurs? (value)

Inversion: col_original = flag * value
"""

import pandas as pd
from analyzer.dataset_analyzer import DatasetProfile


class ZeroInflatedSplitter:
    """
    Splits zero-inflated columns into _flag + _value pairs.

    Usage:
        splitter = ZeroInflatedSplitter(profile, verbose=True)
        df_out   = splitter.fit_transform(df)
        # splitter.zero_inflated_map → {col_flag/col_value: original_col}
    """

    def __init__(self, profile: DatasetProfile, verbose: bool = True):
        self.profile          = profile
        self.verbose          = verbose
        self.zero_inflated_map: dict[str, str] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.verbose:
            print("\nStep 3 — Zero-inflated split")
            print("-" * 40)

        zero_inflated_cols = [
            col for col in df.columns
            if self.profile.get_column(col) is not None
            and self.profile.get_column(col).distribution_type == "zero_inflated"
        ]

        if not zero_inflated_cols:
            if self.verbose:
                print("  No zero-inflated columns — step skipped")
            return df

        for col in zero_inflated_cols:
            series     = df[col]
            zero_ratio = (series == 0).sum() / len(series)
            non_zero   = int((series != 0).sum())

            flag_col  = f"{col}_flag"
            value_col = f"{col}_value"

            df[flag_col]  = (series != 0).astype(int)
            df[value_col] = series.copy()

            self.zero_inflated_map[flag_col]  = col
            self.zero_inflated_map[value_col] = col

            df.drop(columns=[col], inplace=True)

            if self.verbose:
                print(f"  {col:<20} → {flag_col} + {value_col}")
                print(f"    zeros: {zero_ratio:.1%}  |  non-zero rows: {non_zero}")

        return df