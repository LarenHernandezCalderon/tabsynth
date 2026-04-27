"""
step2_missing_handler.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 — Step 2: Missing value handling (Strategy B).

For every column with at least one null:
    1. Creates a binary indicator <col>_missing (1=was null, 0=had value)
    2. Imputes the missing value (median for numeric, mode for categorical)

The model learns the missingness pattern via the indicator column,
capturing MNAR/MAR/MCAR structure rather than just the ratio.
"""

import pandas as pd


class MissingHandler:
    """
    Creates _missing indicators and imputes null values.

    Usage:
        handler = MissingHandler(verbose=True)
        df_out  = handler.fit_transform(df)
        # handler.imputation_values → {col: value}
    """

    def __init__(self, verbose: bool = True):
        self.verbose           = verbose
        self.imputation_values : dict[str, float] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.verbose:
            print("\nStep 2 — Missing indicators (Strategy B)")
            print("-" * 40)

        cols_with_missing = [c for c in df.columns if df[c].isna().any()]

        if not cols_with_missing:
            if self.verbose:
                print("  No missing values detected — step skipped")
            return df

        for col in cols_with_missing:
            missing_count = int(df[col].isna().sum())
            missing_ratio = missing_count / len(df)

            # Create binary indicator
            flag_col    = f"{col}_missing"
            df[flag_col] = df[col].isna().astype(int)

            # Compute imputation value
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                impute_val = float(df[col].median())
                strategy   = "median"
            else:
                mode_vals  = df[col].mode()
                impute_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "UNKNOWN"
                strategy   = "mode"

            df[col] = df[col].fillna(impute_val)
            self.imputation_values[col] = impute_val

            if self.verbose:
                print(f"  {col:<25} → {flag_col} created  "
                      f"({missing_count} nulls, {missing_ratio:.1%})  "
                      f"imputed with {strategy}={impute_val}")

        return df