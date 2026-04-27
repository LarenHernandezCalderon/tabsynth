"""
step5_categorical_encoder.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 — Step 5: Categorical encoding.

Converts string/object columns to integers via LabelEncoder so they can
enter the PyTorch tensor. Numeric columns are left unchanged.

The full LabelEncoder is saved per column so categories can be decoded
back to their original string values after generation.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class CategoricalEncoder:
    """
    Encodes string columns to integers via LabelEncoder.

    Usage:
        encoder = CategoricalEncoder(verbose=True)
        df_out  = encoder.fit_transform(df)
        # encoder.label_encoders → {col: LabelEncoder}
    """

    def __init__(self, verbose: bool = True):
        self.verbose        = verbose
        self.label_encoders : dict[str, LabelEncoder] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.verbose:
            print("\nStep 5 — Categorical encoding")
            print("-" * 40)

        any_encoded = False

        for col in df.columns:
            series = df[col]

            # Only string/object columns
            if series.dtype != object and not pd.api.types.is_string_dtype(series):
                continue

            le           = LabelEncoder()
            series_clean = series.fillna("UNKNOWN").astype(str)
            df[col]      = le.fit_transform(series_clean)
            self.label_encoders[col] = le
            any_encoded  = True

            if self.verbose:
                classes_preview = list(le.classes_[:4])
                suffix = "..." if len(le.classes_) > 4 else ""
                print(f"  {col:<25} → {len(le.classes_)} classes  {classes_preview}{suffix}")

        if self.verbose and not any_encoded:
            print("  No string columns to encode — step skipped")

        return df