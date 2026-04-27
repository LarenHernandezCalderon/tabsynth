"""
inverter.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 — Inverse transform.

Reconstructs a synthetic DataFrame from the tensor produced by the
diffusion model, reversing every preprocessing step in correct order:

    1. Tensor → numpy → DataFrame with feature_names
    2. Inverse MinMax scaling
    3. Decode categorical columns (LabelEncoder inverse)
    4. Invert numeric transforms (log → expm1, sqrt → x², quantile → inv)
    5. Reconstruct zero-inflated columns (flag * value)
    6. Restore NaNs using _missing indicator columns
    7. Reconstruct redundant columns (exact value mapping)
    8. Reorder columns to match the original dataset order
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from preprocessor.contracts import PreprocessorOutput


class InverseTransformer:
    """
    Inverts all preprocessing steps to recover a DataFrame from a tensor.

    Usage:
        inverter = InverseTransformer(output, original_df, redundant_map, redundant_mappers)
        df_raw   = inverter.inverse_transform(synthetic_tensor)
    """

    def __init__(
        self,
        output             : PreprocessorOutput,
        original_df        : pd.DataFrame,
        redundant_map      : dict[str, str],
        redundant_mappers  : dict[str, dict],
    ):
        self.output            = output
        self.original_df       = original_df
        self.redundant_map     = redundant_map
        self.redundant_mappers = redundant_mappers

    def inverse_transform(self, tensor: torch.Tensor) -> pd.DataFrame:
        o = self.output

        # ── Step 1: Tensor → DataFrame ─────────────────────────────────────
        arr = tensor.detach().numpy()
        df  = pd.DataFrame(arr, columns=o.feature_names)

        # ── Step 2: Inverse MinMax ─────────────────────────────────────────
        for ct in o.transforms:
            col = ct.tensor_name
            if col in df.columns and ct.minmax_scaler is not None:
                df[col] = ct.minmax_scaler.inverse_transform(
                    df[col].values.reshape(-1, 1)
                ).flatten()

        # ── Step 3: Decode categoricals ────────────────────────────────────
        for ct in o.transforms:
            col = ct.tensor_name
            if col not in df.columns or ct.label_encoder is None:
                continue
            le   = ct.label_encoder
            ints = df[col].round().clip(0, len(le.classes_) - 1).astype(int)
            df[col] = le.inverse_transform(ints)

        # ── Step 4: Invert numeric transforms ─────────────────────────────
        for ct in o.transforms:
            col = ct.tensor_name
            if col not in df.columns:
                continue
            if ct.transform_type == "log":
                df[col] = np.expm1(df[col].clip(lower=-1))
            elif ct.transform_type == "sqrt":
                df[col] = np.power(df[col].clip(lower=0), 2)
            elif ct.transform_type == "quantile" and ct.quantile_transformer:
                df[col] = ct.quantile_transformer.inverse_transform(
                    df[col].values.reshape(-1, 1)
                ).flatten()

        # ── Step 5: Reconstruct zero-inflated columns ──────────────────────
        for flag_col, orig_col in o.zero_inflated_map.items():
            if not flag_col.endswith("_flag"):
                continue
            value_col = flag_col.replace("_flag", "_value")
            if flag_col in df.columns and value_col in df.columns:
                flag = (df[flag_col] > 0.5).astype(int)
                df[orig_col] = flag * df[value_col]
                df.drop(columns=[flag_col, value_col], inplace=True)

        # ── Step 6: Restore NaNs from _missing indicators ─────────────────
        for col in list(df.columns):
            if col.endswith("_missing"):
                base_col = col[:-8]
                if base_col in df.columns:
                    mask = df[col] > 0.5
                    df.loc[mask, base_col] = np.nan
                df.drop(columns=[col], inplace=True)

        # ── Step 7: Reconstruct redundant columns ──────────────────────────
        for removed_col, source_col in self.redundant_map.items():
            if source_col in df.columns:
                mapper = self.redundant_mappers.get(removed_col, {})
                if mapper:
                    df[removed_col] = df[source_col].map(mapper)
                    if df[removed_col].isna().any():
                        keys = sorted(mapper.keys())
                        for idx in df[df[removed_col].isna()].index:
                            val     = df.loc[idx, source_col]
                            closest = min(keys, key=lambda k: abs(k - val))
                            df.loc[idx, removed_col] = mapper[closest]

        # ── Step 8: Reorder to original column order ───────────────────────
        original_cols = [c for c in self.original_df.columns if c in df.columns]
        df = df[original_cols]

        return df