"""
null_validator.py
─────────────────────────────────────────────────────────────────────────────
Phase 4 — Null Validator.

Validates and corrects NaN values in the synthetic DataFrame after
TypeEnforcer and ColumnBuilder have run.

The diffusion model learns the missingness pattern from the training data
via the <col>_missing indicator columns (created in Preprocessor Step 2).
The inverse_transform uses those indicators to restore NaNs in the correct
positions. That is the correct behaviour — this validator does NOT undo it.

What this module does:
    1. ALLOW nulls that are statistically expected
       If original missing_ratio > 0 → NaNs in the synthetic column are valid.
       The model learned where to put them. Keep them.

    2. FIX unexpected nulls
       If original missing_ratio == 0 → no NaN should appear in this column.
       Any NaN that slipped through is an artefact of numerical precision in
       the inverse transforms. Impute with median (numeric) or mode (string).

    3. REPORT null rates
       Compare real missing_ratio vs synthetic null_rate per column.
       Surface large discrepancies (> 10 pp) as warnings in the quality report.

    4. CAST dtypes
       After null resolution, cast each column to its original dtype where
       possible, respecting the presence of remaining NaN values
       (int columns with NaN must stay as float or use pd.Int64Dtype()).

Dependencies:
    pandas, numpy

Part of:
    Synthetic Data Engine — Auto-Adaptive Diffusion System
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from analyzer.dataset_analyzer import DatasetProfile


# ── Thresholds ────────────────────────────────────────────────────────────────

# If synthetic null rate deviates from real by more than this, emit a warning
NULL_RATE_WARNING_THRESHOLD = 0.10  # 10 percentage points

# If synth null rate exceeds real by this multiplier, clamp excess nulls back.
# Example: real=20%, synth=35% → ratio=1.75 > 1.5 → re-impute excess nulls.
# This corrects over-flagging by the Age_missing binary feature in the model
# without touching nulls that are within a normal variance band.
NULL_RATE_MAX_RATIO = 1.5


# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass
class NullReport:
    """Records null validation outcomes per column."""
    unexpected_nulls_fixed  : dict[str, int]   = field(default_factory=dict)  # col → n fixed
    null_rate_real          : dict[str, float]  = field(default_factory=dict)  # col → real ratio
    null_rate_synthetic     : dict[str, float]  = field(default_factory=dict)  # col → synth ratio
    null_rate_warnings      : list[str]         = field(default_factory=list)  # cols with big gap
    dtype_casts             : dict[str, str]    = field(default_factory=dict)  # col → dtype applied


# ── Main validator ────────────────────────────────────────────────────────────

class NullValidator:
    """
    Validates NaN values in the synthetic DataFrame and casts dtypes.

    Usage:
        validator = NullValidator(profile, original_df)
        df_validated, report = validator.validate(df)
    """

    def __init__(
        self,
        profile     : DatasetProfile,
        original_df : pd.DataFrame,
    ):
        self.profile     = profile
        self.original_df = original_df

    def validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, NullReport]:
        """
        Validate nulls and cast dtypes for all columns in df.

        Returns:
            df_validated : cleaned DataFrame
            report       : NullReport with per-column outcomes
        """
        df     = df.copy()
        report = NullReport()

        for col in df.columns:
            col_profile = self.profile.get_column(col)

            # Real missing ratio — 0 if column not in profile (e.g. sequential IDs)
            if col_profile is not None:
                real_ratio = col_profile.missing_ratio
            else:
                real_ratio = 0.0

            n          = len(df)
            n_nulls    = int(df[col].isna().sum())
            synth_rate = n_nulls / n if n > 0 else 0.0

            report.null_rate_real[col]      = real_ratio
            report.null_rate_synthetic[col] = synth_rate

            # ── Fix unexpected nulls ───────────────────────────────────────
            if real_ratio == 0.0 and n_nulls > 0:
                fixed = self._impute_column(df, col)
                report.unexpected_nulls_fixed[col] = fixed

            # ── Clamp excess nulls (over-flagging correction) ──────────────
            # If the model generates significantly more nulls than the real
            # ratio (e.g. Age: real=20%, synth=35%), re-impute the excess.
            # Only applies when real_ratio > 0 (expected-null columns).
            elif real_ratio > 0.0 and synth_rate > real_ratio * NULL_RATE_MAX_RATIO:
                target_n_nulls = int(round(real_ratio * n))
                excess         = n_nulls - target_n_nulls
                if excess > 0:
                    null_idx = df[col].isna()
                    # Randomly keep only target_n_nulls of the nulls
                    null_positions = df.index[null_idx].tolist()
                    rng            = np.random.default_rng(seed=42)
                    to_fill        = rng.choice(null_positions, size=excess, replace=False)
                    if pd.api.types.is_numeric_dtype(df[col]):
                        if col in self.original_df.columns:
                            fill_val = float(self.original_df[col].median())
                        else:
                            fill_val = float(df[col].median())
                    else:
                        mode_vals = self.original_df[col].mode() if col in self.original_df.columns else df[col].mode()
                        fill_val  = mode_vals.iloc[0] if len(mode_vals) > 0 else "UNKNOWN"
                    df.loc[to_fill, col] = fill_val
                    report.unexpected_nulls_fixed[col] = excess

            # ── Warn on large discrepancy ──────────────────────────────────
            gap = abs(df[col].isna().mean() - real_ratio)  # recalc after possible fix
            if gap > NULL_RATE_WARNING_THRESHOLD:
                report.null_rate_warnings.append(col)

        # ── Cast dtypes ────────────────────────────────────────────────────
        df, report = self._cast_dtypes(df, report)

        return df, report

    # ── Imputation ────────────────────────────────────────────────────────────

    def _impute_column(self, df: pd.DataFrame, col: str) -> int:
        """
        Impute unexpected NaN values with median (numeric) or mode (string).
        Returns the number of values imputed.
        """
        series  = df[col]
        n_nulls = int(series.isna().sum())

        if n_nulls == 0:
            return 0

        if pd.api.types.is_numeric_dtype(series):
            # Use real column median if available for better fidelity
            if col in self.original_df.columns:
                fill_val = float(self.original_df[col].median())
            else:
                fill_val = float(series.median())
            df[col] = series.fillna(fill_val)

        else:
            if col in self.original_df.columns:
                mode_vals = self.original_df[col].mode()
            else:
                mode_vals = series.mode()
            fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "UNKNOWN"
            df[col] = series.fillna(fill_val)

        return n_nulls

    # ── Dtype casting ─────────────────────────────────────────────────────────

    def _cast_dtypes(
        self,
        df     : pd.DataFrame,
        report : NullReport,
    ) -> tuple[pd.DataFrame, NullReport]:
        """
        Cast every column to its original dtype where possible.

        Rules:
            - int64 with no NaN  → int64
            - int64 with NaN     → pd.Int64Dtype() (nullable integer)
            - float64            → float64
            - object (string)    → object
            - bool               → bool
        """
        for col in df.columns:
            if col not in self.original_df.columns:
                continue

            target_dtype = self.original_df[col].dtype
            has_nulls    = df[col].isna().any()

            try:
                if pd.api.types.is_integer_dtype(target_dtype):
                    if has_nulls:
                        # Nullable integer — preserves NaN without converting to float
                        df[col] = df[col].astype(pd.Int64Dtype())
                        report.dtype_casts[col] = "Int64 (nullable)"
                    else:
                        df[col] = df[col].round().astype("int64")
                        report.dtype_casts[col] = "int64"

                elif pd.api.types.is_float_dtype(target_dtype):
                    df[col] = df[col].astype("float64")
                    report.dtype_casts[col] = "float64"

                elif pd.api.types.is_bool_dtype(target_dtype):
                    df[col] = df[col].astype(bool)
                    report.dtype_casts[col] = "bool"

                # object / string → no cast needed

            except (ValueError, TypeError):
                # If casting fails, leave as-is — better than crashing
                pass

        return df, report