"""
type_enforcer.py
─────────────────────────────────────────────────────────────────────────────
Phase 4 — Type Enforcer.

Receives the raw synthetic DataFrame produced by Preprocessor.inverse_transform()
and enforces correct types and valid ranges on every column, based on the
original DatasetProfile from Phase 1.

Problems it solves:
    - Binary columns (Survived) come out as 0.004618, 0.901133 → must be 0 or 1
    - Ordinal integer columns (SibSp, Parch) come out as 1.446172 → must be int
    - Continuous columns (Age, Fare) may drift outside the real-world range
    - Ordinal columns (Pclass) must stay within the known set {1, 2, 3}

Strategy per column type:
    binary (numeric 0/1)    → round → clip(0,1) → int
    binary (string)         → already decoded by LabelEncoder — no action
    ordinal (integer)       → round → clip(min_real, max_real) → int
    ordinal (categorical)   → already decoded — no action
    numerical_continuous    → clip(min_real - k*std, max_real + k*std)
                              k=0.5 by default — slight extrapolation allowed
    categorical             → already decoded by LabelEncoder — no action

The TypeEnforcer is purely a post-processing step — it does NOT touch the
model or the preprocessor. It operates only on the final DataFrame.

Dependencies:
    pandas, numpy

Part of:
    Synthetic Data Engine — Auto-Adaptive Diffusion System
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from analyzer.dataset_analyzer import DatasetProfile


# ── Constants ─────────────────────────────────────────────────────────────────

# How many std deviations beyond real min/max we allow for continuous columns.
# 0.5 → allows slight extrapolation while preventing wild outliers.
CONTINUOUS_EXTRAPOLATION_K = 0.5


# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass
class EnforcementReport:
    """
    Records what the TypeEnforcer did to each column.
    Surfaced in the GeneratorOutput quality report.
    """
    binary_cols_rounded   : list[str] = field(default_factory=list)
    ordinal_cols_cast     : list[str] = field(default_factory=list)
    continuous_cols_clipped: list[str] = field(default_factory=list)
    n_values_changed      : dict[str, int] = field(default_factory=dict)


# ── Main enforcer ─────────────────────────────────────────────────────────────

class TypeEnforcer:
    """
    Enforces correct types and ranges on the raw synthetic DataFrame.

    Usage:
        enforcer = TypeEnforcer(profile, original_df)
        df_clean, report = enforcer.enforce(df_raw)
    """

    def __init__(
        self,
        profile     : DatasetProfile,
        original_df : pd.DataFrame,
    ):
        self.profile     = profile
        self.original_df = original_df

    def enforce(self, df: pd.DataFrame) -> tuple[pd.DataFrame, EnforcementReport]:
        """
        Apply type enforcement to every column in df.

        Returns:
            df_clean : corrected DataFrame
            report   : EnforcementReport with what was changed
        """
        df     = df.copy()
        report = EnforcementReport()

        for col in df.columns:
            col_profile = self.profile.get_column(col)
            if col_profile is None:
                continue  # column not in profile — skip safely

            dtype = col_profile.detected_type

            if dtype == "binary":
                df, report = self._enforce_binary(df, col, col_profile, report)

            elif dtype == "ordinal":
                df, report = self._enforce_ordinal(df, col, col_profile, report)

            elif dtype == "numerical_continuous":
                df, report = self._enforce_continuous(df, col, col_profile, report)

            # categorical / unknown_string → LabelEncoder already handled them
            # No numeric enforcement needed

        return df, report

    # ── Binary ────────────────────────────────────────────────────────────────

    def _enforce_binary(self, df, col, col_profile, report):
        """
        Binary numeric columns (0/1): round to nearest int, clip to {0, 1}.
        Binary string columns: already decoded correctly by LabelEncoder.
        """
        series = df[col]

        # Only act on numeric columns — strings are already correct
        if not pd.api.types.is_numeric_dtype(series):
            return df, report

        before   = series.copy()
        enforced = series.round().clip(0, 1).astype(int)
        changed  = int((before - enforced).abs().gt(0).sum())

        df[col] = enforced
        report.binary_cols_rounded.append(col)
        if changed:
            report.n_values_changed[col] = changed

        return df, report

    # ── Ordinal ───────────────────────────────────────────────────────────────

    def _enforce_ordinal(self, df, col, col_profile, report):
        """
        Ordinal integer columns: snap to the nearest value in the real discrete set.

        Improvement over simple round+clip: instead of rounding to any integer in
        [min, max], we snap to the exact set of values that appear in the real data.

        Example for SibSp: real values are {0, 1, 2, 3, 4, 5, 8} (note: no 6 or 7).
        A model-generated value of 6.3 would round to 6 with simple rounding,
        but 6 never appears in the real data. Snap maps it to the closest real value
        which is 5 — statistically correct.

        Fallback: if fewer than 2 unique real values, use simple round+clip.
        """
        series = df[col]

        # String ordinals already decoded — nothing to do
        if not pd.api.types.is_numeric_dtype(series):
            return df, report

        if col not in self.original_df.columns:
            return df, report

        real_series = self.original_df[col].dropna()
        if real_series.empty:
            return df, report

        real_values = sorted(real_series.unique().astype(int).tolist())

        before = series.copy()

        if len(real_values) >= 2:
            # Snap to nearest value in the real discrete set
            real_arr = np.array(real_values)
            def snap_to_real(x):
                idx = np.argmin(np.abs(real_arr - x))
                return int(real_arr[idx])
            enforced = series.apply(snap_to_real)
        else:
            # Fallback: simple round + clip
            real_min = int(real_series.min())
            real_max = int(real_series.max())
            enforced = series.round().clip(real_min, real_max).astype(int)

        changed = int((before - enforced).abs().gt(0).sum())

        df[col] = enforced
        report.ordinal_cols_cast.append(col)
        if changed:
            report.n_values_changed[col] = changed

        return df, report

    # ── Continuous ────────────────────────────────────────────────────────────

    def _enforce_continuous(self, df, col, col_profile, report):
        """
        Continuous columns: soft clip.
        Allows slight extrapolation (k * std beyond real min/max) to preserve
        the tails of the distribution, but prevents wild values.
        """
        series = df[col]

        if not pd.api.types.is_numeric_dtype(series):
            return df, report

        if col not in self.original_df.columns:
            return df, report

        real_series = self.original_df[col].dropna()
        if real_series.empty:
            return df, report

        real_min = float(real_series.min())
        real_max = float(real_series.max())
        real_std = float(real_series.std())

        k        = CONTINUOUS_EXTRAPOLATION_K
        lo       = real_min - k * real_std
        hi       = real_max + k * real_std

        # For columns that are naturally non-negative (Age, Fare...), floor at 0
        if real_min >= 0:
            lo = max(lo, 0.0)

        before   = series.copy()
        enforced = series.clip(lo, hi)
        changed  = int((before - enforced).abs().gt(1e-9).sum())

        df[col] = enforced
        report.continuous_cols_clipped.append(col)
        if changed:
            report.n_values_changed[col] = changed

        return df, report