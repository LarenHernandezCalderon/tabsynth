"""
calibrator.py
─────────────────────────────────────────────────────────────────────────────
Post-hoc rank-based quantile matching.

After the diffusion model generates synthetic data and the preprocessor's
inverse_transform reconstructs the DataFrame, this module rank-matches the
marginal distribution of every continuous column to the corresponding real
column. Discrete columns (ordinal, binary, low-cardinality) are skipped —
their grid is preserved exactly.

Why this matters for Fidelity:
    Even with log1p / sqrt / QuantileTransformer in Phase 2, the diffusion
    model never reproduces marginals perfectly, especially in long tails
    (e.g. Fare in Titanic, price in Diamonds). Rank matching is a no-cost
    post-processing step that aligns the per-column histogram to the real
    one without re-shuffling rows — column ranks are preserved, so the
    Spearman correlation structure is unchanged and Pearson shifts only
    slightly.

What it does NOT do:
    - It does NOT re-order rows: row i stays as row i.
    - It does NOT fabricate values absent from the real dataset.
    - It does NOT change the joint dependency structure beyond what is
      implied by per-column rank-preserving substitution.

Privacy note:
    Calibration redistributes already-generated synthetic rows along the
    real ECDF. It does NOT copy real records. Re-identification risk is
    governed by the joint structure (which is unchanged) and is measured
    independently by the Privacy / Distance Ratio metric.

Dependencies:
    pandas, numpy, scipy

Part of:
    Synthetic Tabular Data Engine — generator/ (post-hoc step in Phase 4)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional


# Columns with fewer than this many unique real values are treated as discrete
# (ordinal / encoded categorical) and skipped — their grid must be preserved.
MIN_UNIQUE_FOR_CALIBRATION = 20

# Soft clip on the rank percentile before mapping to the real ECDF.
# Prevents the synthetic extremes from always landing on the absolute min/max
# of the real distribution, which would create unnatural spikes at the tails.
SAFE_QUANTILE_CLIP = (0.001, 0.999)


class Calibrator:
    """
    Rank-based quantile matcher. Aligns synthetic marginal distributions
    with the real ones, column by column, on continuous columns only.

    Usage:
        zero_inflated = set(preprocessor_output.zero_inflated_map.values())
        cal = Calibrator(real_df=df, zero_inflated=zero_inflated)
        synthetic_df = cal.transform(synthetic_df)
        cal.report()        # optional human-readable summary

    Args:
        real_df       : the original DataFrame, before any preprocessing
        zero_inflated : set of original column names that are zero-inflated.
                        For these, only the non-zero subset is calibrated and
                        zeros are preserved exactly. Pull this from
                        preprocessor_output.zero_inflated_map.values().
        verbose       : print per-column calibration log to stdout
    """

    def __init__(
        self,
        real_df: pd.DataFrame,
        zero_inflated: Optional[set[str]] = None,
        verbose: bool = True,
    ):
        self.real_df       = real_df
        self.zero_inflated = zero_inflated or set()
        self.verbose       = verbose

        self._calibrated: list[str]              = []
        self._skipped:    list[tuple[str, str]]  = []

    # ── Public ────────────────────────────────────────────────────────────────

    def transform(self, synth_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a copy of synth_df with continuous columns rank-matched
        to the real distribution. Index and row order are preserved.
        """
        out = synth_df.copy()
        self._calibrated.clear()
        self._skipped.clear()

        if self.verbose:
            print("\nPost-hoc quantile calibration")
            print("-" * 40)

        for col in out.columns:
            reason = self._should_skip(col, out)
            if reason is not None:
                self._skipped.append((col, reason))
                continue

            real_vals = self.real_df[col].dropna()

            if col in self.zero_inflated:
                out[col] = self._calibrate_zero_inflated(out[col], real_vals)
            else:
                out[col] = self._calibrate(out[col], real_vals)

            self._calibrated.append(col)
            if self.verbose:
                print(f"  ✓ {col}")

        if self.verbose:
            self.report()

        return out

    def report(self) -> None:
        """Prints a summary of which columns were calibrated and which skipped."""
        print(f"\nCalibrated : {len(self._calibrated)} columns")
        for col, reason in self._skipped:
            print(f"  · {col:<25} skipped — {reason}")
        print(f"Skipped    : {len(self._skipped)} columns")

    # ── Decision logic ────────────────────────────────────────────────────────

    def _should_skip(self, col: str, synth_df: pd.DataFrame) -> Optional[str]:
        """Returns a skip reason string, or None if the column should be calibrated."""
        if col not in self.real_df.columns:
            return "not in real_df"

        if not pd.api.types.is_numeric_dtype(synth_df[col]):
            return "non-numeric (categorical / string)"
        if not pd.api.types.is_numeric_dtype(self.real_df[col]):
            return "real column is non-numeric"

        real_non_null = self.real_df[col].dropna()
        if len(real_non_null) == 0:
            return "real column is fully null"

        unique_count = real_non_null.nunique()
        if unique_count < MIN_UNIQUE_FOR_CALIBRATION:
            return f"discrete ({unique_count} unique)"

        return None  # calibrate

    # ── Core calibration ──────────────────────────────────────────────────────

    @staticmethod
    def _calibrate(
        synth_col: pd.Series,
        real_vals: pd.Series,
    ) -> pd.Series:
        """
        Rank-based quantile match. Preserves NaN positions in synthetic.

        Algorithm:
            1. Rank each synthetic non-null value among the synthetic non-nulls
               (average-rank ties handling), normalise to (0, 1).
            2. Soft-clip the rank to avoid mapping every extreme synthetic
               value onto the absolute min/max of the real distribution.
            3. Look up the corresponding quantile in the sorted real values.
        """
        result = synth_col.copy()
        notna  = synth_col.notna()
        if notna.sum() == 0:
            return result

        synth_vals  = synth_col[notna].values
        real_sorted = np.sort(real_vals.values)

        ranks = stats.rankdata(synth_vals, method="average") / (len(synth_vals) + 1)
        ranks = np.clip(ranks, *SAFE_QUANTILE_CLIP)

        idx = (ranks * len(real_sorted)).astype(int)
        idx = np.clip(idx, 0, len(real_sorted) - 1)

        result.loc[notna] = real_sorted[idx]
        return result

    @classmethod
    def _calibrate_zero_inflated(
        cls,
        synth_col: pd.Series,
        real_vals: pd.Series,
    ) -> pd.Series:
        """
        Calibrates only the non-zero subset, leaving zeros untouched.
        The diffusion model already decided how many rows are zero — that
        ratio is preserved. Only the values of the non-zero rows are
        rank-matched to the real non-zero distribution.
        """
        result = synth_col.copy()

        nz_mask = (synth_col != 0) & synth_col.notna()
        if nz_mask.sum() == 0:
            return result

        real_non_zero = real_vals[real_vals != 0]
        if len(real_non_zero) == 0:
            return result

        nz_synth   = synth_col[nz_mask].reset_index(drop=True)
        calibrated = cls._calibrate(nz_synth, real_non_zero)
        result.loc[nz_mask] = calibrated.values
        return result