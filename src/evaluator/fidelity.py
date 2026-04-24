"""
fidelity.py
─────────────────────────────────────────────────────────────────────────────
Phase 5 — Fidelity Evaluator.

Measures how well the synthetic data reproduces the marginal distributions
of the real data, column by column.

Metrics:
    Numeric columns:
        KS Test (Kolmogorov-Smirnov)
            Compares the empirical CDFs of real vs synthetic.
            H0: both samples come from the same distribution.
            p-value > 0.05 → distributions are statistically indistinguishable.
            statistic      → maximum distance between the two CDFs (0=identical).

        Wasserstein Distance (Earth Mover's Distance)
            Measures how much "work" it takes to transform one distribution
            into the other. Expressed in the original units of the column.
            Lower is better. 0 = identical distributions.
            Normalised by the real column's std for cross-column comparability.

    Categorical / binary / ordinal columns:
        TVD (Total Variation Distance)
            TVD = 0.5 * Σ |P_real(x) - P_synth(x)|
            Ranges from 0 (identical) to 1 (completely different).
            Captures differences in category proportions.

Scoring:
    KS:          score = fraction of columns with p-value > 0.05
    Wasserstein: score = mean(1 - min(normalised_distance, 1))
    TVD:         score = 1 - mean(TVD)
    Final:       weighted average (KS 40%, Wasserstein 30%, TVD 30%)

Dependencies:
    scipy, numpy, pandas

Part of:
    Synthetic Data Engine — Auto-Adaptive Diffusion System
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from scipy import stats
from scipy.stats import wasserstein_distance


# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class KSResult:
    column      : str
    statistic   : float
    p_value     : float
    passed      : bool        # p_value > 0.05
    wasserstein : float       # raw
    wasserstein_norm: float   # normalised by real std


@dataclass
class TVDResult:
    column  : str
    tvd     : float
    passed  : bool            # tvd < 0.10


@dataclass
class FidelityResult:
    ks_results        : list[KSResult]
    tvd_results       : list[TVDResult]
    tvd_excluded      : list[str]   # high-cardinality cols excluded from TVD
    ks_score          : float       # fraction of columns passing KS
    wasserstein_score : float       # mean normalised wasserstein score
    tvd_score         : float       # 1 - mean TVD
    score             : float       # weighted final score 0-1


# ── Main evaluator ────────────────────────────────────────────────────────────

class FidelityEvaluator:
    """
    Evaluates marginal distribution fidelity between real and synthetic data.

    Usage:
        evaluator = FidelityEvaluator()
        result    = evaluator.evaluate(real_df, synthetic_df)
    """

    NUMERIC_TYPES = {"numerical_continuous", "ordinal"}

    # Columns whose unique_ratio exceeds this threshold are excluded from TVD.
    # They are generated externally (Faker, regex) and their per-value
    # distribution is meaningless — divergence is expected by design.
    TVD_CARDINALITY_THRESHOLD = 0.50

    def evaluate(
        self,
        real_df  : pd.DataFrame,
        synth_df : pd.DataFrame,
    ) -> FidelityResult:

        n_rows      = len(real_df)
        common_cols = [c for c in real_df.columns if c in synth_df.columns]

        # Detect high-cardinality columns to exclude from TVD
        # Rule: unique_ratio > 0.50 in the real dataset → external generation
        # (names, codes, IDs) → TVD is meaningless for these columns.
        tvd_excluded = [
            c for c in common_cols
            if (
                real_df[c].dropna().nunique() /
                max(real_df[c].dropna().count(), 1)
            ) > self.TVD_CARDINALITY_THRESHOLD
        ]
        tvd_cols = [c for c in common_cols if c not in tvd_excluded]

        ks_results  = []
        tvd_results = []

        for col in common_cols:
            real_s  = real_df[col].dropna()
            synth_s = synth_df[col].dropna()

            if len(real_s) == 0 or len(synth_s) == 0:
                continue

            if pd.api.types.is_numeric_dtype(real_s):
                ks_results.append(self._ks_wasserstein(col, real_s, synth_s))
            elif col in tvd_cols:
                tvd_results.append(self._tvd(col, real_s, synth_s))

        # Also compute TVD for low-cardinality numeric columns (ordinal/binary)
        for col in tvd_cols:
            real_s = real_df[col].dropna()
            if pd.api.types.is_numeric_dtype(real_s) and real_s.nunique() <= 10:
                tvd_results.append(self._tvd(col, real_df[col], synth_df[col]))

        # ── Scores ────────────────────────────────────────────────────────
        ks_score = (
            sum(r.passed for r in ks_results) / len(ks_results)
            if ks_results else 1.0
        )

        wasserstein_score = (
            float(np.mean([1 - min(r.wasserstein_norm, 1) for r in ks_results]))
            if ks_results else 1.0
        )

        tvd_score = (
            float(1 - np.mean([r.tvd for r in tvd_results]))
            if tvd_results else 1.0
        )

        score = 0.40 * ks_score + 0.30 * wasserstein_score + 0.30 * tvd_score

        return FidelityResult(
            ks_results        = ks_results,
            tvd_results       = tvd_results,
            tvd_excluded      = tvd_excluded,
            ks_score          = ks_score,
            wasserstein_score = wasserstein_score,
            tvd_score         = tvd_score,
            score             = float(np.clip(score, 0, 1)),
        )

    # ── KS + Wasserstein ──────────────────────────────────────────────────────

    def _ks_wasserstein(
        self,
        col    : str,
        real_s : pd.Series,
        synth_s: pd.Series,
    ) -> KSResult:
        real_arr  = real_s.astype(float).values
        synth_arr = synth_s.astype(float).values

        ks_stat, p_value = stats.ks_2samp(real_arr, synth_arr)
        w_dist           = float(wasserstein_distance(real_arr, synth_arr))
        real_std         = float(real_s.std()) or 1.0
        w_norm           = w_dist / real_std

        return KSResult(
            column           = col,
            statistic        = float(ks_stat),
            p_value          = float(p_value),
            passed           = p_value > 0.05,
            wasserstein      = w_dist,
            wasserstein_norm = w_norm,
        )

    # ── TVD ───────────────────────────────────────────────────────────────────

    def _tvd(
        self,
        col    : str,
        real_s : pd.Series,
        synth_s: pd.Series,
    ) -> TVDResult:
        real_counts  = real_s.value_counts(normalize=True)
        synth_counts = synth_s.value_counts(normalize=True)

        all_cats = set(real_counts.index) | set(synth_counts.index)
        tvd = 0.5 * sum(
            abs(real_counts.get(c, 0) - synth_counts.get(c, 0))
            for c in all_cats
        )

        return TVDResult(
            column = col,
            tvd    = float(tvd),
            passed = tvd < 0.10,
        )