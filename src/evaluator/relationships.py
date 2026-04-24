"""
relationships.py
─────────────────────────────────────────────────────────────────────────────
Phase 5 — Relationships Evaluator.

Measures how well the synthetic data preserves the pairwise relationships
(correlations) between columns.

Metrics:
    Pearson correlation matrix
        Linear relationships between numeric columns.
        Computes the matrix for real and synthetic separately,
        then measures the element-wise difference.

    Spearman correlation matrix
        Monotonic relationships — more robust than Pearson for
        non-normal distributions and ordinal columns.

    Frobenius norm
        ||M_real - M_synth||_F  — single number summarising the total
        difference between the two correlation matrices.
        Lower is better. Normalised by matrix size for comparability.

    Mean absolute error per pair
        Average |r_real - r_synth| across all column pairs.
        Threshold: < 0.05 per pair is considered preserved.

Scoring:
    pair_score  = fraction of pairs where |delta| < 0.10
    frob_score  = 1 - min(normalised_frobenius, 1)
    Final       = 0.50 * pair_score + 0.50 * frob_score

Dependencies:
    scipy, numpy, pandas

Part of:
    Synthetic Data Engine — Auto-Adaptive Diffusion System
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import stats


# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class CorrelationPairResult:
    col_a         : str
    col_b         : str
    pearson_real  : float
    pearson_synth : float
    pearson_delta : float
    spearman_real : float
    spearman_synth: float
    spearman_delta: float
    preserved     : bool   # max(pearson_delta, spearman_delta) < 0.10


@dataclass
class RelationshipsResult:
    pair_results        : list[CorrelationPairResult]
    pearson_matrix_real : pd.DataFrame
    pearson_matrix_synth: pd.DataFrame
    spearman_matrix_real: pd.DataFrame
    spearman_matrix_synth:pd.DataFrame
    frobenius_pearson   : float   # normalised
    frobenius_spearman  : float   # normalised
    mean_delta_pearson  : float
    mean_delta_spearman : float
    pair_score          : float   # fraction of pairs preserved
    frob_score          : float
    score               : float   # final 0-1


# ── Main evaluator ────────────────────────────────────────────────────────────

class RelationshipsEvaluator:
    """
    Evaluates correlation structure preservation.

    Usage:
        evaluator = RelationshipsEvaluator()
        result    = evaluator.evaluate(real_df, synthetic_df)
    """

    def evaluate(
        self,
        real_df  : pd.DataFrame,
        synth_df : pd.DataFrame,
    ) -> RelationshipsResult:

        # Only numeric columns present in both DataFrames
        num_cols = [
            c for c in real_df.columns
            if c in synth_df.columns
            and pd.api.types.is_numeric_dtype(real_df[c])
            and pd.api.types.is_numeric_dtype(synth_df[c])
            and real_df[c].nunique() > 1
            and synth_df[c].nunique() > 1
        ]

        if len(num_cols) < 2:
            # Not enough numeric columns — return neutral result
            empty_df = pd.DataFrame()
            return RelationshipsResult(
                pair_results=[], pearson_matrix_real=empty_df,
                pearson_matrix_synth=empty_df, spearman_matrix_real=empty_df,
                spearman_matrix_synth=empty_df, frobenius_pearson=0.0,
                frobenius_spearman=0.0, mean_delta_pearson=0.0,
                mean_delta_spearman=0.0, pair_score=1.0, frob_score=1.0,
                score=1.0,
            )

        real_num  = real_df[num_cols].copy()
        synth_num = synth_df[num_cols].copy()

        # ── Correlation matrices ───────────────────────────────────────────
        pearson_real   = real_num.corr(method="pearson")
        pearson_synth  = synth_num.corr(method="pearson")
        spearman_real  = real_num.corr(method="spearman")
        spearman_synth = synth_num.corr(method="spearman")

        # Fill NaN (e.g. zero-variance columns after filtering)
        for m in [pearson_real, pearson_synth, spearman_real, spearman_synth]:
            m.fillna(0.0, inplace=True)

        # ── Frobenius norm ─────────────────────────────────────────────────
        n = len(num_cols)
        frob_p = float(np.linalg.norm(
            (pearson_real - pearson_synth).values, "fro"
        )) / n
        frob_s = float(np.linalg.norm(
            (spearman_real - spearman_synth).values, "fro"
        )) / n

        # ── Per-pair results ───────────────────────────────────────────────
        pair_results = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                a, b = num_cols[i], num_cols[j]

                pr  = float(pearson_real.loc[a, b])
                ps  = float(pearson_synth.loc[a, b])
                sr  = float(spearman_real.loc[a, b])
                ss  = float(spearman_synth.loc[a, b])
                pd_ = abs(pr - ps)
                sd  = abs(sr - ss)

                pair_results.append(CorrelationPairResult(
                    col_a          = a,
                    col_b          = b,
                    pearson_real   = pr,
                    pearson_synth  = ps,
                    pearson_delta  = pd_,
                    spearman_real  = sr,
                    spearman_synth = ss,
                    spearman_delta = sd,
                    preserved      = max(pd_, sd) < 0.10,
                ))

        # ── Summary stats ──────────────────────────────────────────────────
        mean_dp = float(np.mean([r.pearson_delta  for r in pair_results])) if pair_results else 0.0
        mean_ds = float(np.mean([r.spearman_delta for r in pair_results])) if pair_results else 0.0

        pair_score = (
            sum(r.preserved for r in pair_results) / len(pair_results)
            if pair_results else 1.0
        )

        frob_score = float(1 - min((frob_p + frob_s) / 2, 1))

        score = 0.50 * pair_score + 0.50 * frob_score

        return RelationshipsResult(
            pair_results         = pair_results,
            pearson_matrix_real  = pearson_real,
            pearson_matrix_synth = pearson_synth,
            spearman_matrix_real = spearman_real,
            spearman_matrix_synth= spearman_synth,
            frobenius_pearson    = frob_p,
            frobenius_spearman   = frob_s,
            mean_delta_pearson   = mean_dp,
            mean_delta_spearman  = mean_ds,
            pair_score           = float(pair_score),
            frob_score           = float(frob_score),
            score                = float(np.clip(score, 0, 1)),
        )