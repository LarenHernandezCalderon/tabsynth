"""
diversity.py
─────────────────────────────────────────────────────────────────────────────
Phase 5 — Diversity Evaluator.

Measures whether the synthetic data covers the same statistical space as
the real data, without collapsing to repetitive patterns (mode collapse).

Metrics:
    Variance ratio
        For each numeric column:
            ratio = var_synth / var_real
        Ideal: ratio ≈ 1.0
        ratio << 1 → synthetic data is too concentrated (low diversity)
        ratio >> 1 → synthetic data is too spread out (unrealistic)
        Score: 1 - mean(|1 - ratio_clipped|) — penalises both extremes.

    PCA spread
        Projects both datasets into the principal component space fitted
        on the real data (first 5 components or fewer if not enough features).
        Measures the variance explained by the synthetic data in that space.
        Ideal: synthetic variance per PC ≈ real variance per PC.
        Score: 1 - mean(|1 - (var_synth_pc / var_real_pc)|) capped at 1.

    Unique value coverage (categorical)
        For low-cardinality columns:
            coverage = |categories_synth ∩ categories_real| / |categories_real|
        Ideal: 1.0 — all real categories appear in synthetic.
        Score: mean coverage across categorical columns.

Scoring:
    Final = 0.40 * variance_score + 0.40 * pca_score + 0.20 * coverage_score

Dependencies:
    sklearn, numpy, pandas

Part of:
    Synthetic Data Engine — Auto-Adaptive Diffusion System
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from sklearn.decomposition import PCA
from sklearn.impute        import SimpleImputer
from sklearn.preprocessing import LabelEncoder


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_PCA_COMPONENTS = 5
VARIANCE_CLIP      = 3.0   # clip variance ratio to [1/clip, clip] before scoring


# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class VarianceRatioResult:
    column        : str
    var_real      : float
    var_synth     : float
    ratio         : float   # var_synth / var_real
    score         : float   # 0-1


@dataclass
class PCARatioResult:
    component     : int
    var_real      : float
    var_synth     : float
    ratio         : float
    score         : float


@dataclass
class CoverageResult:
    column        : str
    n_real_cats   : int
    n_synth_cats  : int
    n_covered     : int
    coverage      : float   # 0-1


@dataclass
class DiversityResult:
    variance_results  : list[VarianceRatioResult]
    pca_results       : list[PCARatioResult]
    coverage_results  : list[CoverageResult]
    variance_score    : float
    pca_score         : float
    coverage_score    : float
    score             : float   # final 0-1


# ── Main evaluator ────────────────────────────────────────────────────────────

class DiversityEvaluator:
    """
    Evaluates diversity and coverage of the synthetic dataset.

    Usage:
        evaluator = DiversityEvaluator()
        result    = evaluator.evaluate(real_df, synthetic_df)
    """

    def evaluate(
        self,
        real_df  : pd.DataFrame,
        synth_df : pd.DataFrame,
    ) -> DiversityResult:

        common_cols = [c for c in real_df.columns if c in synth_df.columns]

        # ── Variance ratio ────────────────────────────────────────────────
        variance_results = []
        for col in common_cols:
            if not pd.api.types.is_numeric_dtype(real_df[col]):
                continue
            var_r = float(real_df[col].var())
            var_s = float(synth_df[col].var())
            if var_r < 1e-9:
                continue
            ratio = var_s / var_r
            # Score: penalise deviation from 1.0 in both directions
            ratio_clipped = np.clip(ratio, 1 / VARIANCE_CLIP, VARIANCE_CLIP)
            score = float(1 - abs(1 - ratio_clipped) / (VARIANCE_CLIP - 1))
            variance_results.append(VarianceRatioResult(
                column=col, var_real=var_r, var_synth=var_s,
                ratio=ratio, score=float(np.clip(score, 0, 1)),
            ))

        variance_score = (
            float(np.mean([r.score for r in variance_results]))
            if variance_results else 1.0
        )

        # ── PCA spread ────────────────────────────────────────────────────
        pca_results, pca_score = self._pca_spread(real_df, synth_df, common_cols)

        # ── Categorical coverage ───────────────────────────────────────────
        coverage_results = []
        for col in common_cols:
            s_real  = real_df[col].dropna()
            s_synth = synth_df[col].dropna()
            if s_real.nunique() > 30 or pd.api.types.is_float_dtype(s_real):
                continue
            real_cats  = set(s_real.unique())
            synth_cats = set(s_synth.unique())
            n_covered  = len(real_cats & synth_cats)
            coverage   = n_covered / len(real_cats) if real_cats else 1.0
            coverage_results.append(CoverageResult(
                column=col, n_real_cats=len(real_cats),
                n_synth_cats=len(synth_cats),
                n_covered=n_covered, coverage=float(coverage),
            ))

        coverage_score = (
            float(np.mean([r.coverage for r in coverage_results]))
            if coverage_results else 1.0
        )

        score = float(np.clip(
            0.40 * variance_score + 0.40 * pca_score + 0.20 * coverage_score,
            0, 1,
        ))

        return DiversityResult(
            variance_results = variance_results,
            pca_results      = pca_results,
            coverage_results = coverage_results,
            variance_score   = variance_score,
            pca_score        = pca_score,
            coverage_score   = coverage_score,
            score            = score,
        )

    # ── PCA helper ────────────────────────────────────────────────────────────

    def _pca_spread(self, real_df, synth_df, common_cols):
        """Fit PCA on real, project both, compare variance per component."""
        try:
            num_cols = [
                c for c in common_cols
                if pd.api.types.is_numeric_dtype(real_df[c])
                and real_df[c].nunique() > 1
            ]
            if len(num_cols) < 2:
                return [], 1.0

            X_real  = real_df[num_cols].values.astype(float)
            X_synth = synth_df[num_cols].values.astype(float)

            imp = SimpleImputer(strategy="median")
            imp.fit(X_real)
            X_real  = imp.transform(X_real)
            X_synth = imp.transform(X_synth)

            n_components = min(MAX_PCA_COMPONENTS, len(num_cols), len(X_real) - 1)
            pca = PCA(n_components=n_components, random_state=42)
            pca.fit(X_real)

            proj_real  = pca.transform(X_real)
            proj_synth = pca.transform(X_synth)

            results = []
            scores  = []
            for i in range(n_components):
                var_r = float(np.var(proj_real[:, i]))
                var_s = float(np.var(proj_synth[:, i]))
                ratio = var_s / var_r if var_r > 1e-9 else 1.0
                ratio_c = np.clip(ratio, 1 / VARIANCE_CLIP, VARIANCE_CLIP)
                sc = float(np.clip(1 - abs(1 - ratio_c) / (VARIANCE_CLIP - 1), 0, 1))
                results.append(PCARatioResult(
                    component=i + 1, var_real=var_r, var_synth=var_s,
                    ratio=ratio, score=sc,
                ))
                scores.append(sc)

            return results, float(np.mean(scores)) if scores else 1.0

        except Exception:
            return [], 1.0