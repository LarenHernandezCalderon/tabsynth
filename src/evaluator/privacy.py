"""
privacy.py
─────────────────────────────────────────────────────────────────────────────
Phase 5 — Privacy Evaluator.

Measures the re-identification risk of the synthetic dataset by comparing
the distances between synthetic and real records.

Metrics:
    Nearest-Neighbour Distance (NND)
        For each synthetic record, find its closest real record (L2 distance
        in the normalised feature space).
        If NND is very small, the synthetic record is nearly a copy of a real
        one — high re-identification risk.

    Distance to Closest Record (DCR)
        Mean and minimum NND across all synthetic records.
        Higher is better — larger minimum means no synthetic record is
        dangerously close to any real record.

    Distance Ratio (Privacy Ratio)
        Ratio = NND_synth / NND_real
        NND_real = mean distance between real records and their nearest
                   real neighbour (the "natural" density of the real data).
        Ratio >= 1.0 means synthetic records are at least as far from real
        records as real records are from each other → safe.
        Ratio < 0.5 means synthetic records are significantly closer to real
        records than real records are to each other → privacy risk.

Scoring:
    dcr_score   = min(mean_nnd / reference_distance, 1)
    ratio_score = min(distance_ratio, 1)
    Final       = 0.50 * dcr_score + 0.50 * ratio_score

Note on sampling:
    For large datasets (> 2000 rows), we sample 2000 rows for efficiency.
    This gives a statistically robust estimate without O(n²) cost.

Dependencies:
    sklearn, numpy, pandas

Part of:
    Synthetic Data Engine — Auto-Adaptive Diffusion System
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute        import SimpleImputer
from sklearn.neighbors     import NearestNeighbors
import warnings


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_SAMPLE_SIZE = 2000   # rows to sample for efficiency


# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass
class PrivacyResult:
    mean_nnd          : float   # mean nearest-neighbour distance synth→real
    min_nnd           : float   # minimum NND (most at-risk record)
    mean_nnd_real     : float   # mean NND real→real (reference density)
    distance_ratio    : float   # mean_nnd / mean_nnd_real
    dcr_score         : float   # 0-1
    ratio_score       : float   # 0-1
    score             : float   # final 0-1
    n_at_risk         : int     # synthetic records with NND < 0.05 (very close)
    privacy_warning   : bool    # True if distance_ratio < 0.5


# ── Main evaluator ────────────────────────────────────────────────────────────

class PrivacyEvaluator:
    """
    Evaluates re-identification risk via nearest-neighbour distances.

    Usage:
        evaluator = PrivacyEvaluator()
        result    = evaluator.evaluate(real_df, synthetic_df)
    """

    def evaluate(
        self,
        real_df  : pd.DataFrame,
        synth_df : pd.DataFrame,
    ) -> PrivacyResult:

        # ── Prepare numeric feature matrix ────────────────────────────────
        X_real, X_synth = self._prepare(real_df, synth_df)

        if X_real is None or X_synth is None or len(X_real) < 2:
            return PrivacyResult(
                mean_nnd=1.0, min_nnd=1.0, mean_nnd_real=1.0,
                distance_ratio=1.0, dcr_score=1.0, ratio_score=1.0,
                score=1.0, n_at_risk=0, privacy_warning=False,
            )

        # ── Sample for efficiency ──────────────────────────────────────────
        if len(X_real) > MAX_SAMPLE_SIZE:
            idx      = np.random.choice(len(X_real), MAX_SAMPLE_SIZE, replace=False)
            X_real   = X_real[idx]
        if len(X_synth) > MAX_SAMPLE_SIZE:
            idx      = np.random.choice(len(X_synth), MAX_SAMPLE_SIZE, replace=False)
            X_synth  = X_synth[idx]

        # ── NND: synthetic → real ──────────────────────────────────────────
        nbrs_real = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(X_real)
        distances_sr, _ = nbrs_real.kneighbors(X_synth)
        nnd_sr = distances_sr.flatten()

        mean_nnd  = float(np.mean(nnd_sr))
        min_nnd   = float(np.min(nnd_sr))
        n_at_risk = int(np.sum(nnd_sr < 0.05))

        # ── NND: real → real (reference density) ──────────────────────────
        # Use k=2 because k=1 would find the record itself (distance=0)
        nbrs_rr = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(X_real)
        distances_rr, _ = nbrs_rr.kneighbors(X_real)
        nnd_rr = distances_rr[:, 1].flatten()   # second column = nearest OTHER record

        mean_nnd_real = float(np.mean(nnd_rr))

        # ── Distance ratio ─────────────────────────────────────────────────
        distance_ratio = mean_nnd / mean_nnd_real if mean_nnd_real > 0 else 1.0

        # ── Scores ────────────────────────────────────────────────────────
        # DCR score: how far synthetic is from real relative to reference
        dcr_score   = float(min(mean_nnd / max(mean_nnd_real, 1e-9), 1.0))

        # Ratio score: direct ratio capped at 1
        ratio_score = float(min(distance_ratio, 1.0))

        score = float(np.clip(0.50 * dcr_score + 0.50 * ratio_score, 0, 1))

        return PrivacyResult(
            mean_nnd       = mean_nnd,
            min_nnd        = min_nnd,
            mean_nnd_real  = mean_nnd_real,
            distance_ratio = distance_ratio,
            dcr_score      = dcr_score,
            ratio_score    = ratio_score,
            score          = score,
            n_at_risk      = n_at_risk,
            privacy_warning= distance_ratio < 0.5,
        )

    # ── Preparation ───────────────────────────────────────────────────────────

    def _prepare(self, real_df, synth_df):
        """Encode, impute and normalise to [0,1] for distance computation."""
        try:
            common_cols = [
                c for c in real_df.columns
                if c in synth_df.columns
            ]

            real_enc  = real_df[common_cols].copy()
            synth_enc = synth_df[common_cols].copy()

            # Encode categoricals
            for col in common_cols:
                if not pd.api.types.is_numeric_dtype(real_enc[col]):
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    all_vals = pd.concat([
                        real_enc[col].astype(str),
                        synth_enc[col].astype(str),
                    ]).fillna("__missing__")
                    le.fit(all_vals)
                    real_enc[col]  = le.transform(real_enc[col].astype(str).fillna("__missing__"))
                    synth_enc[col] = le.transform(synth_enc[col].astype(str).fillna("__missing__"))

            X_real  = real_enc.values.astype(float)
            X_synth = synth_enc.values.astype(float)

            # Impute
            imp = SimpleImputer(strategy="median")
            imp.fit(X_real)
            X_real  = imp.transform(X_real)
            X_synth = imp.transform(X_synth)

            # Normalise to [0, 1] so all features contribute equally
            scaler = MinMaxScaler()
            scaler.fit(X_real)
            X_real  = scaler.transform(X_real)
            X_synth = scaler.transform(X_synth)

            return X_real, X_synth

        except Exception:
            return None, None