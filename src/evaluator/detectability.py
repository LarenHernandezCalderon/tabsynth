"""
detectability.py
─────────────────────────────────────────────────────────────────────────────
Phase 5 — Detectability Evaluator.

Measures how hard it is for a classifier to distinguish real from synthetic.

Principle:
    If synthetic data is perfect, a classifier trained to label rows as
    "real" or "synthetic" should perform no better than random chance
    (accuracy ≈ 0.50, AUC ≈ 0.50).

    The closer to 0.50, the better.
    If accuracy > 0.70, the synthetic data has detectable patterns that
    differ from the real data — a clear quality problem.

How it works:
    1. Combine real and synthetic into one dataset.
       Label real rows as 1, synthetic rows as 0.
    2. Train a RandomForest binary classifier on 80% of this combined set.
    3. Evaluate on the held-out 20%.
    4. Report accuracy and ROC-AUC.

Scoring:
    Both metrics are scored by distance from 0.50 (random chance):
        detection_score = 1 - 2 * |metric - 0.50|
        metric=0.50 → score=1.0 (perfect — undetectable)
        metric=0.75 → score=0.50
        metric=1.00 → score=0.0 (completely detectable)
    Final = mean(accuracy_score, auc_score)

Dependencies:
    sklearn, numpy, pandas

Part of:
    Synthetic Data Engine — Auto-Adaptive Diffusion System
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import accuracy_score, roc_auc_score
from sklearn.impute          import SimpleImputer
import warnings


# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass
class DetectabilityResult:
    accuracy                  : float
    roc_auc                   : float
    accuracy_score            : float   # 0-1 (1 = undetectable)
    auc_score                 : float   # 0-1 (1 = undetectable)
    score                     : float   # final 0-1
    n_real                    : int
    n_synthetic               : int
    excluded_high_cardinality : list    # columns excluded from classifier
    n_features_used           : int     # columns actually used


# ── Main evaluator ────────────────────────────────────────────────────────────

class DetectabilityEvaluator:
    """
    Evaluates how distinguishable real and synthetic data are.

    Usage:
        evaluator = DetectabilityEvaluator()
        result    = evaluator.evaluate(real_df, synthetic_df)
    """

    def evaluate(
        self,
        real_df  : pd.DataFrame,
        synth_df : pd.DataFrame,
    ) -> DetectabilityResult:

        n_real  = len(real_df)
        n_synth = len(synth_df)

        # ── Build combined dataset ─────────────────────────────────────────
        # Exclude high-cardinality columns (names, codes, IDs, free-text).
        # These columns are unique per row by construction — the classifier
        # would trivially learn "unknown value = synthetic" and produce an
        # artificially high detectability score that has nothing to do with
        # the statistical quality of the numeric/categorical columns.
        #
        # Exclusion rules:
        #   1. Unique ratio > 50% in the real dataset → almost certainly an
        #      ID, name, or code column (e.g. Name, Ticket, Cabin)
        #   2. String column with unique ratio > 20% → free-text or code
        #   3. Sequential integer (min=1, max=n, no gaps) → PassengerId-like
        n_rows = len(real_df)
        common_cols = []
        excluded_high_cardinality = []

        for c in real_df.columns:
            if c not in synth_df.columns:
                continue
            s = real_df[c].dropna()
            unique_ratio = s.nunique() / n_rows if n_rows > 0 else 0

            # Rule 1: very high cardinality (any dtype)
            if unique_ratio > 0.50:
                excluded_high_cardinality.append(c)
                continue

            # Rule 2: string column with moderate-high cardinality
            if not pd.api.types.is_numeric_dtype(s) and unique_ratio > 0.20:
                excluded_high_cardinality.append(c)
                continue

            # Rule 3: sequential integer (PassengerId pattern)
            if pd.api.types.is_integer_dtype(s):
                if s.min() == 1 and s.max() == n_rows and s.nunique() == n_rows:
                    excluded_high_cardinality.append(c)
                    continue

            common_cols.append(c)

        real_part  = real_df[common_cols].copy()
        synth_part = synth_df[common_cols].copy()

        real_part["__label__"]  = 1
        synth_part["__label__"] = 0

        combined = pd.concat([real_part, synth_part], ignore_index=True)
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

        # ── Encode & impute ────────────────────────────────────────────────
        X_df = combined.drop(columns=["__label__"])
        y    = combined["__label__"].values

        for col in X_df.columns:
            if not pd.api.types.is_numeric_dtype(X_df[col]):
                le = LabelEncoder()
                X_df[col] = le.fit_transform(
                    X_df[col].astype(str).fillna("__missing__")
                )

        X = X_df.values.astype(float)
        imp = SimpleImputer(strategy="median")
        X   = imp.fit_transform(X)

        # ── Train / test split ─────────────────────────────────────────────
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42
            )

        # ── Train classifier ───────────────────────────────────────────────
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train, y_train)

        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = float(accuracy_score(y_test, y_pred))
        roc_auc  = float(roc_auc_score(y_test, y_proba))

        # ── Score: distance from 0.50 ──────────────────────────────────────
        acc_score = float(max(0, 1 - 2 * abs(accuracy - 0.50)))
        auc_score = float(max(0, 1 - 2 * abs(roc_auc  - 0.50)))
        score     = float(np.clip((acc_score + auc_score) / 2, 0, 1))

        return DetectabilityResult(
            accuracy                  = accuracy,
            roc_auc                   = roc_auc,
            accuracy_score            = acc_score,
            auc_score                 = auc_score,
            score                     = score,
            n_real                    = n_real,
            n_synthetic               = n_synth,
            excluded_high_cardinality = excluded_high_cardinality,
            n_features_used           = len(common_cols),
        )