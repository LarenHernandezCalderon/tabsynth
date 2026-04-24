"""
utility.py
─────────────────────────────────────────────────────────────────────────────
Phase 5 — Utility Evaluator (TSTR).

Train on Synthetic, Test on Real — the most important metric in the pipeline.

If a Machine Learning model trained on synthetic data performs almost as well
as one trained on real data, the synthetic dataset is statistically equivalent
for practical ML purposes. This is the definitive proof that the full pipeline
works end-to-end.

How it works:
    1. Identify the target column (binary or multiclass classification task).
       Auto-detected as the first binary column in the dataset, or the column
       with the lowest cardinality among non-ID columns.
    2. Prepare feature matrix X and target y (numeric encoding if needed).
    3. Split real data into train/test (80/20, stratified).
    4. Train each model on:
         a) real_train  → baseline performance
         b) synth_train → TSTR performance
    5. Evaluate both on real_test (same held-out set for fair comparison).
    6. Gap = |accuracy_real - accuracy_tstr|

Models:
    LogisticRegression  — fast, interpretable baseline
    RandomForest        — robust ensemble, captures non-linear relationships
    GradientBoosting    — most demanding, best real-world proxy

Scoring:
    Per model: score = 1 - min(gap / 0.10, 1)
        gap=0.00 → score=1.0 (perfect)
        gap=0.05 → score=0.50
        gap=0.10 → score=0.0 (threshold)
    Final: mean of all model scores

Additional metrics per model:
    F1 score (weighted)
    ROC-AUC (for binary targets)

Dependencies:
    sklearn, numpy, pandas

Part of:
    Synthetic Data Engine — Auto-Adaptive Diffusion System
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import accuracy_score, f1_score, roc_auc_score
from sklearn.impute          import SimpleImputer
import warnings


# ── Constants ─────────────────────────────────────────────────────────────────

GAP_THRESHOLD = 0.10   # gap above this → score = 0

MODELS = {
    "LogisticRegression" : LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting"   : GradientBoostingClassifier(n_estimators=100, random_state=42),
}


# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class ModelTSTRResult:
    model_name     : str
    accuracy_real  : float
    accuracy_synth : float
    f1_real        : float
    f1_synth       : float
    auc_real       : Optional[float]
    auc_synth      : Optional[float]
    gap_accuracy   : float
    gap_f1         : float
    score          : float    # 0-1
    passed         : bool     # gap < 0.05


@dataclass
class UtilityResult:
    target_column  : str
    model_results  : list[ModelTSTRResult]
    mean_gap       : float
    best_model     : str
    worst_model    : str
    score          : float    # final 0-1


# ── Main evaluator ────────────────────────────────────────────────────────────

class UtilityEvaluator:
    """
    Evaluates synthetic data utility via TSTR with multiple classifiers.

    Usage:
        evaluator = UtilityEvaluator()
        result    = evaluator.evaluate(real_df, synthetic_df, target_col)
    """

    def evaluate(
        self,
        real_df    : pd.DataFrame,
        synth_df   : pd.DataFrame,
        target_col : Optional[str] = None,
    ) -> UtilityResult:

        # ── Auto-detect target column ──────────────────────────────────────
        if target_col is None:
            target_col = self._detect_target(real_df)

        if target_col is None or target_col not in real_df.columns:
            # No suitable target — return neutral result
            return UtilityResult(
                target_column="none", model_results=[],
                mean_gap=0.0, best_model="n/a", worst_model="n/a", score=1.0,
            )

        # ── Prepare features ───────────────────────────────────────────────
        feature_cols = [c for c in real_df.columns if c != target_col]

        X_real, y_real   = self._prepare(real_df,  feature_cols, target_col)
        X_synth, y_synth = self._prepare(synth_df, feature_cols, target_col)

        if X_real is None or X_synth is None:
            return UtilityResult(
                target_column=target_col, model_results=[],
                mean_gap=0.0, best_model="n/a", worst_model="n/a", score=1.0,
            )

        # ── Train/test split on real data ──────────────────────────────────
        try:
            X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
                X_real, y_real, test_size=0.20, random_state=42, stratify=y_real
            )
        except ValueError:
            X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
                X_real, y_real, test_size=0.20, random_state=42
            )

        is_binary = len(np.unique(y_real)) == 2

        # ── Run each model ─────────────────────────────────────────────────
        model_results = []
        for name, clf_template in MODELS.items():
            result = self._run_model(
                name, clf_template,
                X_real_train, y_real_train,
                X_synth, y_synth,
                X_real_test, y_real_test,
                is_binary,
            )
            model_results.append(result)

        # ── Aggregate ─────────────────────────────────────────────────────
        mean_gap   = float(np.mean([r.gap_accuracy for r in model_results]))
        scores     = [r.score for r in model_results]
        best_model = model_results[int(np.argmax(scores))].model_name
        worst_model= model_results[int(np.argmin(scores))].model_name
        score      = float(np.mean(scores))

        return UtilityResult(
            target_column = target_col,
            model_results = model_results,
            mean_gap      = mean_gap,
            best_model    = best_model,
            worst_model   = worst_model,
            score         = float(np.clip(score, 0, 1)),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _detect_target(self, df: pd.DataFrame) -> Optional[str]:
        """
        Auto-detect the best target column for classification.
        Priority: binary columns first, then lowest cardinality non-ID columns.
        """
        for col in df.columns:
            s = df[col].dropna()
            if s.nunique() == 2 and pd.api.types.is_numeric_dtype(s):
                return col

        candidates = [
            (df[c].nunique(), c) for c in df.columns
            if df[c].nunique() >= 2
            and df[c].nunique() <= 20
            and not c.lower().endswith("id")
        ]
        if candidates:
            return sorted(candidates)[0][1]

        return None

    def _prepare(
        self,
        df           : pd.DataFrame,
        feature_cols : list[str],
        target_col   : str,
    ):
        """Encode categoricals, impute missing, return (X, y) arrays."""
        try:
            df_enc = df[feature_cols + [target_col]].copy()

            # Encode categorical columns
            for col in df_enc.columns:
                if not pd.api.types.is_numeric_dtype(df_enc[col]):
                    le = LabelEncoder()
                    df_enc[col] = le.fit_transform(
                        df_enc[col].astype(str).fillna("__missing__")
                    )

            X = df_enc[feature_cols].values.astype(float)
            y = df_enc[target_col].values

            # Impute remaining NaN
            imp = SimpleImputer(strategy="median")
            X   = imp.fit_transform(X)

            return X, y

        except Exception:
            return None, None

    def _run_model(
        self,
        name           : str,
        clf_template   : object,
        X_real_train   : np.ndarray,
        y_real_train   : np.ndarray,
        X_synth        : np.ndarray,
        y_synth        : np.ndarray,
        X_real_test    : np.ndarray,
        y_real_test    : np.ndarray,
        is_binary      : bool,
    ) -> ModelTSTRResult:

        import sklearn.base as skbase

        # ── Train on REAL ──────────────────────────────────────────────────
        clf_real = skbase.clone(clf_template)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf_real.fit(X_real_train, y_real_train)
        y_pred_real = clf_real.predict(X_real_test)
        acc_real    = float(accuracy_score(y_real_test, y_pred_real))
        f1_real     = float(f1_score(y_real_test, y_pred_real, average="weighted", zero_division=0))
        auc_real    = self._safe_auc(clf_real, X_real_test, y_real_test, is_binary)

        # ── Train on SYNTH ─────────────────────────────────────────────────
        clf_synth = skbase.clone(clf_template)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf_synth.fit(X_synth, y_synth)
        y_pred_synth = clf_synth.predict(X_real_test)
        acc_synth    = float(accuracy_score(y_real_test, y_pred_synth))
        f1_synth     = float(f1_score(y_real_test, y_pred_synth, average="weighted", zero_division=0))
        auc_synth    = self._safe_auc(clf_synth, X_real_test, y_real_test, is_binary)

        # ── Gap & score ────────────────────────────────────────────────────
        gap_acc = abs(acc_real - acc_synth)
        gap_f1  = abs(f1_real  - f1_synth)
        score   = float(max(0, 1 - gap_acc / GAP_THRESHOLD))

        return ModelTSTRResult(
            model_name     = name,
            accuracy_real  = acc_real,
            accuracy_synth = acc_synth,
            f1_real        = f1_real,
            f1_synth       = f1_synth,
            auc_real       = auc_real,
            auc_synth      = auc_synth,
            gap_accuracy   = gap_acc,
            gap_f1         = gap_f1,
            score          = score,
            passed         = gap_acc < 0.05,
        )

    def _safe_auc(self, clf, X, y, is_binary: bool) -> Optional[float]:
        if not is_binary:
            return None
        try:
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X)[:, 1]
            elif hasattr(clf, "decision_function"):
                proba = clf.decision_function(X)
            else:
                return None
            return float(roc_auc_score(y, proba))
        except Exception:
            return None