"""
correlation_analyzer.py
─────────────────────────────────────────────────────────────────────────────
Layer 4 of the analysis pipeline. Analyzes statistical dependencies between
ALL column pairs in the dataset — numeric, categorical, and mixed.

Why this matters for the diffusion model:
    The model must preserve inter-variable relationships in synthetic data.
    A synthetic dataset where Pclass and Fare are uncorrelated is statistically
    wrong and useless for ML training. This layer maps all those dependencies
    so the preprocessor and model architecture can account for them.

What it produces:
    - Numeric  ↔ Numeric      : Pearson + Spearman correlation
    - Categoric ↔ Categorical  : Cramér's V (chi-square based, range 0–1)
    - Numeric  ↔ Categorical   : Eta-squared (effect size, range 0–1)

Output per analyzed pair (CorrelationPair):
    - col_a, col_b          : column names
    - method                : pearson | spearman | cramers_v | eta_squared
    - score                 : correlation strength (0.0 to 1.0, always positive)
    - strength              : weak | moderate | strong | very_strong
    - direction             : positive | negative | n/a (only for numeric pairs)

Also produces:
    - Numeric correlation matrix (Pearson)
    - Full flagged pairs list sorted by score descending
    - Summary report with strong/very_strong pairs highlighted

Strength thresholds:
    |score| < 0.20  → weak
    |score| < 0.40  → moderate
    |score| < 0.60  → strong
    |score| >= 0.60 → very_strong

Dependencies:
    pandas, numpy, scipy, itertools

Part of:
    Synthetic Tabular Data Engine — Auto-Adaptive Diffusion System
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from itertools import combinations
from scipy import stats
from scipy.stats import chi2_contingency

from analyzer.metadata_extractor import ColumnMetadata


# ── Thresholds ────────────────────────────────────────────────────────────────
WEAK_THRESHOLD       = 0.20
MODERATE_THRESHOLD   = 0.40
STRONG_THRESHOLD     = 0.60

# Types considered numeric for correlation purposes
NUMERIC_TYPES  = {"numerical_continuous", "ordinal", "binary"}

# Types considered categorical for correlation purposes
CATEGORICAL_TYPES = {"categorical", "binary"}

# String binary columns (e.g. Sex: male/female) need encoding before analysis
STRING_BINARY_TYPES = {"binary"}


# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class CorrelationPair:
    """Correlation result between two columns."""
    col_a     : str
    col_b     : str
    method    : str          # pearson | spearman | cramers_v | eta_squared
    score     : float        # always positive (0.0 to 1.0)
    strength  : str          # weak | moderate | strong | very_strong
    direction : str          # positive | negative | n/a


# ── Main analyzer ─────────────────────────────────────────────────────────────

class CorrelationAnalyzer:
    """
    Analyzes pairwise correlations across all column types.

    Automatically selects the correct statistical method for each pair:
        - Numeric  ↔ Numeric     → Pearson + Spearman
        - Categoric ↔ Categorical → Cramér's V
        - Numeric  ↔ Categorical  → Eta-squared

    Skips: id_like, datetime, unknown_string, and non-generatable code columns.
    """

    def __init__(self, df: pd.DataFrame, columns: list[ColumnMetadata]):
        self.df       = df
        self.columns  = columns
        self.pairs:   list[CorrelationPair] = []
        self.matrix:  Optional[pd.DataFrame] = None

        # Build lookup: column_name → detected_type
        self.type_map = {col.column_name: col.detected_type for col in columns}

    def analyze(self) -> list[CorrelationPair]:
        """Run full correlation analysis across all valid column pairs."""
        self.pairs  = []

        # Filter out columns that should not be analyzed
        valid_cols = self._get_valid_columns()

        # Encode string binary columns (e.g. Sex → 0/1) for numeric analysis
        encoded_df = self._encode_string_binaries(valid_cols)

        # Build numeric correlation matrix (Pearson) for numeric columns only
        numeric_cols = [
            c for c in valid_cols
            if self.type_map[c] in NUMERIC_TYPES
            and pd.api.types.is_numeric_dtype(self.df[c])
        ]
        if len(numeric_cols) >= 2:
            self.matrix = encoded_df[numeric_cols].corr(method="pearson")

        # Analyze all pairs
        for col_a, col_b in combinations(valid_cols, 2):
            pair = self._analyze_pair(encoded_df, col_a, col_b)
            if pair is not None:
                self.pairs.append(pair)

        # Sort by score descending
        self.pairs.sort(key=lambda p: p.score, reverse=True)

        return self.pairs

    # ── Valid column filter ───────────────────────────────────────────────────

    def _get_valid_columns(self) -> list[str]:
        """
        Returns columns eligible for correlation analysis.
        Skips: id_like, datetime, unknown_string columns.
        """
        skip_types = {"id_like", "datetime", "unknown_string", "empty"}
        return [
            col.column_name for col in self.columns
            if col.detected_type not in skip_types
        ]

    # ── String binary encoder ─────────────────────────────────────────────────

    def _encode_string_binaries(self, valid_cols: list[str]) -> pd.DataFrame:
        """
        Encodes string binary columns (e.g. Sex: male/female → 0/1)
        so they can participate in numeric correlation analysis.
        Returns a working copy of the dataframe with encoded columns.
        """
        df_work = self.df[valid_cols].copy()

        for col in valid_cols:
            if self.type_map[col] == "binary" and df_work[col].dtype == object:
                # Label encode: first unique value → 0, second → 1
                unique_vals = df_work[col].dropna().unique()
                if len(unique_vals) == 2:
                    mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                    df_work[col] = df_work[col].map(mapping)

        return df_work

    # ── Pair dispatcher ───────────────────────────────────────────────────────

    def _analyze_pair(
        self,
        df    : pd.DataFrame,
        col_a : str,
        col_b : str,
    ) -> Optional[CorrelationPair]:
        """
        Selects the correct method for each pair and computes correlation.
        Returns None if the pair cannot be analyzed.
        """
        type_a = self.type_map[col_a]
        type_b = self.type_map[col_b]

        is_numeric_a = type_a in NUMERIC_TYPES and pd.api.types.is_numeric_dtype(df[col_a])
        is_numeric_b = type_b in NUMERIC_TYPES and pd.api.types.is_numeric_dtype(df[col_b])
        is_categ_a   = type_a in CATEGORICAL_TYPES
        is_categ_b   = type_b in CATEGORICAL_TYPES

        # Drop rows where either column is null
        pair_df = df[[col_a, col_b]].dropna()
        if len(pair_df) < 10:
            return None

        # ── Numeric ↔ Numeric ─────────────────────────────────────────────────
        if is_numeric_a and is_numeric_b:
            return self._pearson_spearman(pair_df, col_a, col_b)

        # ── Categorical ↔ Categorical ─────────────────────────────────────────
        if is_categ_a and is_categ_b:
            return self._cramers_v(pair_df, col_a, col_b)

        # ── Numeric ↔ Categorical (mixed) ─────────────────────────────────────
        if is_numeric_a and is_categ_b:
            return self._eta_squared(pair_df, numeric_col=col_a, categ_col=col_b)
        if is_categ_a and is_numeric_b:
            return self._eta_squared(pair_df, numeric_col=col_b, categ_col=col_a)

        return None

    # ── Pearson + Spearman (numeric ↔ numeric) ────────────────────────────────

    def _pearson_spearman(
        self,
        df    : pd.DataFrame,
        col_a : str,
        col_b : str,
    ) -> CorrelationPair:
        """
        Uses Spearman when data is non-normal or ordinal,
        Pearson as the primary score otherwise.
        Reports the higher of the two absolute values.
        """
        pearson_r,  _ = stats.pearsonr(df[col_a], df[col_b])
        spearman_r, _ = stats.spearmanr(df[col_a], df[col_b])

        # Use the stronger signal
        if abs(spearman_r) > abs(pearson_r):
            score     = abs(spearman_r)
            method    = "spearman"
            direction = "positive" if spearman_r > 0 else "negative"
        else:
            score     = abs(pearson_r)
            method    = "pearson"
            direction = "positive" if pearson_r > 0 else "negative"

        return CorrelationPair(
            col_a     = col_a,
            col_b     = col_b,
            method    = method,
            score     = round(score, 4),
            strength  = self._classify_strength(score),
            direction = direction,
        )

    # ── Cramér's V (categorical ↔ categorical) ────────────────────────────────

    def _cramers_v(
        self,
        df    : pd.DataFrame,
        col_a : str,
        col_b : str,
    ) -> Optional[CorrelationPair]:
        """
        Cramér's V is a symmetric measure based on chi-square.
        Range: 0 (no association) to 1 (perfect association).
        """
        try:
            contingency = pd.crosstab(df[col_a], df[col_b])
            chi2, _, _, _ = chi2_contingency(contingency)
            n    = len(df)
            k    = min(contingency.shape) - 1
            if k == 0 or n == 0:
                return None
            v = np.sqrt(chi2 / (n * k))
            v = min(float(v), 1.0)  # clamp to [0, 1]

            return CorrelationPair(
                col_a     = col_a,
                col_b     = col_b,
                method    = "cramers_v",
                score     = round(v, 4),
                strength  = self._classify_strength(v),
                direction = "n/a",
            )
        except Exception:
            return None

    # ── Eta-squared (numeric ↔ categorical) ──────────────────────────────────

    def _eta_squared(
        self,
        df          : pd.DataFrame,
        numeric_col : str,
        categ_col   : str,
    ) -> Optional[CorrelationPair]:
        """
        Eta-squared measures how much variance in the numeric column
        is explained by the categorical grouping.
        Range: 0 (no effect) to 1 (perfect grouping).
        """
        try:
            groups      = df.groupby(categ_col)[numeric_col].apply(list)
            grand_mean  = df[numeric_col].mean()
            ss_total    = ((df[numeric_col] - grand_mean) ** 2).sum()

            ss_between = sum(
                len(g) * (np.mean(g) - grand_mean) ** 2
                for g in groups
            )

            if ss_total == 0:
                return None

            eta2 = float(ss_between / ss_total)
            eta2 = min(eta2, 1.0)  # clamp to [0, 1]

            return CorrelationPair(
                col_a     = numeric_col,
                col_b     = categ_col,
                method    = "eta_squared",
                score     = round(eta2, 4),
                strength  = self._classify_strength(eta2),
                direction = "n/a",
            )
        except Exception:
            return None

    # ── Strength classifier ───────────────────────────────────────────────────

    def _classify_strength(self, score: float) -> str:
        if score < WEAK_THRESHOLD:
            return "weak"
        elif score < MODERATE_THRESHOLD:
            return "moderate"
        elif score < STRONG_THRESHOLD:
            return "strong"
        else:
            return "very_strong"

    # ── Summary printer ───────────────────────────────────────────────────────

    def summary(self) -> None:
        if not self.pairs:
            print("No correlation results to display.")
            return

        strength_icons = {
            "weak"        : "⬜",
            "moderate"    : "🟨",
            "strong"      : "🟧",
            "very_strong" : "🟥",
        }

        method_labels = {
            "pearson"     : "Pearson",
            "spearman"    : "Spearman",
            "cramers_v"   : "Cramér's V",
            "eta_squared" : "Eta²",
        }

        print("\n🔗 Correlation Analysis Report (Layer 4)")
        print("=" * 55)
        print("\nAll pairs sorted by correlation strength:\n")

        for p in self.pairs:
            icon   = strength_icons.get(p.strength, "⬜")
            method = method_labels.get(p.method, p.method)
            dir_str = f" ({p.direction})" if p.direction != "n/a" else ""
            print(f"   {icon} {p.col_a:<12} ↔ {p.col_b:<12}  "
                  f"{method:<12}  score={p.score:.3f}  [{p.strength}]{dir_str}")

        print("\n" + "=" * 55)


        # Print numeric correlation matrix if available
        if self.matrix is not None:
            print("\n📊 Numeric Correlation Matrix (Pearson):")
            print(self.matrix.round(3).to_string())

        print()