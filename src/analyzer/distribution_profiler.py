"""
distribution_profiler.py
─────────────────────────────────────────────────────────────────────────────
Analyzes the statistical distribution shape of every numerical column
in the dataset.

While MetadataExtractor computes basic stats (mean, std, min, max),
this module goes deeper — it identifies the actual distribution type,
severity of skewness, outlier presence, and recommends the best
transformation to apply before feeding the data into the diffusion model.

What it produces (DistributionProfile) per numerical column:
    - distribution_type      : shape of the distribution
                               (normal, log_normal, skewed, bimodal,
                                zero_inflated, uniform, exponential, peaked)
    - normality_test         : Shapiro-Wilk result (is it truly normal?)
    - skewness_severity      : none | mild | moderate | severe
    - outlier_analysis       : count, ratio, and IQR bounds
    - recommended_transform  : what to apply before model training
                               (none, log, sqrt, quantile)

Why this matters:
    The diffusion model learns distributions during training.
    If Fare is severely skewed and we feed it raw, the model struggles.
    With a log transform, Fare becomes roughly normal — model learns easily
    — we invert the transform after generation — realistic synthetic Fare.

Dependencies:
    pandas, numpy, scipy

Part of:
    Synthetic Tabular Data Engine — Auto-Adaptive Diffusion System
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from scipy import stats

from analyzer.metadata_extractor import ColumnMetadata


# ── Thresholds ────────────────────────────────────────────────────────────────
SKEW_MILD_THRESHOLD     = 0.5    # |skew| > 0.5  → mild skew
SKEW_MODERATE_THRESHOLD = 1.0    # |skew| > 1.0  → moderate skew
SKEW_SEVERE_THRESHOLD   = 2.0    # |skew| > 2.0  → severe skew
NORMALITY_P_THRESHOLD   = 0.05   # p < 0.05 → reject normality
ZERO_INFLATED_THRESHOLD = 0.5    # > 50% zeros → zero-inflated
OUTLIER_Z_THRESHOLD     = 3.0    # Z-score for outlier detection
SHAPIRO_MAX_SAMPLE      = 5000   # Shapiro-Wilk max recommended sample size
ORDINAL_MAX_UNIQUE      = 15     # threshold to distinguish discrete vs continuous


# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class NormalityTest:
    """Result of the Shapiro-Wilk normality test."""
    is_normal  : bool
    p_value    : float
    statistic  : float
    note       : str


@dataclass
class OutlierAnalysis:
    """IQR-based outlier detection results."""
    count       : int
    ratio       : float       # outliers / total rows
    lower_bound : float
    upper_bound : float
    note        : str


@dataclass
class DistributionProfile:
    """Full distribution analysis for a single numerical column."""
    column_name           : str
    distribution_type     : str        # normal | log_normal | exponential |
                                       # skewed | bimodal | zero_inflated | uniform | peaked
    skewness_severity     : str        # none | mild | moderate | severe
    normality             : NormalityTest
    outliers              : OutlierAnalysis
    recommended_transform : str        # none | log | sqrt | quantile
    transform_reason      : str        # human-readable explanation


# ── Main profiler ─────────────────────────────────────────────────────────────

class DistributionProfiler:
    """
    Analyzes the distribution shape of all numerical columns.

    Receives the list of ColumnMetadata from MetadataExtractor and
    produces a DistributionProfile for each numerical column.
    """

    def __init__(self, df: pd.DataFrame, columns: list[ColumnMetadata]):
        self.df      = df
        self.columns = columns
        self.profiles: list[DistributionProfile] = []

    def profile(self) -> list[DistributionProfile]:
        """Run distribution analysis on all numerical columns."""
        self.profiles = []

        numerical_types = {"numerical_continuous", "ordinal", "binary"}
        numerical_cols  = [
            col for col in self.columns
            if col.detected_type in numerical_types
        ]

        for meta in numerical_cols:
            series = self.df[meta.column_name].dropna()
            if len(series) < 10:
                continue
            # Guard: skip columns that are not actually numeric dtype
            # (e.g. binary string columns like Sex: "male"/"female")
            if not pd.api.types.is_numeric_dtype(series):
                continue
            profile = self._profile_column(meta.column_name, series)
            self.profiles.append(profile)

        return self.profiles

    # ── Column profiler ───────────────────────────────────────────────────────

    def _profile_column(self, col: str, series: pd.Series) -> DistributionProfile:
        skewness          = float(series.skew())
        skewness_severity = self._classify_skewness(skewness)
        normality         = self._test_normality(series)
        outliers          = self._detect_outliers(series)
        dist_type         = self._classify_distribution(series, skewness, normality)
        transform, reason = self._recommend_transform(dist_type, skewness, skewness_severity, series)

        return DistributionProfile(
            column_name           = col,
            distribution_type     = dist_type,
            skewness_severity     = skewness_severity,
            normality             = normality,
            outliers              = outliers,
            recommended_transform = transform,
            transform_reason      = reason,
        )

    # ── Skewness classifier ───────────────────────────────────────────────────

    def _classify_skewness(self, skewness: float) -> str:
        abs_skew = abs(skewness)
        if abs_skew <= SKEW_MILD_THRESHOLD:
            return "none"
        elif abs_skew <= SKEW_MODERATE_THRESHOLD:
            return "mild"
        elif abs_skew <= SKEW_SEVERE_THRESHOLD:
            return "moderate"
        else:
            return "severe"

    # ── Normality test (Shapiro-Wilk) ─────────────────────────────────────────

    def _test_normality(self, series: pd.Series) -> NormalityTest:
        # Shapiro-Wilk is unreliable above ~5000 samples → subsample
        sample = series.sample(min(len(series), SHAPIRO_MAX_SAMPLE), random_state=42)

        try:
            stat, p_value = stats.shapiro(sample)
            is_normal     = p_value >= NORMALITY_P_THRESHOLD

            note = (
                "Consistent with normal distribution"
                if is_normal
                else "Significantly deviates from normal distribution"
            )

            return NormalityTest(
                is_normal = is_normal,
                p_value   = round(float(p_value), 6),
                statistic = round(float(stat), 6),
                note      = note,
            )

        except Exception as e:
            return NormalityTest(
                is_normal = False,
                p_value   = 0.0,
                statistic = 0.0,
                note      = f"Normality test failed: {e}",
            )

    # ── Outlier detection (IQR method) ────────────────────────────────────────

    def _detect_outliers(self, series: pd.Series) -> OutlierAnalysis:
        q1  = float(series.quantile(0.25))
        q3  = float(series.quantile(0.75))
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        count        = int(outlier_mask.sum())
        ratio        = round(count / len(series), 4)

        if ratio == 0:
            note = "No outliers detected"
        elif ratio < 0.01:
            note = "Very few outliers — minimal impact"
        elif ratio < 0.05:
            note = "Moderate outliers — worth monitoring"
        else:
            note = "High outlier presence — transform recommended"

        return OutlierAnalysis(
            count       = count,
            ratio       = ratio,
            lower_bound = round(lower_bound, 4),
            upper_bound = round(upper_bound, 4),
            note        = note,
        )

    # ── Distribution classifier ───────────────────────────────────────────────

    def _classify_distribution(
        self,
        series    : pd.Series,
        skewness  : float,
        normality : NormalityTest
    ) -> str:

        # Binary: only 2 unique values — checked FIRST before zero_inflated
        if series.nunique() == 2:
            return "binary"

        # Zero-inflated: more than 50% of values are exactly zero
        if (series == 0).sum() / len(series) > ZERO_INFLATED_THRESHOLD:
            return "zero_inflated"

        # Normal: passes normality test
        if normality.is_normal:
            return "normal"

        # Log-normal: positively skewed + all values positive
        if skewness > SKEW_MODERATE_THRESHOLD and series.min() >= 0:
            log_series = np.log1p(series)
            _, p = stats.shapiro(
                log_series.sample(min(len(log_series), SHAPIRO_MAX_SAMPLE), random_state=42)
            )
            if p >= NORMALITY_P_THRESHOLD:
                return "log_normal"

        # Exponential: severe positive skew + all values positive
        if skewness > SKEW_SEVERE_THRESHOLD and series.min() >= 0:
            return "exponential"

        # Peaked: high kurtosis (leptokurtic) — checked BEFORE uniform
        # hours-per-week: low std/range but Q1=Q2=40, kurtosis>>3 → dominant spike, not uniform
        if float(series.kurt()) > 3.0:
            return "peaked"

        # Uniform: very low std relative to range AND low skewness AND healthy IQR
        # IQR guard: true uniform has IQR ≈ range/2, not near zero
        value_range = series.max() - series.min()
        iqr = float(series.quantile(0.75) - series.quantile(0.25))
        if (value_range > 0
                and series.std() / value_range < 0.15
                and abs(skewness) < SKEW_MILD_THRESHOLD
                and iqr / value_range > 0.10):
            return "uniform"

        # Bimodal: negative kurtosis suggests flat/multimodal shape
        if float(series.kurt()) < -1.0:
            return "bimodal"

        # General skewed fallback
        return "skewed"

    # ── Transform recommender ─────────────────────────────────────────────────

    def _recommend_transform(
        self,
        dist_type         : str,
        skewness          : float,
        skewness_severity : str,
        series            : pd.Series,
    ) -> tuple[str, str]:

        zero_ratio = (series == 0).sum() / len(series)

        if dist_type == "normal":
            return "none", "Already normally distributed — no transform needed"

        if dist_type == "binary":
            return "none", "Binary column — will be encoded separately"

        if dist_type == "zero_inflated":
            non_zero               = series[series != 0]
            non_zero_is_continuous = non_zero.nunique() > ORDINAL_MAX_UNIQUE
            sub_type               = "continuous" if non_zero_is_continuous else "discrete"
            return "none", (
                f"Zero-inflated — {zero_ratio:.0%} zeros. "
                f"Non-zero component is {sub_type} ({non_zero.nunique()} unique values). "
                f"Preprocessor must split into: binary zero-flag + conditional {sub_type} distribution."
            )

        if dist_type in ("log_normal", "exponential"):
            if series.min() >= 0:
                return "log", "Log transform will normalize this right-skewed distribution"

        if dist_type == "skewed":
            if skewness_severity == "mild":
                return "quantile", "Mild skew — quantile transform is more robust than sqrt for diffusion models"
            if skewness_severity in ("moderate", "severe"):
                if series.min() >= 0:
                    return "log", "Moderate/severe skew — log transform recommended"
                return "quantile", "Negative values present — quantile transform safer than log"

        if dist_type == "uniform":
            return "none", "Uniform distribution — diffusion model handles this well"

        if dist_type == "peaked":
            if skewness_severity in ("moderate", "severe"):
                if series.min() >= 0:
                    return "log", "Peaked + skewed distribution — log transform reduces skew and flattens spike"
                return "quantile", "Peaked + skewed with negatives — quantile transform safer"
            return "quantile", "Peaked distribution (leptokurtic) — quantile transform normalizes the spike"

        if dist_type == "bimodal":
            return "quantile", "Bimodal shape — quantile transform maps to uniform before model"

        return "quantile", "Unknown distribution shape — quantile transform as safe default"

    # ── Summary printer ───────────────────────────────────────────────────────

    def summary(self) -> None:
        if not self.profiles:
            self.profile()

        print("\n📊 Distribution Profiler Report")
        print("=" * 55)

        transform_icons = {
            "none"     : "✅",
            "log"      : "📐",
            "sqrt"     : "📏",
            "quantile" : "🔄",
        }

        dist_icons = {
            "normal"        : "🔔",
            "log_normal"    : "📈",
            "exponential"   : "🚀",
            "skewed"        : "↗️ ",
            "bimodal"       : "🏔️ ",
            "zero_inflated" : "0️⃣ ",
            "uniform"       : "▬ ",
            "binary"        : "⚡",
            "peaked"        : "🗻",
        }

        for p in self.profiles:
            dist_icon      = dist_icons.get(p.distribution_type, "❓")
            transform_icon = transform_icons.get(p.recommended_transform, "❓")

            print(f"\n📌 {p.column_name}")
            print(f"   distribution  : {dist_icon} {p.distribution_type}")
            print(f"   skewness      : {p.skewness_severity}")
            print(f"   normality     : {'✅ normal' if p.normality.is_normal else '❌ not normal'} (p={p.normality.p_value})")
            print(f"   outliers      : {p.outliers.count} ({p.outliers.ratio*100:.1f}%) — {p.outliers.note}")
            print(f"   transform     : {transform_icon} {p.recommended_transform}")
            print(f"   reason        : {p.transform_reason}")

        print("\n" + "=" * 55)

        # Transform summary at the end
        print("\n🛠️  Transform Plan:")
        for p in self.profiles:
            icon = transform_icons.get(p.recommended_transform, "❓")
            print(f"   {icon} {p.column_name:<20} → {p.recommended_transform}")