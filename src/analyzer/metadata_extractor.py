"""
metadata_extractor.py
─────────────────────────────────────────────────────────────────────────────
Responsible for extracting safe, privacy-preserving metadata from every column
in a structured dataset (CSV or Excel).

This module is Layer 1 of the analysis pipeline. It inspects each column
locally — no data leaves the environment at this stage.

Core principle:
    Real data values are NEVER stored, logged, or exposed.
    Only aggregated statistics and structural patterns are extracted.

What it produces (ColumnMetadata) per column:
    - Detected type     : numerical, categorical, ordinal, binary, datetime...
    - Missing ratio     : proportion of null values
    - Numerical stats   : mean, std, skewness, percentiles (for numeric columns)
    - Distribution      : value frequencies (only for low-cardinality columns)
    - String metadata   : structural signals only, never actual values
                          (for high-cardinality strings → sent to Layer 2)

Columns flagged as 'unknown_string' cannot be resolved locally and are
forwarded to the SemanticAnalyzer (semantic_analyzer.py) for GPT-based
classification using only the safe metadata payload built here.

Dependencies:
    pandas, numpy, dataclasses

Part of:
    Synthetic Tabular Data Engine — Auto-Adaptive Diffusion System
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Safety thresholds ────────────────────────────────────────────────────────
CATEGORICAL_UNIQUE_RATIO    = 0.05   # below this → low cardinality → safe to send values
ID_LIKE_UNIQUE_RATIO        = 0.95   # above this → almost certainly an ID
ORDINAL_MAX_UNIQUE          = 15     # integers with few unique values → ordinal
SAMPLE_LENGTHS_COUNT        = 20     # how many length samples to collect


# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class NumericalStats:
    mean:        float
    std:         float
    min:         float
    max:         float
    skewness:    float
    kurtosis:    float
    percentiles: dict[str, float]   # "25", "50", "75"


@dataclass
class StringMetadata:
    """Safe string metadata — no real values ever stored here."""
    avg_char_length:          float
    avg_word_count:           float
    unique_ratio:             float
    starts_uppercase_ratio:   float
    has_numbers_ratio:        float
    has_special_chars_ratio:  float
    sample_lengths:           list[int]   # lengths only, never values


@dataclass
class ColumnMetadata:
    column_name:    str
    dtype:          str
    detected_type:  str          # Layer 1 type (may be "unknown_string")
    missing_ratio:  float
    total_rows:     int
    unique_count:   int

    # Populated depending on detected_type
    stats:           Optional[NumericalStats]  = None   # numerics
    string_metadata: Optional[StringMetadata]  = None   # high-cardinality strings
    distribution:    Optional[dict]            = None   # low-cardinality categoricals


# ── Main extractor ────────────────────────────────────────────────────────────

class MetadataExtractor:
    """
    Extracts safe, privacy-preserving metadata from every column in a DataFrame.

    Real data values are NEVER stored or exposed — only aggregated statistics,
    structural patterns, and frequency distributions for low-cardinality columns.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.metadata: list[ColumnMetadata] = []

    def extract(self) -> list[ColumnMetadata]:
        """Run extraction on all columns. Returns list of ColumnMetadata."""
        self.metadata = []

        for col in self.df.columns:
            col_meta = self._extract_column(col)
            self.metadata.append(col_meta)

        return self.metadata

    # ── Column dispatcher ─────────────────────────────────────────────────────

    def _extract_column(self, col: str) -> ColumnMetadata:
        series       = self.df[col]
        series_clean = series.dropna()

        total_rows    = len(series)
        unique_count  = series_clean.nunique()
        missing_ratio = round(series.isna().sum() / total_rows, 4)
        unique_ratio  = unique_count / len(series_clean) if len(series_clean) > 0 else 0

        # ── Detect Layer 1 type ───────────────────────────────────────────────
        detected_type = self._detect_type(series_clean, unique_ratio)

        # ── Base metadata object ──────────────────────────────────────────────
        meta = ColumnMetadata(
            column_name   = col,
            dtype         = str(series.dtype),
            detected_type = detected_type,
            missing_ratio = missing_ratio,
            total_rows    = total_rows,
            unique_count  = unique_count,
        )

        # ── Enrich based on type ──────────────────────────────────────────────
        if detected_type in ("numerical_continuous", "ordinal", "binary") and \
           pd.api.types.is_numeric_dtype(series_clean):
            meta.stats = self._extract_numerical_stats(series_clean)

        elif detected_type == "categorical":
            # Low cardinality → safe to expose value distribution
            meta.distribution = self._extract_distribution(series_clean)

        elif detected_type == "unknown_string":
            # High cardinality string → only structural metadata, never values
            meta.string_metadata = self._extract_string_metadata(series_clean)

        return meta

    # ── Type detector (Layer 1) ───────────────────────────────────────────────

    def _detect_type(self, series: pd.Series, unique_ratio: float) -> str:
        """
        Detection order matters:
        1. Empty guard
        2. Datetime (before numeric — some dates look like ints)
        3. Numeric types (BEFORE id_like — floats are naturally near-unique)
        4. id_like (only meaningful for strings/IDs, not continuous numbers)
        5. String/categorical resolution
        """
        if len(series) == 0:
            return "empty"

        # ── Datetime first ────────────────────────────────────────────────────
        if self._is_datetime(series):
            return "datetime"

        # ── Numeric: check BEFORE id_like ─────────────────────────────────────
        if pd.api.types.is_float_dtype(series):
            return "numerical_continuous"

        if pd.api.types.is_integer_dtype(series):
            if series.nunique() == 2:
                return "binary"
            if series.nunique() <= ORDINAL_MAX_UNIQUE:
                return "ordinal"
            if unique_ratio >= ID_LIKE_UNIQUE_RATIO:
                return "id_like"
            return "numerical_continuous"

        # ── String columns: handles both old 'object' and new 'str' dtype ─────
        # ── String columns: handles both old 'object' and new 'str' dtype ─────
        is_string = series.dtype == object or pd.api.types.is_string_dtype(series)
        if is_string:
            if series.nunique() == 2:
                return "binary"
            # ↓ removed id_like check for strings — GPT decides this in Layer 2
            if unique_ratio < CATEGORICAL_UNIQUE_RATIO:
                return "categorical"
            return "unknown_string"    # → ALL high cardinality strings go to GPT

        return "unknown"

    # ── Numerical stats ───────────────────────────────────────────────────────

    def _extract_numerical_stats(self, series: pd.Series) -> NumericalStats:
        percs = series.quantile([0.25, 0.50, 0.75])

        return NumericalStats(
            mean        = round(float(series.mean()), 4),
            std         = round(float(series.std()), 4),
            min         = round(float(series.min()), 4),
            max         = round(float(series.max()), 4),
            skewness    = round(float(series.skew()), 4),
            kurtosis    = round(float(series.kurt()), 4),
            percentiles = {
                "25": round(float(percs[0.25]), 4),
                "50": round(float(percs[0.50]), 4),
                "75": round(float(percs[0.75]), 4),
            }
        )

    # ── Categorical distribution ──────────────────────────────────────────────

    def _extract_distribution(self, series: pd.Series) -> dict:
        """Frequency ratio per category. Safe: only works on low cardinality."""
        freq = series.value_counts(normalize=True)
        return {str(k): round(float(v), 4) for k, v in freq.items()}

    # ── String metadata (privacy-safe) ────────────────────────────────────────

    def _extract_string_metadata(self, series: pd.Series) -> StringMetadata:
        """
        Extracts only structural patterns from string columns.
        No actual values are stored — only aggregated signals.
        """
        lengths     = series.str.len().dropna()
        word_counts = series.str.split().str.len().dropna()

        # Pattern ratios
        starts_upper   = series.str.match(r'^[A-Z]').sum() / len(series)
        has_numbers    = series.str.contains(r'\d').sum() / len(series)
        has_special    = series.str.contains(r'[^a-zA-Z0-9\s]').sum() / len(series)

        # Sample of lengths only — never sample of values
        sample_lengths = lengths.head(SAMPLE_LENGTHS_COUNT).astype(int).tolist()

        return StringMetadata(
            avg_char_length         = round(float(lengths.mean()), 2),
            avg_word_count          = round(float(word_counts.mean()), 2),
            unique_ratio            = round(series.nunique() / len(series), 4),
            starts_uppercase_ratio  = round(float(starts_upper), 4),
            has_numbers_ratio       = round(float(has_numbers), 4),
            has_special_chars_ratio = round(float(has_special), 4),
            sample_lengths          = sample_lengths,
        )

    # ── Datetime helper ───────────────────────────────────────────────────────

    def _is_datetime(self, series: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if series.dtype == object:
            try:
                pd.to_datetime(series.head(20), infer_datetime_format=True)
                return True
            except Exception:
                return False
        return False

    # ── Summary printer ───────────────────────────────────────────────────────

    def summary(self) -> None:
        if not self.metadata:
            self.extract()

        print("\n🔍 Metadata Extraction Report")
        print("=" * 55)

        for meta in self.metadata:
            print(f"\n📌 {meta.column_name}")
            print(f"   dtype         : {meta.dtype}")
            print(f"   detected_type : {meta.detected_type}")
            print(f"   missing       : {meta.missing_ratio * 100:.1f}%")
            print(f"   unique values : {meta.unique_count}")

            if meta.stats:
                s = meta.stats
                print(f"   mean / std    : {s.mean} / {s.std}")
                print(f"   min / max     : {s.min} / {s.max}")
                print(f"   skewness      : {s.skewness}")
                print(f"   p25/p50/p75   : {s.percentiles}")

            if meta.distribution:
                print(f"   distribution  : {meta.distribution}")

            if meta.string_metadata:
                sm = meta.string_metadata
                print(f"   avg length    : {sm.avg_char_length} chars")
                print(f"   avg words     : {sm.avg_word_count}")
                print(f"   starts upper  : {sm.starts_uppercase_ratio * 100:.1f}%")
                print(f"   has numbers   : {sm.has_numbers_ratio * 100:.1f}%")
                print(f"   has special   : {sm.has_special_chars_ratio * 100:.1f}%")
                print(f"   len samples   : {sm.sample_lengths}")

        print("\n" + "=" * 55)
        flagged = [m for m in self.metadata if m.detected_type == "unknown_string"]
        if flagged:
            print(f"\n⚠️  {len(flagged)} column(s) flagged for semantic analysis (Layer 2):")
            for m in flagged:
                print(f"   → {m.column_name}")