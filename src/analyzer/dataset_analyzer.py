"""
dataset_analyzer.py
─────────────────────────────────────────────────────────────────────────────
Layer 5 — Pipeline Orchestrator.

Coordinates all analysis layers in order and consolidates their outputs
into a single unified DatasetProfile object that the preprocessor consumes.

Pipeline execution order:
    Layer 1 → MetadataExtractor     : column type detection
    Layer 2 → SemanticAnalyzer      : GPT semantic classification + PII detection
    Layer 2b→ RegexMiner            : local regex refinement for 'code' columns
    Layer 3 → DistributionProfiler  : distribution shape + transform plan
    Layer 4 → CorrelationAnalyzer   : pairwise correlation analysis

What it produces (DatasetProfile):
    - One ColumnProfile per column with all layer outputs merged
    - Critical correlation pairs the diffusion model must preserve
    - Numeric Pearson correlation matrix
    - Lists of generatable vs skipped columns
    - Dataset-level statistics
    - GDPR summary: PII columns flagged with category

ColumnProfile fields:
    column_name             : original column name
    detected_type           : Layer 1 type (numerical_continuous, binary, ...)
    semantic_type           : Layer 2 GPT classification (full_name, code, ...)
    locale                  : Faker locale detected by GPT (es_ES, en_US, ...)
    regex_pattern           : refined regex from RegexMiner (or GPT fallback)
    regex_coverage          : fraction of values matched by the regex (0.0–1.0)
    regex_from_miner        : True if regex was refined by RegexMiner locally
    distribution_type       : Layer 3 distribution shape
    skewness_severity       : none | mild | moderate | severe
    recommended_transform   : none | log | sqrt | quantile
    generation_strategy     : how to generate this column in the synthesizer
    pii_detected            : True if column contains Personal Identifiable Info
    pii_category            : direct_identifier | quasi_identifier | non_pii
    should_skip             : True if column cannot/should not be generated
    skip_reason             : explanation of why column is skipped
    missing_ratio           : proportion of null values

Skip rules (automatic, no user intervention):
    id_like                 → regenerated as sequential integers
    unknown_string + code without regex → skipped with warning
    free_text               → skipped (not generatable in v1)
    unknown                 → skipped with warning
    high missing + no pattern → skipped with warning

Dependencies:
    pandas, dataclasses, os

Part of:
    Synthetic Tabular Data Engine — Auto-Adaptive Diffusion System
"""

import os
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from analyzer.metadata_extractor    import MetadataExtractor,    ColumnMetadata
from analyzer.semantic_analyzer     import (
    SemanticAnalyzer, SemanticResult, PII_CATEGORY_LABELS
)
from analyzer.distribution_profiler import DistributionProfiler, DistributionProfile
from analyzer.correlation_analyzer  import CorrelationAnalyzer,  CorrelationPair
from analyzer.regex_miner           import RegexMiner


# ── Categorical PII map ──────────────────────────────────────────────────────
# Columns that never reach Layer 2 (low-cardinality categoricals / binaries)
# but carry clear PII semantics based on their column name.
# Key: lowercase column name fragment → (pii_detected, pii_category)
# Matched via substring so "customer_gender", "gender_code", etc. all match.
CATEGORICAL_PII_KEYWORDS: dict[str, tuple[bool, str]] = {
    # Quasi-identifiers — re-identification risk when combined (GDPR Art. 4)
    "gender"        : (True, "quasi_identifier"),
    "sex"           : (True, "quasi_identifier"),
    "race"          : (True, "quasi_identifier"),
    "ethnicity"     : (True, "quasi_identifier"),
    "nationality"   : (True, "quasi_identifier"),
    "country"       : (True, "quasi_identifier"),
    "native"        : (True, "quasi_identifier"),   # native-country
    "marital"       : (True, "quasi_identifier"),
    "age"           : (True, "quasi_identifier"),
    "birth"         : (True, "quasi_identifier"),
    "zipcode"       : (True, "quasi_identifier"),
    "postal"        : (True, "quasi_identifier"),
    # Sensitive — GDPR Art. 9 special categories
    "religion"      : (True, "sensitive"),
    "health"        : (True, "sensitive"),
    "diagnosis"     : (True, "sensitive"),
    "disability"    : (True, "sensitive"),
    # NOTE: "income", "salary", "occupation", "relationship" deliberately excluded —
    # these are model features / census labels, not PII in the GDPR sense
    # when expressed as broad categorical buckets (e.g. <=50K, Husband, Prof-specialty)
}


# ── Skip rules ────────────────────────────────────────────────────────────────
SKIP_TYPES = {"free_text", "unknown"}
ID_TYPES   = {"id_like"}

HIGH_MISSING_SKIP_THRESHOLD = 0.80


# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class ColumnProfile:
    """
    Unified profile for a single column — merges all layer outputs.
    This is the primary input to the Preprocessor.
    """
    column_name             : str
    detected_type           : str
    semantic_type           : Optional[str]  = None
    locale                  : Optional[str]  = None

    # Regex — may be refined by RegexMiner after GPT classification
    regex_pattern           : Optional[str]  = None
    regex_coverage          : float          = 0.0    # fraction of values matched
    regex_from_miner        : bool           = False  # True = refined locally

    distribution_type       : Optional[str]  = None
    skewness_severity       : Optional[str]  = None
    recommended_transform   : Optional[str]  = None
    generation_strategy     : Optional[str]  = None

    # GDPR / PII
    pii_detected            : bool           = False
    pii_category            : str            = "non_pii"

    should_skip             : bool           = False
    skip_reason             : Optional[str]  = None
    is_id                   : bool           = False
    missing_ratio           : float          = 0.0


@dataclass
class DatasetProfile:
    """
    Complete dataset-level profile produced by the analysis pipeline.
    Primary input to the Preprocessor and Diffusion Model trainer.
    """
    columns             : list[ColumnProfile]
    critical_pairs      : list[CorrelationPair]
    all_pairs           : list[CorrelationPair]
    correlation_matrix  : Optional[pd.DataFrame]
    n_rows              : int
    n_cols              : int
    generatable_columns : list[str]
    skipped_columns     : list[str]
    id_columns          : list[str]
    pii_columns         : list[str]      # columns flagged as PII

    def get_column(self, name: str) -> Optional[ColumnProfile]:
        for col in self.columns:
            if col.column_name == name:
                return col
        return None


# ── Main orchestrator ─────────────────────────────────────────────────────────

class DatasetAnalyzer:
    """
    Orchestrates the full analysis pipeline and produces a DatasetProfile.

    Usage:
        analyzer = DatasetAnalyzer(df, openai_api_key)
        profile  = analyzer.analyze()
    """

    def __init__(
        self,
        df             : pd.DataFrame,
        openai_api_key : str,
        gpt_model      : str  = "gpt-4o-mini",
        verbose        : bool = True,
    ):
        self.df        = df
        self.api_key   = openai_api_key
        self.gpt_model = gpt_model
        self.verbose   = verbose
        self.profile   : Optional[DatasetProfile] = None

        self._metadata      : list[ColumnMetadata]      = []
        self._semantic      : list[SemanticResult]      = []
        self._distributions : list[DistributionProfile] = []
        self._pairs         : list[CorrelationPair]     = []
        self._matrix        : Optional[pd.DataFrame]    = None

    def analyze(self) -> DatasetProfile:
        if self.verbose:
            print("\n" + "=" * 55)
            print("  Synthetic Data Engine — Analysis Pipeline")
            print("=" * 55)

        self._run_layer1()
        self._run_layer2()
        self._run_layer2b()   # RegexMiner — local refinement
        self._run_layer3()
        self._run_layer4()

        self.profile = self._build_profile()

        if self.verbose:
            self.summary()

        return self.profile

    # ── Layer runners ─────────────────────────────────────────────────────────

    def _run_layer1(self) -> None:
        if self.verbose:
            print("\nLayer 1 — Metadata Extraction")
            print("-" * 40)
        extractor      = MetadataExtractor(self.df)
        self._metadata = extractor.extract()
        if self.verbose:
            extractor.summary()

    def _run_layer2(self) -> None:
        if self.verbose:
            print("\nLayer 2 — Semantic Analysis + PII Detection")
            print("-" * 40)
        analyzer       = SemanticAnalyzer(api_key=self.api_key, model=self.gpt_model)
        self._semantic = analyzer.analyze(self._metadata)
        if self.verbose:
            analyzer.summary()

    def _run_layer2b(self) -> None:
        """
        Layer 2b — Local Regex Refinement (RegexMiner).

        Runs only on columns classified as 'code' by GPT.
        Analyzes actual values locally (no data leaves the environment)
        to infer precise regex patterns — replacing the GPT rough estimate.
        """
        code_results = [r for r in self._semantic if r.semantic_type == "code"]

        if not code_results:
            return

        if self.verbose:
            print(f"\nLayer 2b — Regex Refinement ({len(code_results)} code column(s))")
            print("-" * 40)

        miner = RegexMiner()

        for result in code_results:
            col_name  = result.column_name
            series    = self.df[col_name].dropna().astype(str)
            gpt_regex = result.regex_pattern

            mine_result = miner.mine(col_name, series, gpt_regex=gpt_regex)

            # Update the SemanticResult in-place with the refined regex
            result.regex_pattern = mine_result.regex

            if self.verbose:
                source = "GPT fallback" if mine_result.fallback else "RegexMiner ✅"
                print(f"\n📐 {col_name}")
                print(f"   gpt_regex     : {gpt_regex or 'none'}")
                print(f"   mined_regex   : {mine_result.regex}")
                print(f"   coverage      : {mine_result.coverage:.0%}")
                print(f"   source        : {source}")
                print(f"   patterns      : {mine_result.patterns}")
                print(f"   notes         : {mine_result.notes}")

            # Store mine metadata on the result object for profile building
            result._mine_coverage     = mine_result.coverage
            result._mine_from_miner   = not mine_result.fallback

    def _run_layer3(self) -> None:
        if self.verbose:
            print("\nLayer 3 — Distribution Profiling")
            print("-" * 40)
        profiler            = DistributionProfiler(self.df, self._metadata)
        self._distributions = profiler.profile()
        if self.verbose:
            profiler.summary()

    def _run_layer4(self) -> None:
        if self.verbose:
            print("\nLayer 4 — Correlation Analysis")
            print("-" * 40)
        correlator   = CorrelationAnalyzer(self.df, self._metadata)
        self._pairs  = correlator.analyze()
        self._matrix = correlator.matrix
        if self.verbose:
            correlator.summary()

    # ── Profile builder ───────────────────────────────────────────────────────

    def _build_profile(self) -> DatasetProfile:
        semantic_map = {r.column_name: r for r in self._semantic}
        dist_map     = {p.column_name: p for p in self._distributions}

        column_profiles = []

        for meta in self._metadata:
            col  = meta.column_name
            sem  = semantic_map.get(col)
            dist = dist_map.get(col)

            profile = ColumnProfile(
                column_name   = col,
                detected_type = meta.detected_type,
                missing_ratio = meta.missing_ratio,
            )

            if sem:
                profile.semantic_type       = sem.semantic_type
                profile.locale              = sem.locale
                profile.regex_pattern       = sem.regex_pattern
                profile.generation_strategy = sem.generation_strategy
                profile.pii_detected        = sem.pii_detected
                profile.pii_category        = sem.pii_category

                # Regex miner metadata (set by _run_layer2b via duck attributes)
                profile.regex_coverage   = getattr(sem, "_mine_coverage",   0.0)
                profile.regex_from_miner = getattr(sem, "_mine_from_miner", False)

            if dist:
                profile.distribution_type     = dist.distribution_type
                profile.skewness_severity     = dist.skewness_severity
                profile.recommended_transform = dist.recommended_transform

            if not profile.generation_strategy:
                profile.generation_strategy = self._infer_strategy(meta, sem)

            # Apply categorical PII rules for columns that bypass Layer 2
            # (low-cardinality categoricals and binaries never reach GPT)
            if not profile.pii_detected:
                self._apply_categorical_pii(profile)

            self._apply_skip_rules(profile, meta, sem)

            column_profiles.append(profile)

        generatable = [
            p.column_name for p in column_profiles
            if not p.should_skip and not p.is_id
        ]
        skipped = [
            p.column_name for p in column_profiles
            if p.should_skip
        ]
        id_cols = [
            p.column_name for p in column_profiles
            if p.is_id
        ]
        pii_cols = [
            p.column_name for p in column_profiles
            if p.pii_detected
        ]
        critical_pairs = [
            p for p in self._pairs
            if p.strength in ("strong", "very_strong")
        ]

        return DatasetProfile(
            columns             = column_profiles,
            critical_pairs      = critical_pairs,
            all_pairs           = self._pairs,
            correlation_matrix  = self._matrix,
            n_rows              = len(self.df),
            n_cols              = len(self.df.columns),
            generatable_columns = generatable,
            skipped_columns     = skipped,
            id_columns          = id_cols,
            pii_columns         = pii_cols,
        )

    # ── Strategy inferrer ─────────────────────────────────────────────────────

    def _infer_strategy(
        self,
        meta : ColumnMetadata,
        sem  : Optional[SemanticResult],
    ) -> Optional[str]:
        t = meta.detected_type
        if t == "numerical_continuous" : return "diffusion_model"
        if t == "ordinal"              : return "diffusion_model"
        if t == "binary"               : return "diffusion_model"
        if t == "categorical"          : return "sample_distribution"
        if t == "datetime"             : return "datetime_range"
        if t == "id_like"              : return "sequential_id"
        return None

    # ── Categorical PII classifier ───────────────────────────────────────────

    def _apply_categorical_pii(self, profile: ColumnProfile) -> None:
        """
        Applies PII classification to categorical/binary columns that never
        reach Layer 2 (GPT). Matches column name against known PII keywords.
        Case-insensitive substring match — handles prefixes, suffixes, hyphens.
        """
        col_lower = profile.column_name.lower().replace("-", "_").replace(" ", "_")
        for keyword, (pii_detected, pii_category) in CATEGORICAL_PII_KEYWORDS.items():
            if keyword in col_lower:
                profile.pii_detected = pii_detected
                profile.pii_category = pii_category
                break

    # ── Skip rule engine ──────────────────────────────────────────────────────

    def _apply_skip_rules(
        self,
        profile : ColumnProfile,
        meta    : ColumnMetadata,
        sem     : Optional[SemanticResult],
    ) -> None:
        if meta.detected_type == "id_like":
            profile.is_id               = True
            profile.generation_strategy = "sequential_id"
            return

        if sem and sem.semantic_type in SKIP_TYPES:
            profile.should_skip = True
            profile.skip_reason = f"Semantic type '{sem.semantic_type}' is not generatable in v1"
            return

        if sem and sem.semantic_type == "code" and not sem.regex_pattern:
            profile.should_skip = True
            profile.skip_reason = "Code column — neither GPT nor RegexMiner could infer a pattern"
            return

        if (meta.missing_ratio >= HIGH_MISSING_SKIP_THRESHOLD
                and not profile.generation_strategy):
            profile.should_skip = True
            profile.skip_reason = (
                f"High missing ratio ({meta.missing_ratio:.0%}) "
                f"and no generation strategy could be inferred"
            )
            return

    # ── Summary printer ───────────────────────────────────────────────────────

    def summary(self) -> None:
        if not self.profile:
            print("No profile available — run analyze() first.")
            return

        p = self.profile

        print("\n" + "=" * 55)
        print("  DATASET PROFILE SUMMARY")
        print("=" * 55)

        print(f"\n  Dataset size        : {p.n_rows} rows x {p.n_cols} columns")
        print(f"  Generatable columns : {len(p.generatable_columns)}  {p.generatable_columns}")
        print(f"  Skipped columns     : {len(p.skipped_columns)}  {p.skipped_columns}")
        print(f"  ID columns          : {len(p.id_columns)}  {p.id_columns}")
        print(f"  Critical pairs      : {len(p.critical_pairs)}")

        # ── GDPR / PII summary ─────────────────────────────────────────────
        print("\n" + "-" * 55)
        print("  GDPR / PII REPORT")
        print("-" * 55)

        if p.pii_columns:
            print(f"\n  🔴 {len(p.pii_columns)} PII column(s) detected — generation via Faker (no real data used)\n")
            for col in p.columns:
                if col.pii_detected:
                    label = PII_CATEGORY_LABELS.get(col.pii_category, col.pii_category)
                    print(f"  ⚠️  {col.column_name:<20} [{col.pii_category}]")
                    print(f"      {label}")
                    print(f"      strategy: {col.generation_strategy}")
                    if col.locale:
                        print(f"      locale  : {col.locale}")
        else:
            print("\n  🟢 No PII columns detected.")

        # ── Per-column full detail ─────────────────────────────────────────
        print("\n" + "-" * 55)
        print("  COLUMN PLAN")
        print("-" * 55)

        for col in p.columns:
            if col.is_id:
                status = "ID"
            elif col.should_skip:
                status = "SKIP"
            else:
                status = "GENERATE"

            pii_tag = " 🔴PII" if col.pii_detected else ""

            print(f"\n  Column          : {col.column_name}{pii_tag}")
            print(f"  Status          : {status}")
            print(f"  detected_type   : {col.detected_type}")
            print(f"  semantic_type   : {col.semantic_type or 'n/a'}")
            print(f"  missing_ratio   : {col.missing_ratio * 100:.1f}%")
            print(f"  distribution    : {col.distribution_type or 'n/a'}")
            print(f"  skewness        : {col.skewness_severity or 'n/a'}")
            print(f"  transform       : {col.recommended_transform or 'none'}")
            print(f"  strategy        : {col.generation_strategy or 'n/a'}")
            print(f"  pii_detected    : {col.pii_detected}  [{col.pii_category}]")

            if col.locale:
                print(f"  locale          : {col.locale}")
            if col.regex_pattern:
                source = "RegexMiner" if col.regex_from_miner else "GPT"
                print(f"  regex_pattern   : {col.regex_pattern}")
                print(f"  regex_source    : {source}  (coverage: {col.regex_coverage:.0%})")
            if col.should_skip and col.skip_reason:
                print(f"  skip_reason     : {col.skip_reason}")

        # ── Correlation pairs ──────────────────────────────────────────────
        print("\n" + "-" * 55)
        print("  CORRELATION ANALYSIS — ALL PAIRS")
        print("-" * 55)

        strength_order  = ["very_strong", "strong", "moderate", "weak"]
        strength_labels = {
            "very_strong" : "VERY STRONG  (score >= 0.60)",
            "strong"      : "STRONG       (score 0.40 - 0.59)",
            "moderate"    : "MODERATE     (score 0.20 - 0.39)",
            "weak"        : "WEAK         (score < 0.20)",
        }

        for strength in strength_order:
            group = [pair for pair in p.all_pairs if pair.strength == strength]
            if not group:
                continue
            print(f"\n  {strength_labels[strength]}")
            print(f"  {'Column A':<15} {'Column B':<15} {'Method':<12} {'Score':>6}  Direction")
            print(f"  {'-'*14} {'-'*14} {'-'*11} {'-'*6}  {'-'*9}")
            for pair in group:
                print(
                    f"  {pair.col_a:<15} {pair.col_b:<15} "
                    f"{pair.method:<12} {pair.score:>6.3f}  {pair.direction}"
                )

        # ── Numeric correlation matrix ─────────────────────────────────────
        if p.correlation_matrix is not None:
            print("\n" + "-" * 55)
            print("  NUMERIC CORRELATION MATRIX (Pearson)")
            print("-" * 55)
            print()
            print(p.correlation_matrix.round(3).to_string())

        print("\n" + "=" * 55 + "\n")