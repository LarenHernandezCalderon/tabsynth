"""
generator.py
─────────────────────────────────────────────────────────────────────────────
Phase 4 — The Generator.

Transforms the raw synthetic tensor (output of Phase 3) into a complete,
production-ready synthetic DataFrame with the exact same structure as the
original dataset.

This is the only public entry point of the generator/ package.
It orchestrates three internal modules in order:

    1. Preprocessor.inverse_transform(tensor)
       → Raw DataFrame: columns that were in the tensor, approximately typed
       → Problems: binary columns are floats, ordinals are floats, missing
         columns (excluded from tensor), potential values out of range

    2. TypeEnforcer.enforce(df_raw)
       → Types corrected: Survived → {0,1}, SibSp → int, Age clipped
       → Every numeric column has its real-world constraints applied

    3. ColumnBuilder.build(df_typed)
       → Excluded columns added: PassengerId, Name, Ticket, Cabin, ...
       → Each generated with its real strategy (sequential, Faker, regex)
       → Missing ratio preserved per column

    4. NullValidator.validate(df_complete)
       → Unexpected NaNs imputed (columns with real missing_ratio == 0)
       → Expected NaNs left intact (the model learned the pattern)
       → Dtypes cast to match the original dataset
       → Large null rate discrepancies flagged as warnings

    5. Final column reordering
       → Output DataFrame columns are in the exact same order as the original

Output (GeneratorOutput):
    synthetic_df     : pd.DataFrame  — complete, typed, ordered
    n_rows           : int
    n_cols           : int
    columns_match    : bool          — True if column list == original
    dtypes_match     : bool          — True if all dtypes match original
    null_rates       : dict[str, float]  — synthetic null rate per column
    quality_report   : dict          — full audit trail of every correction

Usage:
    gen    = Generator(profile, prep_output, original_df)
    result = gen.generate(synthetic_tensor, preprocessor)
    result.synthetic_df.to_csv("synthetic.csv", index=False)

Dependencies:
    pandas, numpy, torch

Part of:
    Synthetic Data Engine — Auto-Adaptive Diffusion System
"""

import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from analyzer.dataset_analyzer  import DatasetProfile
from preprocessor.preprocessor  import Preprocessor, PreprocessorOutput
from generator.type_enforcer    import TypeEnforcer,  EnforcementReport
from generator.column_builder   import ColumnBuilder,  BuildReport
from generator.null_validator   import NullValidator,  NullReport


# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass
class GeneratorOutput:
    """
    Full output of Phase 4.

    synthetic_df     : the complete synthetic DataFrame — same structure as original
    n_rows           : number of synthetic rows
    n_cols           : number of columns (should equal original n_cols)
    columns_match    : True if the column list exactly matches the original
    dtypes_match     : True if all dtypes match the original
    null_rates       : synthetic null rate per column {col: ratio}
    quality_report   : detailed audit of every correction made in Phase 4
    """
    synthetic_df   : pd.DataFrame
    n_rows         : int
    n_cols         : int
    columns_match  : bool
    dtypes_match   : bool
    null_rates     : dict
    quality_report : dict


# ── Main generator ────────────────────────────────────────────────────────────

class Generator:
    """
    Phase 4 orchestrator: tensor → complete synthetic DataFrame.

    Usage:
        gen    = Generator(profile, prep_output, original_df)
        result = gen.generate(synthetic_tensor, preprocessor)

    Args:
        profile      : DatasetProfile from Phase 1
        prep_output  : PreprocessorOutput from Phase 2
        original_df  : the original DataFrame (used for range/dtype reference)
        verbose      : print step-by-step progress
    """

    def __init__(
        self,
        profile     : DatasetProfile,
        prep_output : PreprocessorOutput,
        original_df : pd.DataFrame,
        verbose     : bool = True,
    ):
        self.profile     = profile
        self.prep_output = prep_output
        self.original_df = original_df
        self.verbose     = verbose

    def generate(
        self,
        synthetic_tensor : torch.Tensor,
        preprocessor     : Preprocessor,
    ) -> GeneratorOutput:
        """
        Transform a synthetic tensor into a complete DataFrame.

        Args:
            synthetic_tensor : torch.Tensor of shape (n_samples, n_features)
                               produced by TabDDPM.generate()
            preprocessor     : fitted Preprocessor instance (for inverse_transform)

        Returns:
            GeneratorOutput with the complete synthetic DataFrame
        """
        if self.verbose:
            print("\n" + "=" * 55)
            print("  Synthetic Data Engine — Generator (Phase 4)")
            print("=" * 55)

        n = synthetic_tensor.shape[0]

        # ── Step 1: Inverse transform (tensor → raw DataFrame) ─────────────
        if self.verbose:
            print("\nStep 1 — Inverse transform (tensor → raw DataFrame)")
            print("-" * 40)

        df_raw = preprocessor.inverse_transform(synthetic_tensor)

        if self.verbose:
            print(f"  Raw shape    : {df_raw.shape}")
            print(f"  Raw columns  : {list(df_raw.columns)}")

        # ── Step 2: Type enforcement ───────────────────────────────────────
        if self.verbose:
            print("\nStep 2 — Type enforcement")
            print("-" * 40)

        enforcer = TypeEnforcer(self.profile, self.original_df)
        df_typed, enf_report = enforcer.enforce(df_raw)

        if self.verbose:
            self._print_enforcement_report(enf_report)

        # ── Step 3: Column builder ─────────────────────────────────────────
        if self.verbose:
            print("\nStep 3 — Excluded column reconstruction")
            print("-" * 40)

        builder = ColumnBuilder(self.profile, self.prep_output, self.original_df)
        df_complete, build_report = builder.build(df_typed)

        if self.verbose:
            self._print_build_report(build_report)

        # ── Step 4: Null validation ────────────────────────────────────────
        if self.verbose:
            print("\nStep 4 — Null validation & dtype casting")
            print("-" * 40)

        validator = NullValidator(self.profile, self.original_df)
        df_final, null_report = validator.validate(df_complete)

        if self.verbose:
            self._print_null_report(null_report)

        # ── Step 5: Quality checks ─────────────────────────────────────────
        orig_cols     = list(self.original_df.columns)
        synth_cols    = list(df_final.columns)
        columns_match = orig_cols == synth_cols

        dtypes_match  = self._check_dtypes(df_final)

        null_rates = {
            col: float(df_final[col].isna().mean())
            for col in df_final.columns
        }

        quality_report = self._build_quality_report(
            enf_report, build_report, null_report,
            columns_match, dtypes_match, df_final,
        )

        # ── Summary ────────────────────────────────────────────────────────
        if self.verbose:
            self._print_summary(df_final, columns_match, dtypes_match, null_report)

        return GeneratorOutput(
            synthetic_df  = df_final,
            n_rows        = len(df_final),
            n_cols        = len(df_final.columns),
            columns_match = columns_match,
            dtypes_match  = dtypes_match,
            null_rates    = null_rates,
            quality_report= quality_report,
        )

    # ── Quality checks ────────────────────────────────────────────────────────

    def _check_dtypes(self, df: pd.DataFrame) -> bool:
        """
        Returns True if every column in df has the same dtype family as
        the corresponding column in the original DataFrame.

        'Same family' means: both numeric, both string, both bool — not
        necessarily identical dtype string (e.g. int32 vs int64 is fine).
        """
        for col in df.columns:
            if col not in self.original_df.columns:
                return False
            orig_is_num  = pd.api.types.is_numeric_dtype(self.original_df[col])
            synth_is_num = pd.api.types.is_numeric_dtype(df[col])
            if orig_is_num != synth_is_num:
                return False
        return True

    def _build_quality_report(
        self,
        enf_report     : EnforcementReport,
        build_report   : BuildReport,
        null_report    : NullReport,
        columns_match  : bool,
        dtypes_match   : bool,
        df_final       : pd.DataFrame,
    ) -> dict:
        return {
            "columns_match"          : columns_match,
            "dtypes_match"           : dtypes_match,
            "n_rows"                 : len(df_final),
            "n_cols"                 : len(df_final.columns),
            "type_enforcement": {
                "binary_cols_rounded"    : enf_report.binary_cols_rounded,
                "ordinal_cols_cast"      : enf_report.ordinal_cols_cast,
                "continuous_cols_clipped": enf_report.continuous_cols_clipped,
                "n_values_changed"       : enf_report.n_values_changed,
            },
            "column_builder": {
                "sequential_built" : build_report.sequential_built,
                "faker_built"      : build_report.faker_built,
                "regex_built"      : build_report.regex_built,
                "datetime_built"   : build_report.datetime_built,
                "skipped"          : build_report.skipped,
                "nulls_injected"   : build_report.nulls_injected,
            },
            "null_validation": {
                "unexpected_nulls_fixed" : null_report.unexpected_nulls_fixed,
                "null_rate_real"         : null_report.null_rate_real,
                "null_rate_synthetic"    : null_report.null_rate_synthetic,
                "warnings"               : null_report.null_rate_warnings,
                "dtype_casts"            : null_report.dtype_casts,
            },
        }

    # ── Verbose printers ──────────────────────────────────────────────────────

    def _print_enforcement_report(self, r: EnforcementReport) -> None:
        if r.binary_cols_rounded:
            print(f"  Binary rounded   : {r.binary_cols_rounded}")
        if r.ordinal_cols_cast:
            print(f"  Ordinal cast     : {r.ordinal_cols_cast}")
        if r.continuous_cols_clipped:
            print(f"  Continuous clipped: {r.continuous_cols_clipped}")
        if r.n_values_changed:
            for col, n in r.n_values_changed.items():
                print(f"    ✓ {col:<25} → {n} values corrected")
        if not any([r.binary_cols_rounded, r.ordinal_cols_cast,
                    r.continuous_cols_clipped, r.n_values_changed]):
            print("  No corrections needed")

    def _print_build_report(self, r: BuildReport) -> None:
        for col in r.sequential_built:
            print(f"  + {col:<25} → sequential_id  (1 … n)")
        for col in r.faker_built:
            print(f"  + {col:<25} → faker")
        for col in r.regex_built:
            print(f"  + {col:<25} → regex_pattern")
        for col in r.datetime_built:
            print(f"  + {col:<25} → datetime_range")
        for col, n in r.nulls_injected.items():
            print(f"    ~ {col:<23} → {n} nulls injected (real ratio preserved)")
        if r.skipped:
            print(f"  ✗ Skipped        : {r.skipped}")

    def _print_null_report(self, r: NullReport) -> None:
        if r.unexpected_nulls_fixed:
            for col, n in r.unexpected_nulls_fixed.items():
                print(f"  ✓ {col:<25} → {n} unexpected NaN(s) imputed")
        if r.null_rate_warnings:
            for col in r.null_rate_warnings:
                real  = r.null_rate_real.get(col, 0)
                synth = r.null_rate_synthetic.get(col, 0)
                print(f"  ⚠ {col:<25} → null rate gap: real={real:.1%} synth={synth:.1%}")
        if r.dtype_casts:
            for col, dtype in r.dtype_casts.items():
                print(f"  ↻ {col:<25} → cast to {dtype}")
        if not any([r.unexpected_nulls_fixed, r.null_rate_warnings, r.dtype_casts]):
            print("  All nulls valid — no corrections needed")

    def _print_summary(
        self,
        df_final      : pd.DataFrame,
        columns_match : bool,
        dtypes_match  : bool,
        null_report   : NullReport,
    ) -> None:
        print("\n" + "=" * 55)
        print("  GENERATOR SUMMARY")
        print("=" * 55)
        print(f"\n  Output shape   : {df_final.shape}")
        print(f"  Columns match  : {'✅' if columns_match else '❌'}")
        print(f"  Dtypes match   : {'✅' if dtypes_match  else '⚠️  (check quality_report)'}")

        if null_report.null_rate_warnings:
            print(f"\n  ⚠ Null rate warnings ({len(null_report.null_rate_warnings)} columns):")
            for col in null_report.null_rate_warnings:
                real  = null_report.null_rate_real.get(col, 0)
                synth = null_report.null_rate_synthetic.get(col, 0)
                print(f"    {col:<25} real={real:.1%}  synth={synth:.1%}")

        print("\n  Null rates per column:")
        print(f"  {'Column':<25} {'Real':>8}  {'Synth':>8}")
        print(f"  {'-'*24} {'-'*8}  {'-'*8}")
        for col in df_final.columns:
            real  = null_report.null_rate_real.get(col, 0.0)
            synth = null_report.null_rate_synthetic.get(col, 0.0)
            flag  = " ⚠" if col in null_report.null_rate_warnings else ""
            print(f"  {col:<25} {real:>8.1%}  {synth:>8.1%}{flag}")

        print("\n" + "=" * 55)