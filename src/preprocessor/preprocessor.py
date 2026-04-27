"""
preprocessor.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 — Adaptive Preprocessor (orchestrator).

Public API — identical to the original monolithic preprocessor:
    prep   = Preprocessor(df, profile)
    output = prep.fit_transform()
    df_raw = prep.inverse_transform(synthetic_tensor)

Internally delegates each step to a dedicated module:
    Step 0 — NullNormalizer          (steps/step0_null_normalizer.py)
    Step 1 — ColumnFilter            (steps/step1_column_filter.py)
    Step 2 — MissingHandler          (steps/step2_missing_handler.py)
    Step 3 — ZeroInflatedSplitter    (steps/step3_zero_inflated.py)
    Step 4 — NumericTransformer      (steps/step4_numeric_transform.py)
    Step 5 — CategoricalEncoder      (steps/step5_categorical_encoder.py)
    Step 6 — MinMaxNormalizer        (steps/step6_normalizer.py)
    Step 7 — TensorBuilder           (steps/step7_tensor_builder.py)
    Inverse — InverseTransformer     (inverse/inverter.py)
"""

from __future__ import annotations

import pandas as pd
import torch
from typing import Optional

from analyzer.dataset_analyzer import DatasetProfile
from preprocessor.contracts import PreprocessorOutput

from preprocessor.steps.step0_null_normalizer    import NullNormalizer
from preprocessor.steps.step1_column_filter      import ColumnFilter
from preprocessor.steps.step2_missing_handler    import MissingHandler
from preprocessor.steps.step3_zero_inflated      import ZeroInflatedSplitter
from preprocessor.steps.step4_numeric_transform  import NumericTransformer
from preprocessor.steps.step5_categorical_encoder import CategoricalEncoder
from preprocessor.steps.step6_normalizer          import MinMaxNormalizer
from preprocessor.steps.step7_tensor_builder      import TensorBuilder
from preprocessor.inverse.inverter                import InverseTransformer


class Preprocessor:
    """
    Adaptive preprocessor — orchestrates all 8 steps.

    Usage:
        prep   = Preprocessor(df, profile)
        output = prep.fit_transform()
        df_raw = prep.inverse_transform(synthetic_tensor)
    """

    def __init__(
        self,
        df      : pd.DataFrame,
        profile : DatasetProfile,
        verbose : bool = True,
    ):
        self.df      = df.copy()
        self.profile = profile
        self.verbose = verbose
        self.output  : Optional[PreprocessorOutput] = None

        # Step instances — populated during fit_transform
        self._filter   : Optional[ColumnFilter]         = None
        self._inverter : Optional[InverseTransformer]   = None

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(self) -> PreprocessorOutput:
        """Run the full 8-step pipeline and return PreprocessorOutput."""

        if self.verbose:
            print("\n" + "=" * 55)
            print("  Synthetic Data Engine — Preprocessor (Phase 2)")
            print("=" * 55)

        # Step 0 — Masked null normalisation
        s0      = NullNormalizer(self.verbose)
        df      = s0.fit_transform(self.df)

        # Step 1 — Column filtering
        self._filter = ColumnFilter(self.profile, self.df, self.verbose)
        df           = self._filter.fit_transform(df)

        # Step 2 — Missing indicators
        s2 = MissingHandler(self.verbose)
        df = s2.fit_transform(df)

        # Step 3 — Zero-inflated split
        s3 = ZeroInflatedSplitter(self.profile, self.verbose)
        df = s3.fit_transform(df)

        # Step 4 — Numeric transforms
        s4 = NumericTransformer(self.profile, s3.zero_inflated_map, self.verbose)
        df = s4.fit_transform(df)

        # Step 5 — Categorical encoding
        s5 = CategoricalEncoder(self.verbose)
        df = s5.fit_transform(df)

        # Step 6 — MinMax normalisation
        s6 = MinMaxNormalizer(self.verbose)
        df = s6.fit_transform(df)

        # Step 7 — TensorDataset construction
        s7 = TensorBuilder(
            profile               = self.profile,
            zero_inflated_map     = s3.zero_inflated_map,
            quantile_transformers = s4.quantile_transformers,
            label_encoders        = s5.label_encoders,
            minmax_scalers        = s6.minmax_scalers,
            excluded              = self._filter.excluded,
            redundant_pairs       = self._filter.redundant_pairs,
            imputation_values     = s2.imputation_values,
            verbose               = self.verbose,
        )
        output = s7.build(df)
        self.output = output

        # Build inverter — needs the full state from all steps
        self._inverter = InverseTransformer(
            output            = output,
            original_df       = self.df,
            redundant_map     = self._filter.redundant_map,
            redundant_mappers = self._filter.redundant_mappers,
        )

        if self.verbose:
            self.summary()

        return output

    def inverse_transform(self, tensor: torch.Tensor) -> pd.DataFrame:
        """Reconstruct a synthetic DataFrame from a diffusion model tensor."""
        if self._inverter is None:
            raise RuntimeError("Must call fit_transform() before inverse_transform()")
        return self._inverter.inverse_transform(tensor)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> None:
        if self.output is None:
            print("No output available — run fit_transform() first.")
            return

        o = self.output
        print("\n" + "=" * 55)
        print("  PREPROCESSOR SUMMARY")
        print("=" * 55)
        print(f"\n  Input  : {o.n_rows} rows x {self.profile.n_cols} columns")
        print(f"  Output : {o.n_rows} rows x {o.n_features} features in tensor")

        print(f"\n  Excluded columns ({len(o.excluded_columns)}):")
        for col, reason in o.excluded_columns.items():
            print(f"    ✗ {col:<25} → {reason}")

        if o.redundant_pairs:
            print(f"\n  Redundant pairs removed ({len(o.redundant_pairs)}):")
            for a, b, score in o.redundant_pairs:
                print(f"    ≡ {a} ↔ {b}  score={score:.3f}")

        if o.imputation_values:
            print(f"\n  Imputation values applied ({len(o.imputation_values)}):")
            for col, val in o.imputation_values.items():
                print(f"    ~ {col:<25} → {val}")

        if o.zero_inflated_map:
            zi_originals = set(o.zero_inflated_map.values())
            print(f"\n  Zero-inflated columns split ({len(zi_originals)}):")
            for orig in zi_originals:
                print(f"    ÷ {orig} → {orig}_flag + {orig}_value")

        print(f"\n  Features in tensor:")
        print(f"  {'Feature':<30} {'Type':<15} {'Transform'}")
        print(f"  {'-'*29} {'-'*14} {'-'*12}")
        for ct in o.transforms:
            print(f"  {ct.tensor_name:<30} {ct.column_type:<15} {ct.transform_type}")

        print("\n" + "=" * 55)