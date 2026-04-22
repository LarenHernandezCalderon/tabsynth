"""
preprocessor.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 — Adaptive Preprocessor.

Receives the DatasetProfile (Phase 1) + the original DataFrame and produces:
    - A PyTorch TensorDataset ready for TabDDPM
    - A PreprocessorOutput containing everything needed to invert
      the transformations and reconstruct the synthetic DataFrame

Internal pipeline (in order):
    Step 0 — Masked null normalisation
             Replicates the Phase 1 fix on the real DataFrame.
             Replaces sentinel values ("?", "N/A", "none", etc.) with NaN
             across all string columns, using the same MASKED_NULL_VALUES
             list as the MetadataExtractor.

    Step 1 — Column filtering
             Excludes: skipped, id_like, direct PII, code columns
             (all generated separately in Phase 4).
             Detects and handles perfectly redundant column pairs (corr = 1.0).

    Step 2 — Missing indicators (universal Strategy B)
             For every column with missing_ratio > 0, creates a binary
             column <col>_missing (1 = was null, 0 = had a value).
             Then imputes the missing value so the model can train:
               - Numeric / ordinal  → median
               - Categorical / binary → mode

    Step 3 — Zero-inflated split
             Columns with zero_inflated distribution (e.g. capital-gain,
             SibSp) are split into:
               <col>_flag   (binary)  — was the value non-zero?
               <col>_value  (numeric) — the value when flag = 1, else 0

    Step 4 — Numeric transforms
             Applies the transform recommended by the DistributionProfiler:
               log      → log1p(x)            | inverse: expm1(x)
               sqrt     → sqrt(x)             | inverse: x²
               quantile → QuantileTransformer  | inverse: inverse_transform
               none     → no change

    Step 5 — Categorical encoding
             Categorical / binary string columns → LabelEncoder (0, 1, 2, …)
             Numeric binary columns (0/1)        → no change
             Full mapping saved for decoding after generation.

    Step 6 — MinMax normalisation
             Scales every numeric column to [0, 1].
             Scaler parameters saved for inversion.

    Step 7 — TensorDataset construction
             DataFrame → numpy → torch.FloatTensor → TensorDataset
             Builds ColumnTransform list for full inversion.

Output (PreprocessorOutput):
    tensor_dataset      : PyTorch TensorDataset
    feature_names       : ordered list of column names in the tensor
    column_types        : feature → continuous | ordinal | binary | categorical
    n_features          : total number of features in the tensor
    transforms          : list of ColumnTransform objects (one per feature)
    excluded_columns    : excluded columns and the reason for each
    redundant_pairs     : pairs removed due to perfect redundancy

Inversion (synthetic DataFrame reconstruction):
    inverse_transform(tensor) → DataFrame with original columns

Guarantee: the output DataFrame always has the same columns as the
original dataset, in the same order, with the same categories.

Dependencies:
    pandas, numpy, torch, sklearn

Part of:
    Synthetic Data Engine — Auto-Adaptive Diffusion System
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from dataclasses import dataclass, field
from typing import Optional
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, QuantileTransformer

from analyzer.dataset_analyzer import DatasetProfile, ColumnProfile


# ── Constants ─────────────────────────────────────────────────────────────────

# Column types that enter the diffusion model tensor
DIFFUSION_TYPES = {
    "numerical_continuous",
    "ordinal",
    "binary",
    "categorical",
    "unknown_string",   # code columns that survive the filtering step
}

# Generation strategies handled outside the tensor (Phase 4)
EXCLUDED_STRATEGIES = {
    "faker.name()",
    "faker.first_name()",
    "faker.last_name()",
    "faker.email()",
    "faker.phone_number()",
    "faker.address()",
    "faker.country()",
    "faker.city()",
    "faker.zipcode()",
    "regex_pattern",
    "sequential_id",
    "datetime_range",
}

# Threshold for perfect redundancy detection
# Only columns with correlation >= this value are considered identical
REDUNDANCY_THRESHOLD = 0.999

# Same sentinel list as MetadataExtractor — values that represent masked nulls
MASKED_NULL_VALUES = {
    "?", "N/A", "n/a", "NA", "na", "None", "none",
    "NaN", "nan", "null", "NULL", "", " ", "-", "--",
    "unknown", "Unknown",
}


# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class ColumnTransform:
    """
    Stores everything needed to invert the preprocessing of a single column.
    One instance is created per feature in the tensor during fit_transform.
    """
    original_name    : str
    tensor_name      : str           # name in the tensor (may differ after split)
    column_type      : str           # continuous | ordinal | binary | categorical
    transform_type   : str           # none | log | sqrt | quantile
    was_zero_inflated: bool  = False
    missing_ratio    : float = 0.0
    has_missing_flag : bool  = False  # True if <col>_missing was created

    # Inverse transformers — populated during fit_transform
    quantile_transformer : Optional[QuantileTransformer] = None
    label_encoder        : Optional[LabelEncoder]        = None
    minmax_scaler        : Optional[MinMaxScaler]        = None
    sqrt_min             : float = 0.0   # clamp floor before sqrt inverse
    log_min              : float = 0.0   # clamp floor before log inverse
    original_dtype       : str   = "float64"
    original_categories  : Optional[list] = None  # LabelEncoder classes


@dataclass
class PreprocessorOutput:
    """
    Full output of the preprocessor.
    Contains the training-ready tensor plus everything needed for inversion.
    """
    tensor_dataset    : TensorDataset
    feature_names     : list[str]        # ordered column names in the tensor
    column_types      : dict[str, str]   # feature → continuous|ordinal|binary|categorical
    n_features        : int
    n_rows            : int

    # For reconstruction
    transforms        : list[ColumnTransform]  # one per feature in the tensor
    zero_inflated_map : dict[str, str]         # col_flag/col_value → original col

    # Exclusion metadata
    excluded_columns  : dict[str, str]   # col_name → reason for exclusion
    redundant_pairs   : list[tuple]      # (col_a, col_b, score) removed pairs
    imputation_values : dict[str, float] # col → imputation value used


# ── Main preprocessor ─────────────────────────────────────────────────────────

class Preprocessor:
    """
    Adaptive preprocessor that reads the DatasetProfile and builds
    the TensorDataset ready for TabDDPM.

    Usage:
        prep   = Preprocessor(df, profile)
        output = prep.fit_transform()

        # After generation with the diffusion model:
        df_synthetic = prep.inverse_transform(synthetic_tensor)
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

        # Internal state
        self._working_df         : pd.DataFrame         = pd.DataFrame()
        self._excluded           : dict[str, str]        = {}
        self._redundant_pairs    : list[tuple]           = []
        self._redundant_map      : dict[str, str]        = {}  # removed_col → source_col
        self._redundant_mappers  : dict[str, dict]       = {}  # removed_col → {src_val: tgt_val}
        self._transforms         : list[ColumnTransform] = []
        self._zero_inflated_map  : dict[str, str]        = {}
        self._imputation_values  : dict[str, float]      = {}
        self._feature_names      : list[str]             = []
        self._column_types       : dict[str, str]        = {}

    # ── Entry point ───────────────────────────────────────────────────────────

    def fit_transform(self) -> PreprocessorOutput:
        """Run the full pipeline and return the PreprocessorOutput."""

        if self.verbose:
            print("\n" + "=" * 55)
            print("  Synthetic Data Engine — Preprocessor (Phase 2)")
            print("=" * 55)

        self._step0_normalize_masked_nulls()
        self._step1_filter_columns()
        self._step2_missing_indicators()
        self._step3_zero_inflated_split()
        self._step4_numeric_transforms()
        self._step5_encode_categoricals()
        self._step6_normalize()
        output = self._step7_build_tensor()

        self.output = output

        if self.verbose:
            self.summary()

        return output

    # ── Step 0: Masked null normalisation ────────────────────────────────────

    def _step0_normalize_masked_nulls(self) -> None:
        """
        Replicates the MetadataExtractor (Phase 1) sentinel fix on the real DataFrame.

        MetadataExtractor detects masked nulls and updates the missing_ratio in
        the DatasetProfile, but does NOT modify the original DataFrame.
        This step applies the same replacement on self.df before any other
        processing, ensuring full consistency between what the analyser
        described and what the preprocessor actually sees.
        """
        if self.verbose:
            print("\nStep 0 — Masked null normalisation")
            print("-" * 40)

        total_replaced = 0
        replaced_cols  = []

        for col in self.df.columns:
            series = self.df[col]
            if series.dtype != object and not pd.api.types.is_string_dtype(series):
                continue

            mask  = series.isin(MASKED_NULL_VALUES)
            count = int(mask.sum())
            if count > 0:
                self.df[col] = series.where(~mask, other=np.nan)
                total_replaced += count
                replaced_cols.append((col, count))

        if self.verbose:
            if total_replaced == 0:
                print("  No sentinel values detected — step skipped")
            else:
                for col, count in replaced_cols:
                    ratio = count / len(self.df)
                    print(f"  {col:<25} → {count} sentinels replaced "
                          f"({ratio:.1%} of total)")
                print(f"  Total replaced: {total_replaced}")

    # ── Step 1: Column filtering ──────────────────────────────────────────────

    def _step1_filter_columns(self) -> None:
        """
        Determines which columns go into the tensor and which are excluded.

        Excluded:
          - Columns flagged as should_skip in the DatasetProfile
          - id_like columns (regenerated as sequential integers in Phase 4)
          - Direct PII columns (generated with Faker in Phase 4)
          - code/regex columns (generated with RegexMiner in Phase 4)
          - datetime columns (generated by date range in Phase 4)
          - Perfectly redundant columns (correlation >= REDUNDANCY_THRESHOLD)
        """
        if self.verbose:
            print("\nStep 1 — Column filtering")
            print("-" * 40)

        keep_cols = []

        for col_profile in self.profile.columns:
            col = col_profile.column_name

            # Columns flagged for skipping by Phase 1
            if col_profile.should_skip:
                self._excluded[col] = f"skip: {col_profile.skip_reason or 'flagged for skipping'}"
                continue

            # Sequential ID columns
            if col_profile.is_id:
                self._excluded[col] = "id_like: regenerated as sequential integer (1, 2, 3…)"
                continue

            # Strategies handled outside the tensor
            strategy = col_profile.generation_strategy or ""
            if strategy in EXCLUDED_STRATEGIES:
                self._excluded[col] = f"external generation: strategy='{strategy}'"
                continue

            # Datetime columns
            if col_profile.detected_type == "datetime":
                self._excluded[col] = "datetime: generated from date range in Phase 4"
                continue

            keep_cols.append(col)

        # Detect perfectly redundant pairs among the remaining columns
        keep_cols = self._remove_redundant_columns(keep_cols)

        # Build working DataFrame with selected columns only
        available = [c for c in keep_cols if c in self.df.columns]
        self._working_df = self.df[available].copy()

        if self.verbose:
            print(f"  Columns to tensor    : {len(available)}  {available}")
            print(f"  Excluded columns     : {len(self._excluded)}")
            for col, reason in self._excluded.items():
                print(f"    ✗ {col:<25} → {reason}")
            if self._redundant_pairs:
                print(f"  Redundant pairs      : {len(self._redundant_pairs)}")
                for a, b, score in self._redundant_pairs:
                    print(f"    ≡ {a} ↔ {b}  (score={score:.3f}) — removed {b}")

    def _remove_redundant_columns(self, cols: list[str]) -> list[str]:
        """
        Removes one column from each pair with correlation >= REDUNDANCY_THRESHOLD.
        Only triggered for perfect redundancy (1.000) — same information encoded
        twice, like educational-num and education in Adult Income.

        For each removed column, saves the exact value mapping so it can be
        reconstructed in inverse_transform from its retained counterpart.
        The final synthetic dataset always has the same columns as the original.
        """
        to_remove = set()

        # Check Pearson correlation matrix from DatasetProfile (numeric pairs)
        if self.profile.correlation_matrix is not None:
            matrix = self.profile.correlation_matrix
            numeric_cols = [c for c in cols if c in matrix.columns]

            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    a, b = numeric_cols[i], numeric_cols[j]
                    if a in to_remove or b in to_remove:
                        continue
                    score = abs(matrix.loc[a, b])
                    if score >= REDUNDANCY_THRESHOLD:
                        self._redundant_pairs.append((a, b, score))
                        to_remove.add(b)

        # Check all_pairs from DatasetProfile (covers mixed and categorical pairs)
        for pair in self.profile.all_pairs:
            a, b = pair.col_a, pair.col_b
            if a not in cols or b not in cols:
                continue
            if a in to_remove or b in to_remove:
                continue
            if pair.score >= REDUNDANCY_THRESHOLD:
                if (a, b, pair.score) not in self._redundant_pairs:
                    self._redundant_pairs.append((a, b, pair.score))
                to_remove.add(b)

        # For each removed column, build the exact value mapping for reconstruction
        for a, b, score in self._redundant_pairs:
            if a in self.df.columns and b in self.df.columns:
                # Build exact mapping from the original dataset
                mapping_df = self.df[[a, b]].dropna().drop_duplicates()
                mapper = {}
                for _, row in mapping_df.iterrows():
                    mapper[row[a]] = row[b]
                self._redundant_map[b]     = a        # b is reconstructed from a
                self._redundant_mappers[b] = mapper   # using this exact mapping

            self._excluded[b] = (
                f"redundant: correlation={score:.3f} — "
                f"will be reconstructed from '{a}' in inverse_transform"
            )

        return [c for c in cols if c not in to_remove]

    # ── Step 2: Missing indicators ────────────────────────────────────────────

    def _step2_missing_indicators(self) -> None:
        """
        Universal Strategy B for all missing values.

        For every column with at least one null:
          1. Create binary indicator <col>_missing (1 = was null, 0 = had value)
          2. Impute the missing value so the model can train:
               - Numeric / ordinal  → median (robust to outliers)
               - Categorical / binary → mode (most frequent value)

        The model learns the correlation between <col>_missing and all other
        features, capturing the real missingness pattern (MNAR, MAR, or MCAR)
        rather than just the ratio.
        """
        if self.verbose:
            print("\nStep 2 — Missing indicators (Strategy B)")
            print("-" * 40)

        cols_with_missing = [
            col for col in self._working_df.columns
            if self._working_df[col].isna().any()
        ]

        if not cols_with_missing:
            if self.verbose:
                print("  No missing values detected — step skipped")
            return

        for col in cols_with_missing:
            missing_count = self._working_df[col].isna().sum()
            missing_ratio = missing_count / len(self._working_df)

            # Create binary missing indicator column
            flag_col = f"{col}_missing"
            self._working_df[flag_col] = self._working_df[col].isna().astype(int)

            # Compute imputation value based on column type
            dtype = self._working_df[col].dtype

            if pd.api.types.is_numeric_dtype(dtype):
                impute_val = float(self._working_df[col].median())
                strategy   = "median"
            else:
                mode_vals  = self._working_df[col].mode()
                impute_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "UNKNOWN"
                strategy   = "mode"

            self._working_df[col] = self._working_df[col].fillna(impute_val)
            self._imputation_values[col] = impute_val

            if self.verbose:
                print(f"  {col:<25} → {flag_col} created  "
                      f"({missing_count} nulls, {missing_ratio:.1%})  "
                      f"imputed with {strategy}={impute_val}")

    # ── Step 3: Zero-inflated split ───────────────────────────────────────────

    def _step3_zero_inflated_split(self) -> None:
        """
        Splits zero-inflated columns into two simpler features the model
        can learn independently:
          <col>_flag   : 1 if the original value was > 0, else 0
          <col>_value  : the numeric value when flag=1, 0 when flag=0
                         (model learns the conditional distribution)

        This separates two distinct learning problems:
          1. How often does a non-zero value occur? (via flag)
          2. What is the typical non-zero value when it occurs? (via value)

        Inversion: col_original = flag * value
        """
        if self.verbose:
            print("\nStep 3 — Zero-inflated split")
            print("-" * 40)

        zero_inflated_cols = [
            col for col in self._working_df.columns
            if self.profile.get_column(col) is not None
            and self.profile.get_column(col).distribution_type == "zero_inflated"
        ]

        if not zero_inflated_cols:
            if self.verbose:
                print("  No zero-inflated columns — step skipped")
            return

        for col in zero_inflated_cols:
            series     = self._working_df[col]
            zero_ratio = (series == 0).sum() / len(series)

            flag_col  = f"{col}_flag"
            value_col = f"{col}_value"

            # Flag: 1 if the original value was non-zero
            self._working_df[flag_col]  = (series != 0).astype(int)

            # Value: original value (0 when zero, non-zero otherwise)
            self._working_df[value_col] = series.copy()

            # Register the mapping for inversion
            self._zero_inflated_map[flag_col]  = col
            self._zero_inflated_map[value_col] = col

            # Drop the original column from the working DataFrame
            self._working_df.drop(columns=[col], inplace=True)

            if self.verbose:
                non_zero = (series != 0).sum()
                print(f"  {col:<20} → {flag_col} + {value_col}")
                print(f"    zeros: {zero_ratio:.1%}  |  non-zero rows: {non_zero}")

    # ── Step 4: Numeric transforms ────────────────────────────────────────────

    def _step4_numeric_transforms(self) -> None:
        """
        Applies the transforms recommended by the DistributionProfiler (Phase 1).

        Goal: make numeric columns as close to Gaussian as possible before
        feeding them to the diffusion model, which operates in Gaussian space.

        log      → np.log1p(x)          requires x >= 0 (uses +1 to avoid log(0))
        sqrt     → np.sqrt(x)           requires x >= 0
        quantile → QuantileTransformer  no assumptions; non-linear
        none     → no change

        _missing and _flag columns are binary — never transformed.
        """
        if self.verbose:
            print("\nStep 4 — Numeric transforms")
            print("-" * 40)

        for col in list(self._working_df.columns):

            # Resolve the base column name for profile lookup
            original_col = self._zero_inflated_map.get(col, col)
            base_col = original_col
            for suffix in ("_missing", "_flag", "_value"):
                if base_col.endswith(suffix):
                    base_col = base_col[: -len(suffix)]
                    break

            col_profile = self.profile.get_column(base_col)
            if col_profile is None:
                continue

            # Only transform numeric columns
            if not pd.api.types.is_numeric_dtype(self._working_df[col]):
                continue

            # Binary indicator columns (_missing, _flag) are never transformed
            if col.endswith("_missing") or col.endswith("_flag"):
                continue

            transform = col_profile.recommended_transform or "none"
            series    = self._working_df[col].copy()

            if transform == "log":
                # Clamp to 0 to avoid log of negatives
                self._working_df[col] = np.log1p(series.clip(lower=0))
                if self.verbose:
                    print(f"  {col:<30} → log1p")

            elif transform == "sqrt":
                self._working_df[col] = np.sqrt(series.clip(lower=0))
                if self.verbose:
                    print(f"  {col:<30} → sqrt")

            elif transform == "quantile":
                qt = QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=min(1000, len(series)),
                    random_state=42,
                )
                self._working_df[col] = qt.fit_transform(
                    series.values.reshape(-1, 1)
                ).flatten()

                # Save the transformer for inversion
                if not hasattr(self, "_quantile_transformers"):
                    self._quantile_transformers = {}
                self._quantile_transformers[col] = qt

                if self.verbose:
                    print(f"  {col:<30} → quantile (normal output)")

    # ── Step 5: Categorical encoding ──────────────────────────────────────────

    def _step5_encode_categoricals(self) -> None:
        """
        Converts string columns to integers so they can enter the tensor.

        Categorical and binary string columns → LabelEncoder (0, 1, 2, …)
        Numeric binary columns (0/1)           → no change
        Ordinal columns (already numeric)      → no change

        The full LabelEncoder is saved per column so categories can be
        decoded back after generation. Note: typos in category names
        (e.g. 'Very Geod' in the Diamonds dataset) are intentionally
        preserved — the system replicates, it does not correct.
        """
        if self.verbose:
            print("\nStep 5 — Categorical encoding")
            print("-" * 40)

        if not hasattr(self, "_label_encoders"):
            self._label_encoders: dict[str, LabelEncoder] = {}

        for col in self._working_df.columns:
            series = self._working_df[col]

            # Only process string/object columns
            if series.dtype != object and not pd.api.types.is_string_dtype(series):
                continue

            le = LabelEncoder()
            # Fill any residual nulls before encoding
            series_clean = series.fillna("UNKNOWN").astype(str)
            self._working_df[col] = le.fit_transform(series_clean)
            self._label_encoders[col] = le

            if self.verbose:
                print(f"  {col:<25} → {len(le.classes_)} classes  "
                      f"{list(le.classes_[:4])}{'...' if len(le.classes_) > 4 else ''}")

    # ── Step 6: MinMax normalisation ──────────────────────────────────────────

    def _step6_normalize(self) -> None:
        """
        Scales all numeric columns to [0, 1].

        The diffusion model operates in [0, 1]^d space. Without normalisation,
        a column ranging [0, 500 000] would dominate the gradient over a binary
        column in [0, 1].

        MinMaxScaler parameters (min, max) are saved per column for inversion:
            x_original = x_scaled * (max - min) + min
        """
        if self.verbose:
            print("\nStep 6 — MinMax normalisation [0, 1]")
            print("-" * 40)

        if not hasattr(self, "_minmax_scalers"):
            self._minmax_scalers: dict[str, MinMaxScaler] = {}

        for col in self._working_df.columns:
            series = self._working_df[col]

            if not pd.api.types.is_numeric_dtype(series):
                continue

            scaler = MinMaxScaler(feature_range=(0, 1))
            self._working_df[col] = scaler.fit_transform(
                series.values.reshape(-1, 1)
            ).flatten()
            self._minmax_scalers[col] = scaler

        if self.verbose:
            print(f"  {len(self._minmax_scalers)} columns scaled to [0, 1]")

    # ── Step 7: TensorDataset construction ────────────────────────────────────

    def _step7_build_tensor(self) -> PreprocessorOutput:
        """
        Converts the processed DataFrame to a PyTorch TensorDataset.

        Also builds:
          - feature_names : ordered list of column names in the tensor
          - column_types  : semantic type per feature (for TabDDPM loss selection)
          - transforms    : list of ColumnTransform objects for full inversion
        """
        if self.verbose:
            print("\nStep 7 — TensorDataset construction")
            print("-" * 40)

        self._feature_names = list(self._working_df.columns)

        # Classify each feature type for TabDDPM
        for col in self._feature_names:
            self._column_types[col] = self._classify_feature_type(col)

        # Build ColumnTransform list
        self._build_transforms()

        # Convert to tensor
        matrix  = self._working_df.values.astype(np.float32)
        tensor  = torch.FloatTensor(matrix)
        dataset = TensorDataset(tensor)

        if self.verbose:
            print(f"  Tensor shape    : {tuple(tensor.shape)}")
            print(f"  Total features  : {len(self._feature_names)}")
            type_counts: dict[str, int] = {}
            for t in self._column_types.values():
                type_counts[t] = type_counts.get(t, 0) + 1
            for t, count in sorted(type_counts.items()):
                print(f"    {t:<15} : {count} features")

        return PreprocessorOutput(
            tensor_dataset    = dataset,
            feature_names     = self._feature_names,
            column_types      = self._column_types,
            n_features        = len(self._feature_names),
            n_rows            = len(self._working_df),
            transforms        = self._transforms,
            zero_inflated_map = self._zero_inflated_map,
            excluded_columns  = self._excluded,
            redundant_pairs   = self._redundant_pairs,
            imputation_values = self._imputation_values,
        )

    # ── Feature type classifier ───────────────────────────────────────────────

    def _classify_feature_type(self, col: str) -> str:
        """
        Determines the semantic type of a feature for TabDDPM.
        TabDDPM uses different loss functions for each type:
          binary     → binary cross-entropy
          ordinal    → MSE with integer rounding at decode time
          categorical → cross-entropy
          continuous → MSE
        """
        # Split-generated columns
        if col.endswith("_missing") or col.endswith("_flag"):
            return "binary"

        if col.endswith("_value"):
            base     = col[:-6]  # strip '_value'
            base_col = self._zero_inflated_map.get(col, base)
            return "continuous"

        # Original columns — look up in DatasetProfile
        prof = self.profile.get_column(col)
        if prof is None:
            return "continuous"

        if prof.detected_type == "binary":
            return "binary"
        if prof.detected_type == "ordinal":
            return "ordinal"
        if prof.detected_type == "categorical":
            return "categorical"
        return "continuous"

    # ── Build transforms list ─────────────────────────────────────────────────

    def _build_transforms(self) -> None:
        """
        Builds the list of ColumnTransform objects.
        Each object stores the MinMaxScaler, LabelEncoder, and
        QuantileTransformer (if applicable) for a single feature,
        so the full preprocessing chain can be reversed after generation.
        """
        qt_map = getattr(self, "_quantile_transformers", {})
        le_map = getattr(self, "_label_encoders", {})
        mm_map = getattr(self, "_minmax_scalers", {})

        for col in self._feature_names:
            # Resolve base column name for profile lookup
            base_col = self._zero_inflated_map.get(col, col)
            for suffix in ("_missing", "_flag", "_value"):
                if base_col.endswith(suffix):
                    base_col = base_col[: -len(suffix)]
                    break

            prof = self.profile.get_column(base_col)
            transform_type = "none"
            if prof and not col.endswith(("_missing", "_flag", "_value")):
                transform_type = prof.recommended_transform or "none"

            ct = ColumnTransform(
                original_name        = base_col,
                tensor_name          = col,
                column_type          = self._column_types.get(col, "continuous"),
                transform_type       = transform_type,
                was_zero_inflated    = col in self._zero_inflated_map,
                missing_ratio        = prof.missing_ratio if prof else 0.0,
                has_missing_flag     = f"{base_col}_missing" in self._feature_names,
                quantile_transformer = qt_map.get(col),
                label_encoder        = le_map.get(col),
                minmax_scaler        = mm_map.get(col),
                original_categories  = list(le_map[col].classes_) if col in le_map else None,
            )
            self._transforms.append(ct)

    # ── Inversion ─────────────────────────────────────────────────────────────

    def inverse_transform(self, tensor: torch.Tensor) -> pd.DataFrame:
        """
        Reconstructs a synthetic DataFrame from a tensor generated by the
        diffusion model.

        Applies the inverse of fit_transform in reverse order:
          1. Tensor → numpy → DataFrame with feature_names
          2. Inverse MinMax scaling
          3. Decode categorical columns (LabelEncoder inverse)
          4. Invert numeric transforms (log → expm1, sqrt → x², quantile → inv)
          5. Reconstruct zero-inflated columns (flag * value)
          6. Restore NaNs using _missing indicator columns
          7. Reconstruct redundant columns removed in Step 1 (exact value mapping)
          8. Reorder columns to match the original dataset order

        Guarantee: the resulting DataFrame always has the same columns as the
        original dataset, in the same order.

        Returns:
            DataFrame with the same columns as the original dataset
        """
        if self.output is None:
            raise RuntimeError(
                "Must call fit_transform() before inverse_transform()"
            )

        # Step 1: Tensor → DataFrame
        arr = tensor.detach().numpy()
        df  = pd.DataFrame(arr, columns=self._feature_names)

        # Step 2: Inverse MinMax scaling
        mm_map = getattr(self, "_minmax_scalers", {})
        for col in df.columns:
            if col in mm_map:
                df[col] = mm_map[col].inverse_transform(
                    df[col].values.reshape(-1, 1)
                ).flatten()

        # Step 3: Decode categorical columns
        le_map = getattr(self, "_label_encoders", {})
        for col in df.columns:
            if col in le_map:
                le   = le_map[col]
                ints = df[col].round().clip(0, len(le.classes_) - 1).astype(int)
                df[col] = le.inverse_transform(ints)

        # Step 4: Invert numeric transforms
        for ct in self._transforms:
            col = ct.tensor_name
            if col not in df.columns:
                continue
            if ct.transform_type == "log":
                df[col] = np.expm1(df[col].clip(lower=-1))  # clip to avoid -inf
            elif ct.transform_type == "sqrt":
                df[col] = np.power(df[col].clip(lower=0), 2)
            elif ct.transform_type == "quantile" and ct.quantile_transformer:
                df[col] = ct.quantile_transformer.inverse_transform(
                    df[col].values.reshape(-1, 1)
                ).flatten()

        # Step 5: Reconstruct zero-inflated columns
        # flag > 0.5 → use the value; flag <= 0.5 → set to 0
        for flag_col, orig_col in self._zero_inflated_map.items():
            if not flag_col.endswith("_flag"):
                continue
            value_col = flag_col.replace("_flag", "_value")
            if flag_col in df.columns and value_col in df.columns:
                flag = (df[flag_col] > 0.5).astype(int)
                df[orig_col] = flag * df[value_col]
                df.drop(columns=[flag_col, value_col], inplace=True)

        # Step 6: Restore NaNs from _missing indicator columns
        for col in list(df.columns):
            if col.endswith("_missing"):
                base_col = col[:-8]  # strip '_missing'
                if base_col in df.columns:
                    mask = df[col] > 0.5
                    df.loc[mask, base_col] = np.nan
                df.drop(columns=[col], inplace=True)

        # Step 7: Reconstruct columns removed due to perfect redundancy
        # Each removed column is regenerated from its source column
        # using the exact value mapping captured during fit_transform.
        for removed_col, source_col in self._redundant_map.items():
            if source_col in df.columns:
                mapper = self._redundant_mappers.get(removed_col, {})
                if mapper:
                    df[removed_col] = df[source_col].map(mapper)
                    # For any values not covered by the mapping (model outliers),
                    # fall back to the nearest known key
                    if df[removed_col].isna().any():
                        keys = sorted(mapper.keys())
                        for idx in df[df[removed_col].isna()].index:
                            val     = df.loc[idx, source_col]
                            closest = min(keys, key=lambda k: abs(k - val))
                            df.loc[idx, removed_col] = mapper[closest]

        # Step 8: Reorder columns to match the original dataset
        original_cols = [c for c in self.df.columns if c in df.columns]
        df = df[original_cols]

        return df

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