"""
step7_tensor_builder.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 — Step 7: TensorDataset construction.

Converts the fully processed DataFrame into a PyTorch TensorDataset.
Also builds the ColumnTransform list that stores everything needed
to invert every transformation after generation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset
import pandas as pd

from analyzer.dataset_analyzer import DatasetProfile
from preprocessor.contracts import (
    ColumnTransform, PreprocessorOutput,
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, QuantileTransformer


class TensorBuilder:
    """
    Builds the TensorDataset and ColumnTransform list from the processed DataFrame.

    Usage:
        builder = TensorBuilder(profile, state, verbose=True)
        output  = builder.build(df)
    """

    def __init__(
        self,
        profile           : DatasetProfile,
        zero_inflated_map : dict[str, str],
        quantile_transformers: dict[str, QuantileTransformer],
        label_encoders    : dict[str, LabelEncoder],
        minmax_scalers    : dict[str, MinMaxScaler],
        excluded          : dict[str, str],
        redundant_pairs   : list[tuple],
        imputation_values : dict[str, float],
        verbose           : bool = True,
    ):
        self.profile               = profile
        self.zero_inflated_map     = zero_inflated_map
        self.quantile_transformers = quantile_transformers
        self.label_encoders        = label_encoders
        self.minmax_scalers        = minmax_scalers
        self.excluded              = excluded
        self.redundant_pairs       = redundant_pairs
        self.imputation_values     = imputation_values
        self.verbose               = verbose

    def build(self, df: pd.DataFrame) -> PreprocessorOutput:
        if self.verbose:
            print("\nStep 7 — TensorDataset construction")
            print("-" * 40)

        feature_names = list(df.columns)
        column_types  = {col: self._classify(col) for col in feature_names}
        transforms    = self._build_transforms(feature_names, column_types)

        # DataFrame → tensor
        matrix  = df.values.astype(np.float32)
        tensor  = torch.FloatTensor(matrix)
        dataset = TensorDataset(tensor)

        if self.verbose:
            print(f"  Tensor shape    : {tuple(tensor.shape)}")
            print(f"  Total features  : {len(feature_names)}")
            type_counts: dict[str, int] = {}
            for t in column_types.values():
                type_counts[t] = type_counts.get(t, 0) + 1
            for t, count in sorted(type_counts.items()):
                print(f"    {t:<15} : {count} features")

        return PreprocessorOutput(
            tensor_dataset    = dataset,
            feature_names     = feature_names,
            column_types      = column_types,
            n_features        = len(feature_names),
            n_rows            = len(df),
            transforms        = transforms,
            zero_inflated_map = self.zero_inflated_map,
            excluded_columns  = self.excluded,
            redundant_pairs   = self.redundant_pairs,
            imputation_values = self.imputation_values,
        )

    # ── Feature type classifier ───────────────────────────────────────────────

    def _classify(self, col: str) -> str:
        if col.endswith("_missing") or col.endswith("_flag"):
            return "binary"
        if col.endswith("_value"):
            return "continuous"

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

    # ── ColumnTransform builder ───────────────────────────────────────────────

    def _build_transforms(
        self,
        feature_names : list[str],
        column_types  : dict[str, str],
    ) -> list[ColumnTransform]:
        transforms = []

        for col in feature_names:
            # Resolve base column name for profile lookup
            base_col = self.zero_inflated_map.get(col, col)
            for suffix in ("_missing", "_flag", "_value"):
                if base_col.endswith(suffix):
                    base_col = base_col[: -len(suffix)]
                    break

            prof = self.profile.get_column(base_col)
            transform_type = "none"
            if prof and not col.endswith(("_missing", "_flag", "_value")):
                transform_type = prof.recommended_transform or "none"

            le  = self.label_encoders.get(col)
            ct  = ColumnTransform(
                original_name        = base_col,
                tensor_name          = col,
                column_type          = column_types.get(col, "continuous"),
                transform_type       = transform_type,
                was_zero_inflated    = col in self.zero_inflated_map,
                missing_ratio        = prof.missing_ratio if prof else 0.0,
                has_missing_flag     = f"{base_col}_missing" in feature_names,
                quantile_transformer = self.quantile_transformers.get(col),
                label_encoder        = le,
                minmax_scaler        = self.minmax_scalers.get(col),
                original_categories  = list(le.classes_) if le else None,
            )
            transforms.append(ct)

        return transforms