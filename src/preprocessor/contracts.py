"""
contracts.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 — Data contracts.

Central definition of all dataclasses used across the preprocessor package.
Imported by every step and by the public API.

Classes:
    ColumnTransform     — everything needed to invert one feature's transforms
    PreprocessorOutput  — full output of the preprocessor pipeline
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, QuantileTransformer


# ── Constants shared across steps ────────────────────────────────────────────

# Column types that enter the diffusion model tensor
DIFFUSION_TYPES = {
    "numerical_continuous",
    "ordinal",
    "binary",
    "categorical",
    "unknown_string",
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
REDUNDANCY_THRESHOLD = 0.999

# Sentinel values that represent masked nulls
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
    original_name         : str
    tensor_name           : str            # name in the tensor (may differ after split)
    column_type           : str            # continuous | ordinal | binary | categorical
    transform_type        : str            # none | log | sqrt | quantile
    was_zero_inflated     : bool  = False
    missing_ratio         : float = 0.0
    has_missing_flag      : bool  = False  # True if <col>_missing was created

    # Inverse transformers — populated during fit_transform
    quantile_transformer  : Optional[QuantileTransformer] = None
    label_encoder         : Optional[LabelEncoder]        = None
    minmax_scaler         : Optional[MinMaxScaler]        = None
    sqrt_min              : float = 0.0
    log_min               : float = 0.0
    original_dtype        : str   = "float64"
    original_categories   : Optional[list] = None


@dataclass
class PreprocessorOutput:
    """
    Full output of the preprocessor pipeline.
    Contains the training-ready tensor plus everything needed for inversion.
    """
    tensor_dataset    : TensorDataset
    feature_names     : list[str]        # ordered column names in the tensor
    column_types      : dict[str, str]   # feature → continuous|ordinal|binary|categorical
    n_features        : int
    n_rows            : int

    # For reconstruction
    transforms        : list[ColumnTransform]
    zero_inflated_map : dict[str, str]   # col_flag/col_value → original col

    # Exclusion metadata
    excluded_columns  : dict[str, str]   # col_name → reason for exclusion
    redundant_pairs   : list[tuple]      # (col_a, col_b, score)
    imputation_values : dict[str, float] # col → imputation value used