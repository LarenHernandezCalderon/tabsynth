"""
preprocessor/
─────────────────────────────────────────────────────────────────────────────
Phase 2 — Adaptive Preprocessor.

Public API:
    from preprocessor import Preprocessor, PreprocessorOutput

Internal structure:
    preprocessor.py          — orchestrator (8 steps in order)
    contracts.py             — dataclasses: ColumnTransform, PreprocessorOutput
    steps/
        step0_null_normalizer.py
        step1_column_filter.py
        step2_missing_handler.py
        step3_zero_inflated.py
        step4_numeric_transform.py
        step5_categorical_encoder.py
        step6_normalizer.py
        step7_tensor_builder.py
    inverse/
        inverter.py
"""

from .preprocessor import Preprocessor
from .contracts    import PreprocessorOutput, ColumnTransform

__all__ = ["Preprocessor", "PreprocessorOutput", "ColumnTransform"]