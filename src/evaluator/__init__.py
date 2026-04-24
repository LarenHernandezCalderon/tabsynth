"""
evaluator/
─────────────────────────────────────────────────────────────────────────────
Phase 5 — The Evaluator.

Public API:
    from evaluator import Evaluator, EvaluatorOutput

Internal modules (never import directly):
    evaluator.py       — orchestrator
    fidelity.py        — KS Test + Wasserstein + TVD
    relationships.py   — Pearson + Spearman + Frobenius norm
    utility.py         — TSTR multi-model
    detectability.py   — Accuracy + ROC-AUC
    privacy.py         — Nearest-neighbour distance + distance ratio
    diversity.py       — Variance ratio + PCA spread + category coverage
    report.py          — report formatter
"""

from .evaluator import Evaluator, EvaluatorOutput

__all__ = ["Evaluator", "EvaluatorOutput"]