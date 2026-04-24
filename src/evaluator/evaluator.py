"""
evaluator.py
─────────────────────────────────────────────────────────────────────────────
Phase 5 — The Evaluator.

Measures the quality of the synthetic dataset by comparing it against the
original real dataset across six dimensions.

Orchestrates all evaluation modules in parallel and produces a single
EvaluatorOutput with individual block scores, detailed metrics, and a
weighted overall score.

Dimensions and weights:
    FIDELITY       (25%) — marginal distributions match (KS, Wasserstein, TVD)
    RELATIONSHIPS  (20%) — correlation structure preserved (Pearson, Spearman, Frobenius)
    UTILITY        (25%) — TSTR: ML models trained on synthetic work on real data
    DETECTABILITY  (15%) — a classifier cannot tell real from synthetic
    PRIVACY        (10%) — synthetic records are not too close to real ones
    DIVERSITY      ( 5%) — synthetic data covers the full statistical space

Usage:
    evaluator = Evaluator()
    result    = evaluator.evaluate(real_df, synthetic_df)
    result    = evaluator.evaluate(real_df, synthetic_df, target_col="Survived")

    # Print report
    evaluator.report(result)

    # Save report to file
    evaluator.save_report(result, "evaluation_report.txt")

Dependencies:
    pandas, numpy, scipy, sklearn

Part of:
    Synthetic Data Engine — Auto-Adaptive Diffusion System
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from evaluator.fidelity       import FidelityEvaluator,       FidelityResult
from evaluator.relationships  import RelationshipsEvaluator,  RelationshipsResult
from evaluator.utility        import UtilityEvaluator,        UtilityResult
from evaluator.detectability  import DetectabilityEvaluator,  DetectabilityResult
from evaluator.privacy        import PrivacyEvaluator,        PrivacyResult
from evaluator.diversity      import DiversityEvaluator,      DiversityResult
from evaluator.report         import ReportFormatter


# ── Weights ───────────────────────────────────────────────────────────────────

# Evaluator A — Statistical Realism (used in overall score)
WEIGHTS = {
    "fidelity"      : 0.30,
    "relationships" : 0.25,
    "utility"       : 0.30,
    "privacy"       : 0.10,
    "diversity"     : 0.05,
}

# Evaluator B — Adversarial Detectability (diagnostic only, not in overall score)
# Detectability is computed and reported but excluded from the weighted overall.
# Reason: high-cardinality columns (names, codes, IDs) make any classifier
# trivially accurate — this metric diagnoses artefacts, not statistical quality.


# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass
class EvaluatorOutput:
    """Full output of Phase 5."""
    # Block results
    fidelity       : FidelityResult
    relationships  : RelationshipsResult
    utility        : UtilityResult
    detectability  : DetectabilityResult
    privacy        : PrivacyResult
    diversity      : DiversityResult

    # Individual scores (0-1)
    fidelity_score      : float
    relationships_score : float
    utility_score       : float
    detectability_score : float
    privacy_score       : float
    diversity_score     : float

    # Overall
    overall_score  : float

    # Metadata
    n_rows_real    : int
    n_rows_synth   : int
    n_cols         : int
    target_column  : Optional[str]


# ── Main evaluator ────────────────────────────────────────────────────────────

class Evaluator:
    """
    Phase 5 orchestrator: evaluates synthetic data quality.

    Usage:
        ev     = Evaluator()
        result = ev.evaluate(real_df, synth_df)
        ev.report(result)
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def evaluate(
        self,
        real_df    : pd.DataFrame,
        synth_df   : pd.DataFrame,
        target_col : Optional[str] = None,
    ) -> EvaluatorOutput:

        if self.verbose:
            print("\n" + "=" * 55)
            print("  Synthetic Data Engine — Evaluator (Phase 5)")
            print("=" * 55)
            print(f"\n  Evaluating {len(synth_df)} synthetic rows against "
                  f"{len(real_df)} real rows...")

        # ── Run all evaluators ─────────────────────────────────────────────
        if self.verbose:
            print("\n  Running evaluations...")

        self._log("  [1/6] Fidelity       (KS + Wasserstein + TVD)...")
        fidelity = FidelityEvaluator().evaluate(real_df, synth_df)

        self._log("  [2/6] Relationships  (Pearson + Spearman + Frobenius)...")
        relationships = RelationshipsEvaluator().evaluate(real_df, synth_df)

        self._log("  [3/6] Utility        (TSTR — 3 models)...")
        utility = UtilityEvaluator().evaluate(real_df, synth_df, target_col)

        self._log("  [4/6] Detectability  (RandomForest real vs synthetic)...")
        detectability = DetectabilityEvaluator().evaluate(real_df, synth_df)

        self._log("  [5/6] Privacy        (Nearest-neighbour distances)...")
        privacy = PrivacyEvaluator().evaluate(real_df, synth_df)

        self._log("  [6/6] Diversity      (Variance + PCA + coverage)...")
        diversity = DiversityEvaluator().evaluate(real_df, synth_df)

        # ── Individual scores ─────────────────────────────────────────────
        scores = {
            "fidelity"      : fidelity.score,
            "relationships" : relationships.score,
            "utility"       : utility.score,
            "detectability" : detectability.score,  # Evaluator B — diagnostic only
            "privacy"       : privacy.score,
            "diversity"     : diversity.score,
        }

        # ── Weighted overall score (Evaluator A only) ─────────────────────────────────────────
        overall = float(sum(
            scores[k] * WEIGHTS[k] for k in WEIGHTS
        ))

        common_cols = [c for c in real_df.columns if c in synth_df.columns]

        return EvaluatorOutput(
            fidelity            = fidelity,
            relationships       = relationships,
            utility             = utility,
            detectability       = detectability,
            privacy             = privacy,
            diversity           = diversity,
            fidelity_score      = scores["fidelity"],
            relationships_score = scores["relationships"],
            utility_score       = scores["utility"],
            detectability_score = scores["detectability"],
            privacy_score       = scores["privacy"],
            diversity_score     = scores["diversity"],
            overall_score       = float(np.clip(overall, 0, 1)),
            n_rows_real         = len(real_df),
            n_rows_synth        = len(synth_df),
            n_cols              = len(common_cols),
            target_column       = utility.target_column,
        )

    def report(self, output: EvaluatorOutput) -> None:
        """Print the full evaluation report to stdout."""
        formatter = ReportFormatter()
        print(formatter.format(output))

    def save_report(self, output: EvaluatorOutput, path: str) -> None:
        """Save the full evaluation report to a text file."""
        formatter = ReportFormatter()
        formatter.save(output, path)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)