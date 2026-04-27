"""
report.py
─────────────────────────────────────────────────────────────────────────────
Phase 5 — Report Formatter.

Formats the EvaluatorOutput into a human-readable report printed to stdout
and optionally saved to a .txt file.

Output sections:
    1. Header + overall score + global semaphore
    2. Block scores summary (bar chart + individual semaphore)
    3. FIDELITY detail    — KS + Wasserstein + TVD per column
    4. RELATIONSHIPS detail — correlation deltas + Frobenius
    5. UTILITY detail    — TSTR per model
    6. DETECTABILITY detail — accuracy + AUC
    7. PRIVACY detail    — NND + distance ratio
    8. DIVERSITY detail  — variance ratio + PCA + coverage

Part of:
    Synthetic Data Engine — Auto-Adaptive Diffusion System
"""

from __future__ import annotations
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluator.evaluator import EvaluatorOutput


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar(score: float, width: int = 10) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def _semaphore(score: float) -> str:
    if score >= 0.85:
        return "🟢 EXCELLENT"
    elif score >= 0.70:
        return "🟡 GOOD"
    elif score >= 0.55:
        return "🟠 ACCEPTABLE"
    else:
        return "🔴 POOR"


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _fmt(v: float, decimals: int = 3) -> str:
    return f"{v:.{decimals}f}"


def _pass(ok: bool) -> str:
    return "✅" if ok else "⚠️ "


# ── Main formatter ────────────────────────────────────────────────────────────

class ReportFormatter:
    """
    Formats an EvaluatorOutput into a complete evaluation report.

    Usage:
        formatter = ReportFormatter()
        text      = formatter.format(output)
        print(text)
        formatter.save(output, "evaluation_report.txt")
    """

    def format(self, output: "EvaluatorOutput") -> str:
        buf = io.StringIO()
        w   = lambda s="": buf.write(s + "\n")

        self._header(w, output)
        self._scores_summary(w, output)
        self._fidelity(w, output)
        self._relationships(w, output)
        self._utility(w, output)
        self._detectability(w, output)
        self._privacy(w, output)
        self._diversity(w, output)
        self._footer(w, output)

        return buf.getvalue()

    def save(self, output: "EvaluatorOutput", path: str) -> None:
        text = self.format(output)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n  Report saved → {path}")

    # ── Sections ──────────────────────────────────────────────────────────────

    def _header(self, w, o):
        w("=" * 55)
        w("  Synthetic Data Engine — Evaluator (Phase 5)")
        w("=" * 55)
        w(f"\n  Real dataset      : {o.n_rows_real} rows × {o.n_cols} columns")
        w(f"  Synthetic dataset : {o.n_rows_synth} rows × {o.n_cols} columns")
        w(f"  Target column     : {o.target_column or 'auto-detected'}")

    def _scores_summary(self, w, o):
        w()
        w("─" * 55)
        w("  EVALUATOR A — Statistical Realism  (overall score)")
        w("─" * 55)

        blocks_a = [
            ("FIDELITY",      o.fidelity_score,      0.30),
            ("RELATIONSHIPS", o.relationships_score,  0.25),
            ("UTILITY",       o.utility_score,        0.30),
            ("PRIVACY",       o.privacy_score,        0.10),
            ("DIVERSITY",     o.diversity_score,      0.05),
        ]

        for name, score, weight in blocks_a:
            bar = _bar(score)
            sem = _semaphore(score)
            w(f"  {name:<16} {bar}  {score:.2f}  {sem}  ({int(weight*100)}%)")

        w()
        w("─" * 55)
        bar = _bar(o.overall_score)
        sem = _semaphore(o.overall_score)
        w(f"  {'OVERALL SCORE':<16} {bar}  {o.overall_score:.2f}  {sem}")
        w("─" * 55)

        w()
        w("  EVALUATOR B — Adversarial Detectability  (diagnostic only)")
        w("─" * 55)
        r   = o.detectability
        bar = _bar(r.score)
        sem = _semaphore(r.score)
        w(f"  {'DETECTABILITY':<16} {bar}  {r.score:.2f}  {sem}  (not in overall)")
        w(f"  Accuracy={r.accuracy:.3f}  AUC={r.roc_auc:.3f}  (ideal ≈ 0.50)")
        if r.excluded_high_cardinality:
            w(f"  Excluded: {r.excluded_high_cardinality}")
        if r.accuracy > 0.75:
            w("  ⚠️  Some detectable patterns — see DETECTABILITY DETAIL")
        else:
            w("  ✅ No strong detectable patterns")
        w("─" * 55)

    def _fidelity(self, w, o):
        r = o.fidelity
        w()
        w("=" * 55)
        w(f"  FIDELITY DETAIL   score={o.fidelity_score:.2f}  {_semaphore(o.fidelity_score)}")
        w("=" * 55)

        if r.ks_results:
            w()
            w("  KS Test + Wasserstein (numeric columns)")
            w("  Scoring: KS statistic (threshold=0.10), p-value shown for reference only.")
            w(f"  {'Column':<20} {'KS stat':>8}  {'Score':>7}  {'p-value':>9}  {'Pass':>5}  {'Wasserstein':>12}")
            w(f"  {'-'*19} {'-'*8}  {'-'*7}  {'-'*9}  {'-'*5}  {'-'*12}")
            for kr in sorted(r.ks_results, key=lambda x: x.statistic, reverse=True):
                w(f"  {kr.column:<20} {kr.statistic:>8.4f}  {_pct(kr.ks_col_score):>7}  "
                  f"{kr.p_value:>9.4f}  {_pass(kr.passed):>5}  {kr.wasserstein:>12.4f}")
            passed = sum(kr.passed for kr in r.ks_results)
            w(f"\n  KS score={_pct(r.ks_score)}  "
              f"Pass rate (stat<0.10): {passed}/{len(r.ks_results)}  "
              f"Wasserstein score={_pct(r.wasserstein_score)}")

        if r.tvd_excluded:
            w()
            w(f"  Excluded from TVD (unique_ratio > 50% — external generation):")
            w(f"  {r.tvd_excluded}")

        if r.tvd_results:
            w()
            w("  TVD (categorical / ordinal columns)")
            w(f"  {'Column':<20} {'TVD':>8}  {'Pass':>5}")
            w(f"  {'-'*19} {'-'*8}  {'-'*5}")
            seen = set()
            for tr in sorted(r.tvd_results, key=lambda x: x.tvd, reverse=True):
                if tr.column in seen:
                    continue
                seen.add(tr.column)
                w(f"  {tr.column:<20} {tr.tvd:>8.4f}  {_pass(tr.passed):>5}")
            passed = sum(tr.passed for tr in r.tvd_results)
            w(f"\n  TVD passed: {passed}/{len(r.tvd_results)}  score={_pct(r.tvd_score)}")

    def _relationships(self, w, o):
        r = o.relationships
        w()
        w("=" * 55)
        w(f"  RELATIONSHIPS DETAIL   score={o.relationships_score:.2f}  {_semaphore(o.relationships_score)}")
        w("=" * 55)

        w(f"\n  Frobenius norm  Pearson={r.frobenius_pearson:.4f}  "
          f"Spearman={r.frobenius_spearman:.4f}")
        w(f"  Mean Δ Pearson={r.mean_delta_pearson:.4f}  "
          f"Mean Δ Spearman={r.mean_delta_spearman:.4f}")

        if r.pair_results:
            w()
            w("  Correlation pairs (sorted by Pearson delta)")
            w(f"  {'Pair':<28} {'Δ Pearson':>10}  {'Δ Spearman':>10}  {'Preserved':>10}")
            w(f"  {'-'*27} {'-'*10}  {'-'*10}  {'-'*10}")
            for pr in sorted(r.pair_results, key=lambda x: -x.pearson_delta)[:15]:
                pair = f"{pr.col_a} ↔ {pr.col_b}"
                w(f"  {pair:<28} {pr.pearson_delta:>10.4f}  "
                  f"{pr.spearman_delta:>10.4f}  {_pass(pr.preserved):>10}")

            passed = sum(pr.preserved for pr in r.pair_results)
            w(f"\n  Pairs preserved: {passed}/{len(r.pair_results)}  "
              f"pair_score={_pct(r.pair_score)}  frob_score={_pct(r.frob_score)}")

    def _utility(self, w, o):
        r = o.utility
        w()
        w("=" * 55)
        w(f"  UTILITY (TSTR) DETAIL   score={o.utility_score:.2f}  {_semaphore(o.utility_score)}")
        w("=" * 55)
        w(f"\n  Target column : {r.target_column}")
        w(f"  Best model    : {r.best_model}")
        w(f"  Worst model   : {r.worst_model}")
        w(f"  Mean gap      : {_pct(r.mean_gap)}")

        if r.model_results:
            w()
            w(f"  {'Model':<22} {'Acc Real':>9} {'Acc Synth':>10} {'Gap':>7} {'AUC Real':>9} {'AUC Synth':>10} {'Pass':>5}")
            w(f"  {'-'*21} {'-'*9} {'-'*10} {'-'*7} {'-'*9} {'-'*10} {'-'*5}")
            for mr in r.model_results:
                auc_r = f"{mr.auc_real:.4f}"  if mr.auc_real  is not None else "  n/a  "
                auc_s = f"{mr.auc_synth:.4f}" if mr.auc_synth is not None else "  n/a  "
                w(f"  {mr.model_name:<22} {mr.accuracy_real:>9.4f} {mr.accuracy_synth:>10.4f} "
                  f"{mr.gap_accuracy:>7.4f} {auc_r:>9} {auc_s:>10} {_pass(mr.passed):>5}")

    def _detectability(self, w, o):
        r = o.detectability
        w()
        w("=" * 55)
        w(f"  DETECTABILITY DETAIL   score={o.detectability_score:.2f}  {_semaphore(o.detectability_score)}")
        w("=" * 55)
        w(f"\n  Classifier : RandomForest (real vs synthetic label)")
        w(f"  n_real     : {r.n_real}   n_synthetic: {r.n_synthetic}")
        w(f"  Features used      : {r.n_features_used} columns")
        if r.excluded_high_cardinality:
            w(f"  Excluded (high cardinality): {r.excluded_high_cardinality}")
            w(f"  → Names/codes/IDs excluded — trivially distinguishable by construction")
        w()
        w(f"  Accuracy : {r.accuracy:.4f}  (ideal ≈ 0.50)  score={_pct(r.accuracy_score)}")
        w(f"  ROC-AUC  : {r.roc_auc:.4f}  (ideal ≈ 0.50)  score={_pct(r.auc_score)}")

        if r.accuracy > 0.75:
            w("\n  ⚠️  High detectability — synthetic patterns differ noticeably from real.")
        elif r.accuracy < 0.60:
            w("\n  ✅ Low detectability — synthetic data is statistically indistinguishable.")

    def _privacy(self, w, o):
        r = o.privacy
        w()
        w("=" * 55)
        w(f"  PRIVACY DETAIL   score={o.privacy_score:.2f}  {_semaphore(o.privacy_score)}")
        w("=" * 55)
        w(f"\n  Mean NND (synth→real)  : {r.mean_nnd:.4f}")
        w(f"  Min  NND (synth→real)  : {r.min_nnd:.4f}  (most at-risk record)")
        w(f"  Mean NND (real→real)   : {r.mean_nnd_real:.4f}  (reference density)")
        w(f"  Distance ratio         : {r.distance_ratio:.4f}  (ideal ≥ 1.0)")
        w(f"  Records at risk (NND<0.05): {r.n_at_risk}")

        if r.privacy_warning:
            w("\n  ⚠️  Privacy warning: synthetic records are significantly closer")
            w("      to real records than real records are to each other.")
            w("      Re-identification risk detected.")
        else:
            w("\n  ✅ Privacy: synthetic records are sufficiently distant from real data.")

    def _diversity(self, w, o):
        r = o.diversity
        w()
        w("=" * 55)
        w(f"  DIVERSITY DETAIL   score={o.diversity_score:.2f}  {_semaphore(o.diversity_score)}")
        w("=" * 55)

        if r.variance_results:
            w()
            w("  Variance ratio (synth / real) — ideal ≈ 1.0")
            w(f"  {'Column':<20} {'Var Real':>10}  {'Var Synth':>10}  {'Ratio':>7}  {'Score':>7}")
            w(f"  {'-'*19} {'-'*10}  {'-'*10}  {'-'*7}  {'-'*7}")
            for vr in r.variance_results:
                w(f"  {vr.column:<20} {vr.var_real:>10.4f}  {vr.var_synth:>10.4f}  "
                  f"{vr.ratio:>7.3f}  {_pct(vr.score):>7}")
            w(f"\n  Variance score: {_pct(r.variance_score)}")

        if r.pca_results:
            w()
            w("  PCA spread (variance per component — ideal ratio ≈ 1.0)")
            w(f"  {'PC':<6} {'Var Real':>10}  {'Var Synth':>10}  {'Ratio':>7}  {'Score':>7}")
            w(f"  {'-'*5} {'-'*10}  {'-'*10}  {'-'*7}  {'-'*7}")
            for pr in r.pca_results:
                w(f"  PC{pr.component:<4} {pr.var_real:>10.4f}  {pr.var_synth:>10.4f}  "
                  f"{pr.ratio:>7.3f}  {_pct(pr.score):>7}")
            w(f"\n  PCA score: {_pct(r.pca_score)}")

        if r.coverage_results:
            w()
            w("  Category coverage (categorical columns)")
            w(f"  {'Column':<20} {'Real cats':>10}  {'Synth cats':>11}  {'Covered':>8}  {'Coverage':>9}")
            w(f"  {'-'*19} {'-'*10}  {'-'*11}  {'-'*8}  {'-'*9}")
            for cr in r.coverage_results:
                w(f"  {cr.column:<20} {cr.n_real_cats:>10}  {cr.n_synth_cats:>11}  "
                  f"{cr.n_covered:>8}  {_pct(cr.coverage):>9}")
            w(f"\n  Coverage score: {_pct(r.coverage_score)}")

    def _footer(self, w, o):
        w()
        w("=" * 55)
        w("  EVALUATION COMPLETE")
        w("=" * 55)
        w(f"  Overall score : {o.overall_score:.4f}  {_semaphore(o.overall_score)}")
        w()
        w("  Interpretation:")
        w("    🟢 EXCELLENT  (≥ 0.85) — production ready")
        w("    🟡 GOOD       (≥ 0.70) — suitable for most use cases")
        w("    🟠 ACCEPTABLE (≥ 0.55) — use with caution")
        w("    🔴 POOR       (< 0.55) — not recommended for production")
        w("=" * 55)