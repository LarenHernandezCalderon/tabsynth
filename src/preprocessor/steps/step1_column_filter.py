"""
step1_column_filter.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 — Step 1: Column filtering.

Decides which columns enter the diffusion tensor and which are excluded.
Also detects and removes perfectly redundant column pairs (corr ≈ 1.0),
saving the value mapping needed to reconstruct them after generation.

Excluded categories:
    - should_skip columns (Phase 1 flagged)
    - id_like columns → sequential_id in Phase 4
    - faker.* / regex_pattern / datetime → external generation in Phase 4
    - perfectly redundant pairs (correlation >= REDUNDANCY_THRESHOLD)
"""

import pandas as pd
from analyzer.dataset_analyzer import DatasetProfile
from preprocessor.contracts import EXCLUDED_STRATEGIES, REDUNDANCY_THRESHOLD


class ColumnFilter:
    """
    Filters columns and detects redundancies.

    Usage:
        filt = ColumnFilter(profile, df_original, verbose=True)
        df_filtered, excluded, redundant_pairs, redundant_map, redundant_mappers
            = filt.fit_transform(df)
    """

    def __init__(
        self,
        profile     : DatasetProfile,
        df_original : pd.DataFrame,
        verbose     : bool = True,
    ):
        self.profile     = profile
        self.df_original = df_original
        self.verbose     = verbose

        self.excluded           : dict[str, str]  = {}
        self.redundant_pairs    : list[tuple]     = []
        self.redundant_map      : dict[str, str]  = {}
        self.redundant_mappers  : dict[str, dict] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print("\nStep 1 — Column filtering")
            print("-" * 40)

        keep_cols = []

        for col_profile in self.profile.columns:
            col = col_profile.column_name

            if col_profile.should_skip:
                self.excluded[col] = f"skip: {col_profile.skip_reason or 'flagged for skipping'}"
                continue

            if col_profile.is_id:
                self.excluded[col] = "id_like: regenerated as sequential integer (1, 2, 3…)"
                continue

            strategy = col_profile.generation_strategy or ""
            if strategy in EXCLUDED_STRATEGIES:
                self.excluded[col] = f"external generation: strategy='{strategy}'"
                continue

            if col_profile.detected_type == "datetime":
                self.excluded[col] = "datetime: generated from date range in Phase 4"
                continue

            keep_cols.append(col)

        keep_cols = self._remove_redundant_columns(keep_cols, df)
        available = [c for c in keep_cols if c in df.columns]
        df_out    = df[available].copy()

        if self.verbose:
            print(f"  Columns to tensor    : {len(available)}  {available}")
            print(f"  Excluded columns     : {len(self.excluded)}")
            for col, reason in self.excluded.items():
                print(f"    ✗ {col:<25} → {reason}")
            if self.redundant_pairs:
                print(f"  Redundant pairs      : {len(self.redundant_pairs)}")
                for a, b, score in self.redundant_pairs:
                    print(f"    ≡ {a} ↔ {b}  (score={score:.3f}) — removed {b}")

        return df_out

    def _remove_redundant_columns(
        self,
        cols : list[str],
        df   : pd.DataFrame,
    ) -> list[str]:
        to_remove = set()

        # Pearson matrix (numeric pairs)
        if self.profile.correlation_matrix is not None:
            matrix      = self.profile.correlation_matrix
            numeric_cols = [c for c in cols if c in matrix.columns]

            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    a, b = numeric_cols[i], numeric_cols[j]
                    if a in to_remove or b in to_remove:
                        continue
                    score = abs(matrix.loc[a, b])
                    if score >= REDUNDANCY_THRESHOLD:
                        self.redundant_pairs.append((a, b, score))
                        to_remove.add(b)

        # All pairs (covers mixed / categorical)
        for pair in self.profile.all_pairs:
            a, b = pair.col_a, pair.col_b
            if a not in cols or b not in cols:
                continue
            if a in to_remove or b in to_remove:
                continue
            if pair.score >= REDUNDANCY_THRESHOLD:
                if (a, b, pair.score) not in self.redundant_pairs:
                    self.redundant_pairs.append((a, b, pair.score))
                to_remove.add(b)

        # Build value mappings for reconstruction
        for a, b, score in self.redundant_pairs:
            if a in self.df_original.columns and b in self.df_original.columns:
                mapping_df = self.df_original[[a, b]].dropna().drop_duplicates()
                mapper = {row[a]: row[b] for _, row in mapping_df.iterrows()}
                self.redundant_map[b]     = a
                self.redundant_mappers[b] = mapper

            self.excluded[b] = (
                f"redundant: correlation={score:.3f} — "
                f"will be reconstructed from '{a}' in inverse_transform"
            )

        return [c for c in cols if c not in to_remove]