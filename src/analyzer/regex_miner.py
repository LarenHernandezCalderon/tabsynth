"""
regex_miner.py
─────────────────────────────────────────────────────────────────────────────
Local regex pattern miner for string columns classified as 'code' by the
SemanticAnalyzer (Layer 2).

Privacy note:
    This module runs ENTIRELY locally. Values are analyzed in memory and
    never stored, logged, or sent anywhere. The only output is a regex
    string — a structural description, not a data copy.

Why this exists:
    The SemanticAnalyzer (GPT) can only see structural metadata (lengths,
    ratios, patterns). That's enough to detect a 'code' column, but not
    enough to infer the actual regex with precision.

    Examples of what GPT gets wrong with only metadata:
        - Ticket: ^[A-Za-z0-9]{4,16}$  (too generic — misses prefix groups)
        - Cabin:  ^[A-Z][0-9]{1,3}$    (misses multi-cabin: "C23 C25 C27")

    This module sees the real values and infers structure from them.

What it does:
    1. Groups values by structural pattern (prefix, separator, suffix)
    2. Detects the dominant pattern(s) covering ≥ 80% of values
    3. Builds a regex that covers all dominant patterns via alternation
    4. Falls back to the GPT regex if mining fails

Usage:
    miner = RegexMiner()
    result = miner.mine(column_name, series)
    # result.regex     → inferred regex string
    # result.coverage  → fraction of values matching the regex
    # result.patterns  → list of sub-patterns found
    # result.notes     → human-readable explanation

Part of:
    Synthetic Tabular Data Engine — Auto-Adaptive Diffusion System
"""

import re
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


# ── Config ────────────────────────────────────────────────────────────────────
MIN_COVERAGE     = 0.80    # regex must match at least 80% of non-null values
MAX_PATTERNS     = 5       # max sub-patterns in alternation regex
PREFIX_MIN_FREQ  = 0.05    # a prefix must appear in ≥ 5% of values to count


# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass
class RegexMineResult:
    regex     : str
    coverage  : float          # 0.0 → 1.0
    patterns  : list[str]      # sub-patterns found (before combining)
    notes     : str
    fallback  : bool = False   # True if we fell back to the GPT regex


# ── Main miner ────────────────────────────────────────────────────────────────

class RegexMiner:
    """
    Analyzes a string column locally and returns a precise regex pattern.

    Strategy:
        1. Detect multi-value entries (e.g. "C23 C25 C27") → wrap in repeat group
        2. Cluster values by structural token pattern
        3. Build a regex alternation covering dominant clusters
        4. Validate coverage against the full column
    """

    def mine(
        self,
        column_name : str,
        series      : pd.Series,
        gpt_regex   : Optional[str] = None,
    ) -> RegexMineResult:
        """
        Main entry point. Returns the best regex found.
        Falls back to gpt_regex if mining fails or coverage is too low.
        """
        clean = series.dropna().astype(str)
        if len(clean) == 0:
            return self._fallback(gpt_regex, "Empty column")

        try:
            # Step 1: check for multi-value entries (space-separated repeating units)
            if self._is_multi_value(clean):
                result = self._mine_multi_value(clean)
                if result and result.coverage >= MIN_COVERAGE:
                    return result

            # Step 2: cluster by structural token pattern
            result = self._mine_by_token_structure(clean)
            if result and result.coverage >= MIN_COVERAGE:
                return result

            # Step 3: fall back to GPT regex
            return self._fallback(gpt_regex, f"No pattern reached {MIN_COVERAGE:.0%} coverage")

        except Exception as e:
            return self._fallback(gpt_regex, f"Mining error: {e}")

    # ── Multi-value detector ──────────────────────────────────────────────────

    def _is_multi_value(self, series: pd.Series) -> bool:
        """
        Detects if some values are space-separated repetitions of the same
        unit pattern — e.g. Cabin: "C23 C25 C27"
        """
        multi_count = series.str.contains(r' ').sum()
        return (multi_count / len(series)) > 0.10   # >10% have spaces

    def _mine_multi_value(self, series: pd.Series) -> Optional[RegexMineResult]:
        """
        For columns like Cabin where values can be "C23" OR "C23 C25 C27",
        finds the unit pattern (single token) and wraps it in a repeat group.
        """
        # Split all values into tokens
        all_tokens = []
        for val in series:
            all_tokens.extend(str(val).split())

        token_series = pd.Series(all_tokens)
        # Mine the unit pattern from individual tokens
        unit_result = self._mine_by_token_structure(token_series)

        if not unit_result or unit_result.coverage < MIN_COVERAGE:
            return None

        unit_rx = unit_result.regex.strip("^$")
        # Wrap: one or more units separated by spaces
        combined_rx = f"^({unit_rx})( {unit_rx})*$"

        coverage = self._measure_coverage(series, combined_rx)

        return RegexMineResult(
            regex    = combined_rx,
            coverage = coverage,
            patterns = [f"Unit: {unit_rx}"],
            notes    = (
                f"Multi-value column detected. Unit pattern: '{unit_rx}'. "
                f"Supports 1..N space-separated units. Coverage: {coverage:.0%}"
            ),
        )

    # ── Token structure miner ─────────────────────────────────────────────────

    def _mine_by_token_structure(self, series: pd.Series) -> Optional[RegexMineResult]:
        """
        Clusters values by their character-class token pattern,
        then builds one regex per dominant cluster.

        Token pattern example:
            "PC 1234"  →  "[A-Z]+\\s\\d+"
            "A12"      →  "[A-Z]\\d{2}"
        """
        # Build a structural fingerprint per value
        fingerprints: dict[str, list[str]] = {}

        for val in series:
            fp = self._fingerprint(str(val))
            if fp not in fingerprints:
                fingerprints[fp] = []
            fingerprints[fp].append(str(val))

        total   = len(series)
        sorted_fps = sorted(fingerprints.items(), key=lambda x: len(x[1]), reverse=True)

        # Keep dominant patterns (covering enough values)
        dominant: list[tuple[str, list[str]]] = []
        covered = 0
        for fp, vals in sorted_fps[:MAX_PATTERNS]:
            ratio = len(vals) / total
            if ratio >= PREFIX_MIN_FREQ:
                dominant.append((fp, vals))
                covered += len(vals)

        if not dominant:
            return None

        # Convert each fingerprint to a concrete regex
        sub_patterns: list[str] = []
        for fp, vals in dominant:
            rx = self._fingerprint_to_regex(fp, vals)
            sub_patterns.append(rx)

        # Combine into alternation if multiple
        if len(sub_patterns) == 1:
            final_rx = f"^{sub_patterns[0]}$"
        else:
            alts = "|".join(f"({p})" for p in sub_patterns)
            final_rx = f"^({alts})$"

        coverage = self._measure_coverage(series, final_rx)

        return RegexMineResult(
            regex    = final_rx,
            coverage = coverage,
            patterns = sub_patterns,
            notes    = (
                f"Found {len(sub_patterns)} structural pattern(s) from "
                f"{len(dominant)} dominant cluster(s). Coverage: {coverage:.0%}"
            ),
        )

    # ── Fingerprinting ────────────────────────────────────────────────────────

    def _fingerprint(self, value: str) -> str:
        """
        Converts a value to a character-class sequence.
        Collapses consecutive chars of the same class.

        Examples:
            "PC 1234"    → "U+ D+"      (uppercase, space, digits)
            "SOTON/O2"   → "U+/U+D+"
            "A12"        → "UD+"
            "12345"      → "D+"
        """
        result = []
        i = 0
        while i < len(value):
            c = value[i]
            if c.isupper():
                cls = "U"
            elif c.islower():
                cls = "L"
            elif c.isdigit():
                cls = "D"
            elif c == " ":
                cls = " "
            elif c == "/":
                cls = "/"
            elif c == ".":
                cls = "."
            elif c == "-":
                cls = "-"
            else:
                cls = "X"   # other special char

            # Collapse consecutive same class (except separators)
            if result and result[-1].rstrip("+") == cls and cls not in (" ", "/", ".", "-", "X"):
                if not result[-1].endswith("+"):
                    result[-1] = cls + "+"
            else:
                result.append(cls)
            i += 1

        return "".join(result)

    def _fingerprint_to_regex(self, fingerprint: str, examples: list[str]) -> str:
        """
        Converts a fingerprint back to a concrete regex.
        Uses actual length stats from examples to bound quantifiers.
        """
        rx = ""
        i  = 0
        fp = fingerprint

        while i < len(fp):
            token = fp[i]
            plus  = (i + 1 < len(fp) and fp[i + 1] == "+")

            if token == "U":
                if plus:
                    # Estimate max length from examples
                    upper_lens = self._segment_lengths(examples, r'[A-Z]+')
                    mn, mx = self._len_bounds(upper_lens)
                    rx += f"[A-Z]{{{mn},{mx}}}" if mn != mx else f"[A-Z]{{{mn}}}"
                    i += 2
                else:
                    rx += "[A-Z]"
                    i  += 1

            elif token == "L":
                if plus:
                    lower_lens = self._segment_lengths(examples, r'[a-z]+')
                    mn, mx = self._len_bounds(lower_lens)
                    rx += f"[a-z]{{{mn},{mx}}}" if mn != mx else f"[a-z]{{{mn}}}"
                    i  += 2
                else:
                    rx += "[a-z]"
                    i  += 1

            elif token == "D":
                if plus:
                    digit_lens = self._segment_lengths(examples, r'\d+')
                    mn, mx = self._len_bounds(digit_lens)
                    rx += f"\\d{{{mn},{mx}}}" if mn != mx else f"\\d{{{mn}}}"
                    i  += 2
                else:
                    rx += "\\d"
                    i  += 1

            elif token == " ":
                rx += "\\s"
                i  += 1

            elif token == "/":
                rx += "/"
                i  += 1

            elif token == ".":
                rx += "\\."
                i  += 1

            elif token == "-":
                rx += "-"
                i  += 1

            else:   # X — unknown special char
                rx += "[^A-Za-z0-9]"
                i  += 1

        return rx

    # ── Length helpers ────────────────────────────────────────────────────────

    def _segment_lengths(self, examples: list[str], pattern: str) -> list[int]:
        """Extracts lengths of all regex matches across all examples."""
        lengths = []
        for val in examples[:200]:   # cap for speed
            for m in re.findall(pattern, val):
                lengths.append(len(m))
        return lengths if lengths else [1]

    def _len_bounds(self, lengths: list[int]) -> tuple[int, int]:
        """Returns (min, max) length bounds with a small tolerance."""
        mn = max(1, min(lengths))
        mx = max(lengths)
        return mn, mx

    # ── Coverage validator ────────────────────────────────────────────────────

    def _measure_coverage(self, series: pd.Series, pattern: str) -> float:
        """Fraction of non-null values that match the regex."""
        try:
            matches = series.str.match(pattern, na=False).sum()
            return round(matches / len(series), 4)
        except Exception:
            return 0.0

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _fallback(
        self,
        gpt_regex : Optional[str],
        reason    : str,
    ) -> RegexMineResult:
        rx = gpt_regex or r"^.+$"
        return RegexMineResult(
            regex    = rx,
            coverage = 0.0,
            patterns = [],
            notes    = f"Fallback to GPT regex. Reason: {reason}",
            fallback = True,
        )