"""
column_builder.py
─────────────────────────────────────────────────────────────────────────────
Phase 4 — Column Builder.

Reconstructs all columns that were excluded from the diffusion model tensor
(Phase 2, Step 1) and adds them to the synthetic DataFrame.

Why columns are excluded from the tensor:
    The diffusion model only learns statistical structure. Columns that carry
    identity, PII, or pure structural patterns (IDs, names, ticket codes)
    would add noise to the model without contributing to the joint distribution.
    They are generated separately here, AFTER the model has produced the
    statistically coherent core columns.

Column types handled (from PreprocessorOutput.excluded_columns):
    sequential_id       → 1, 2, 3, … (same length as synthetic_df)
    faker.name()        → Faker(locale).name()
    faker.first_name()  → Faker(locale).first_name()
    faker.last_name()   → Faker(locale).last_name()
    faker.email()       → Faker(locale).email()
    faker.phone_number()→ Faker(locale).phone_number()
    faker.address()     → Faker(locale).address()
    faker.country()     → Faker(locale).country()
    faker.city()        → Faker(locale).city()
    faker.zipcode()     → Faker(locale).postcode()
    regex_pattern       → random string matching the regex from ColumnProfile
    datetime_range      → random date between real min_date and max_date
    redundant           → reconstructed from its source column via value mapping
                          (already handled by inverse_transform — skipped here)
    skip / free_text    → column is omitted entirely (cannot be generated)

Realism guarantees:
    - missing_ratio preserved: if Name had 2% nulls in the real data,
      2% of synthetic Names will also be NaN.
    - regex_pattern generation respects sub-pattern frequencies observed
      in the real data (dominant patterns get proportional representation).
    - sequential_id always starts at 1 and has no gaps.
    - Faker locale is read from ColumnProfile.locale (set by SemanticAnalyzer).

Dependencies:
    pandas, numpy, faker, re, random

Part of:
    Synthetic Data Engine — Auto-Adaptive Diffusion System
"""

import re
import random
import string
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from analyzer.dataset_analyzer  import DatasetProfile, ColumnProfile
from preprocessor.preprocessor  import PreprocessorOutput


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_LOCALE = "en_US"

# Faker strategies and their method names
FAKER_STRATEGY_MAP = {
    "faker.name()"         : "name",
    "faker.first_name()"   : "first_name",
    "faker.last_name()"    : "last_name",
    "faker.email()"        : "email",
    "faker.phone_number()" : "phone_number",
    "faker.address()"      : "address",
    "faker.country()"      : "country",
    "faker.city()"         : "city",
    "faker.zipcode()"      : "postcode",
}

# ── Gender detection ──────────────────────────────────────────────────────────

# Column name fragments that strongly suggest a gender column (case-insensitive)
GENDER_COLUMN_KEYWORDS = {
    "sex", "gender", "genero", "sexo", "genre",
    "geschlecht",   # German
    "sexe",         # French
    "geslacht",     # Dutch
    "sesso",        # Italian
    "sexualidade",  # Portuguese
    "kjønn",        # Norwegian
    "kön",          # Swedish
    "sukupuoli",    # Finnish
}

# Values that map to MALE in any language/encoding
MALE_VALUES = {
    "male", "m", "man", "men", "hombre", "masculino",
    "masculin", "männlich", "homme", "maschio",
    "mężczyzna", "мужской", "男", "erkek", "1",
}

# Values that map to FEMALE in any language/encoding
FEMALE_VALUES = {
    "female", "f", "woman", "women", "mujer", "femenino",
    "féminin", "weiblich", "femme", "femmina",
    "kobieta", "женский", "女", "kadın", "0",
}


# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass
class BuildReport:
    """Records which columns were built and by what strategy."""
    sequential_built : list[str] = field(default_factory=list)
    faker_built      : list[str] = field(default_factory=list)
    regex_built      : list[str] = field(default_factory=list)
    datetime_built   : list[str] = field(default_factory=list)
    skipped          : list[str] = field(default_factory=list)
    nulls_injected   : dict[str, int] = field(default_factory=dict)


# ── Main builder ──────────────────────────────────────────────────────────────

class ColumnBuilder:
    """
    Generates all columns excluded from the diffusion tensor and inserts
    them into the synthetic DataFrame with realistic distributions.

    Usage:
        builder = ColumnBuilder(profile, prep_output, original_df)
        df_complete, report = builder.build(df_typed)
    """

    def __init__(
        self,
        profile      : DatasetProfile,
        prep_output  : PreprocessorOutput,
        original_df  : pd.DataFrame,
    ):
        self.profile     = profile
        self.prep_output = prep_output
        self.original_df = original_df
        self._fakers: dict[str, object] = {}   # locale → Faker instance (lazy)
        self._gender_col : Optional[str] = None  # detected gender column name

    def build(
        self,
        df : pd.DataFrame,
    ) -> tuple[pd.DataFrame, BuildReport]:
        """
        Add all excluded columns to df and return the complete DataFrame.

        Column insertion order matches the original dataset column order,
        so the final DataFrame has exactly the same column sequence.
        """
        df     = df.copy()
        report = BuildReport()
        n      = len(df)

        # Detect gender column once — used by all name/firstname generators
        self._gender_col = self._detect_gender_column(df)

        excluded = self.prep_output.excluded_columns  # col → reason string

        for col, reason in excluded.items():
            col_profile = self.profile.get_column(col)
            if col_profile is None:
                report.skipped.append(col)
                continue

            strategy = col_profile.generation_strategy or ""

            # ── Sequential ID ──────────────────────────────────────────────
            if col_profile.is_id or strategy == "sequential_id":
                df[col] = self._build_sequential(n)
                df[col] = df[col].astype(
                    self.original_df[col].dtype
                    if col in self.original_df.columns else "int64"
                )
                report.sequential_built.append(col)

            # ── Faker strategies ───────────────────────────────────────────
            elif strategy in FAKER_STRATEGY_MAP:
                values = self._build_faker(col_profile, n, strategy, df)
                df[col] = values
                report.faker_built.append(col)
                nulls   = self._inject_nulls(df, col, col_profile.missing_ratio, n)
                if nulls:
                    report.nulls_injected[col] = nulls

            # ── Regex pattern ──────────────────────────────────────────────
            elif strategy == "regex_pattern":
                values = self._build_regex(col_profile, n)
                df[col] = values
                report.regex_built.append(col)
                nulls   = self._inject_nulls(df, col, col_profile.missing_ratio, n)
                if nulls:
                    report.nulls_injected[col] = nulls

            # ── Datetime range ─────────────────────────────────────────────
            elif strategy == "datetime_range" or col_profile.detected_type == "datetime":
                values = self._build_datetime(col, n)
                df[col] = values
                report.datetime_built.append(col)
                nulls   = self._inject_nulls(df, col, col_profile.missing_ratio, n)
                if nulls:
                    report.nulls_injected[col] = nulls

            # ── Redundant columns — already handled by inverse_transform ───
            elif "redundant" in reason:
                # inverse_transform already reconstructed these — skip
                pass

            # ── Skip / free_text / unknown ─────────────────────────────────
            else:
                report.skipped.append(col)

        # ── Reorder to match original column order ─────────────────────────
        original_order = list(self.original_df.columns)
        cols_present   = [c for c in original_order if c in df.columns]
        df = df[cols_present]

        return df, report

    # ── Sequential ────────────────────────────────────────────────────────────

    def _build_sequential(self, n: int) -> list[int]:
        return list(range(1, n + 1))

    # ── Faker ─────────────────────────────────────────────────────────────────

    def _build_faker(
        self,
        col_profile : ColumnProfile,
        n           : int,
        strategy    : str,
        df          : Optional[pd.DataFrame] = None,
    ) -> list:
        locale = col_profile.locale or DEFAULT_LOCALE
        faker  = self._get_faker(locale)

        # Gender-aware name generation
        if strategy in ("faker.name()", "faker.first_name()"):
            return self._build_gendered_names(faker, strategy, n, df)

        method = FAKER_STRATEGY_MAP[strategy]
        fn     = getattr(faker, method)
        return [fn() for _ in range(n)]

    def _build_gendered_names(
        self,
        faker    : object,
        strategy : str,
        n        : int,
        df       : Optional[pd.DataFrame],
    ) -> list:
        """
        Generate names coherent with the gender column.

        If a gender column is detected in df, each row gets a name matching
        its gender value. Falls back to faker.name() if no gender column found
        or if the value is unrecognised.
        """
        results = []
        for i in range(n):
            gender = self._get_gender_for_row(df, i)
            if strategy == "faker.name()":
                if gender == "male":
                    first = faker.first_name_male()
                elif gender == "female":
                    first = faker.first_name_female()
                else:
                    first = faker.first_name()
                results.append(f"{first} {faker.last_name()}")
            elif strategy == "faker.first_name()":
                if gender == "male":
                    results.append(faker.first_name_male())
                elif gender == "female":
                    results.append(faker.first_name_female())
                else:
                    results.append(faker.first_name())
        return results

    def _detect_gender_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Auto-detect a gender column in the synthetic DataFrame.

        Two-pass strategy:
          Pass 1 — column name contains a known gender keyword (fast, reliable)
          Pass 2 — binary column whose values match male/female vocabulary
                   (catches columns named 'g', 'tipo', 'cat1', etc.)

        Returns the column name if found, None otherwise.
        """
        # Pass 1: name-based detection
        for col in df.columns:
            col_lower = col.lower().strip()
            for kw in GENDER_COLUMN_KEYWORDS:
                if kw in col_lower:
                    return col

        # Pass 2: content-based detection on binary/low-cardinality columns
        for col in df.columns:
            series = df[col].dropna()
            if series.nunique() > 4:
                continue
            values = set(str(v).lower().strip() for v in series.unique())
            if values & MALE_VALUES and values & FEMALE_VALUES:
                return col
            if values <= (MALE_VALUES | FEMALE_VALUES) and values & (MALE_VALUES | FEMALE_VALUES):
                return col

        return None

    def _get_gender_for_row(
        self,
        df  : Optional[pd.DataFrame],
        idx : int,
    ) -> Optional[str]:
        """Return 'male', 'female', or None for row idx based on the gender column."""
        if df is None or self._gender_col is None:
            return None
        if self._gender_col not in df.columns:
            return None
        val = str(df.iloc[idx][self._gender_col]).lower().strip()
        if val in MALE_VALUES:
            return "male"
        if val in FEMALE_VALUES:
            return "female"
        return None

    def _get_faker(self, locale: str):
        """Lazy Faker instantiation — one instance per locale."""
        if locale not in self._fakers:
            try:
                from faker import Faker
                self._fakers[locale] = Faker(locale)
            except ImportError:
                raise ImportError(
                    "Faker is required for generating PII columns. "
                    "Install it with: pip install faker"
                )
        return self._fakers[locale]

    # ── Regex ─────────────────────────────────────────────────────────────────

    def _build_regex(
        self,
        col_profile : ColumnProfile,
        n           : int,
    ) -> list[Optional[str]]:
        """
        Generate n strings that match col_profile.regex_pattern.

        Strategy:
            1. Parse the real values from original_df to extract sub-pattern
               frequencies (dominant structures observed by RegexMiner).
            2. Generate new strings proportionally to those frequencies.
            3. Fall back to generic regex generation if real values unavailable.

        This preserves the structural distribution of the real column:
        e.g. if Ticket has 40% purely numeric and 60% "PREFIX/NUMBER",
        the synthetic column will have the same proportions.
        """
        col_name = col_profile.column_name
        pattern  = col_profile.regex_pattern

        # ── Learn structure from real values ──────────────────────────────
        if col_name in self.original_df.columns:
            real_vals = (
                self.original_df[col_name]
                .dropna()
                .astype(str)
                .tolist()
            )
            return self._generate_from_real_examples(real_vals, n, pattern)

        # ── Fallback: pure regex generation ───────────────────────────────
        if pattern:
            return [self._generate_from_regex(pattern) for _ in range(n)]

        # ── Last resort: alphanumeric placeholder ──────────────────────────
        return ["".join(random.choices(string.ascii_uppercase + string.digits, k=6))
                for _ in range(n)]

    def _generate_from_real_examples(
        self,
        real_vals : list[str],
        n         : int,
        pattern   : Optional[str],
    ) -> list[str]:
        """
        Learn structural templates from real values and generate new ones.

        Each real value is abstracted into a template:
            "A/1234"    → "{upper}/{digits4}"
            "PC 17599"  → "{upper2} {digits5}"
            "347082"    → "{digits6}"

        Templates are counted by frequency. New values are generated by
        sampling templates proportionally and filling placeholders randomly.
        """
        templates: dict[str, int] = {}

        for val in real_vals:
            tmpl = self._abstract_to_template(val)
            templates[tmpl] = templates.get(tmpl, 0) + 1

        # Sort by frequency descending
        sorted_tmpls = sorted(templates.items(), key=lambda x: -x[1])

        # Compute sampling weights
        total   = sum(c for _, c in sorted_tmpls)
        weights = [c / total for _, c in sorted_tmpls]
        tmpls   = [t for t, _ in sorted_tmpls]

        results = []
        for _ in range(n):
            tmpl = random.choices(tmpls, weights=weights, k=1)[0]
            results.append(self._fill_template(tmpl))

        return results

    def _abstract_to_template(self, val: str) -> str:
        """
        Convert a value to a structural template string.

        Rules:
            A-Z block  → {upperN}
            a-z block  → {lowerN}
            0-9 block  → {digitsN}
            separator  → kept literally
        """
        result = []
        i = 0
        while i < len(val):
            c = val[i]
            if c.isupper():
                j = i
                while j < len(val) and val[j].isupper():
                    j += 1
                result.append(f"{{upper{j-i}}}")
                i = j
            elif c.islower():
                j = i
                while j < len(val) and val[j].islower():
                    j += 1
                result.append(f"{{lower{j-i}}}")
                i = j
            elif c.isdigit():
                j = i
                while j < len(val) and val[j].isdigit():
                    j += 1
                result.append(f"{{digits{j-i}}}")
                i = j
            else:
                result.append(c)  # separator — keep literal
                i += 1
        return "".join(result)

    def _fill_template(self, template: str) -> str:
        """
        Fill a structural template with random characters.

        {upperN}  → N random uppercase letters
        {lowerN}  → N random lowercase letters
        {digitsN} → N random digits
        Literals  → kept as-is
        """
        def replacer(m):
            kind = m.group(1)
            n    = int(m.group(2))
            if kind == "upper":
                return "".join(random.choices(string.ascii_uppercase, k=n))
            elif kind == "lower":
                return "".join(random.choices(string.ascii_lowercase, k=n))
            elif kind == "digits":
                # First digit never 0 for multi-digit numbers
                first = random.choice(string.digits[1:]) if n > 1 else random.choice(string.digits)
                rest  = "".join(random.choices(string.digits, k=n - 1))
                return first + rest
            return m.group(0)

        return re.sub(r"\{(upper|lower|digits)(\d+)\}", replacer, template)

    def _generate_from_regex(self, pattern: str) -> str:
        """
        Very simple regex-to-string generator for fallback use.
        Handles common patterns: [A-Z], [0-9], \\d, \\w, {n,m}, ?, +, *
        """
        try:
            import rstr
            return rstr.xeger(pattern)
        except ImportError:
            pass

        # Manual fallback for simple patterns
        result = []
        i = 0
        while i < len(pattern):
            c = pattern[i]
            if c == "^" or c == "$":
                i += 1
            elif c == "\\" and i + 1 < len(pattern):
                nxt = pattern[i + 1]
                if nxt == "d":
                    result.append(random.choice(string.digits))
                elif nxt == "w":
                    result.append(random.choice(string.ascii_letters + string.digits + "_"))
                elif nxt == "s":
                    result.append(" ")
                else:
                    result.append(nxt)
                i += 2
            elif c == "[":
                j   = pattern.index("]", i)
                cls = pattern[i + 1:j]
                result.append(self._sample_char_class(cls))
                i = j + 1
            elif c == "{":
                j   = pattern.index("}", i)
                rng = pattern[i + 1:j].split(",")
                lo  = int(rng[0])
                hi  = int(rng[-1]) if len(rng) > 1 else lo
                repeat = random.randint(lo, hi)
                if result:
                    last = result[-1]
                    result = result[:-1] + [last] * repeat
                i = j + 1
            elif c in ".?+*":
                i += 1
            else:
                result.append(c)
                i += 1

        return "".join(result)

    def _sample_char_class(self, cls: str) -> str:
        """Sample a single character from a regex character class like A-Z0-9."""
        chars = []
        i     = 0
        while i < len(cls):
            if i + 2 < len(cls) and cls[i + 1] == "-":
                lo = ord(cls[i])
                hi = ord(cls[i + 2])
                chars.extend(chr(c) for c in range(lo, hi + 1))
                i += 3
            else:
                chars.append(cls[i])
                i += 1
        return random.choice(chars) if chars else "X"

    # ── Datetime ──────────────────────────────────────────────────────────────

    def _build_datetime(self, col: str, n: int) -> list:
        """
        Generate random dates between the real min and max dates.
        Falls back to a reasonable default range if the column is not available.
        """
        if col in self.original_df.columns:
            real = pd.to_datetime(self.original_df[col], errors="coerce").dropna()
            if not real.empty:
                ts_min = real.min().timestamp()
                ts_max = real.max().timestamp()
                ts     = [random.uniform(ts_min, ts_max) for _ in range(n)]
                return pd.to_datetime(ts, unit="s").normalize().tolist()

        # Fallback: last 5 years
        import datetime
        end   = datetime.date.today()
        start = end.replace(year=end.year - 5)
        delta = (end - start).days
        return [
            start + datetime.timedelta(days=random.randint(0, delta))
            for _ in range(n)
        ]

    # ── Null injection ────────────────────────────────────────────────────────

    def _inject_nulls(
        self,
        df            : pd.DataFrame,
        col           : str,
        missing_ratio : float,
        n             : int,
    ) -> int:
        """
        Inject NaN values into col at the same rate as in the original data.

        Returns the number of nulls injected.
        """
        if missing_ratio <= 0:
            return 0

        n_nulls = max(1, int(round(missing_ratio * n)))
        idx     = random.sample(range(n), min(n_nulls, n))
        df.loc[idx, col] = np.nan
        return len(idx)