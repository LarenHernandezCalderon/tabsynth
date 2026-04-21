"""
semantic_analyzer.py
─────────────────────────────────────────────────────────────────────────────
Layer 2 of the analysis pipeline. Classifies string columns that Layer 1
(MetadataExtractor) could not resolve locally (flagged as 'unknown_string').

Core privacy principle:
    ONLY safe structural metadata is sent to the GPT API.
    Real data values NEVER leave the local environment.

What it receives:
    A list of ColumnMetadata objects where detected_type == 'unknown_string',
    each containing only aggregated signals (lengths, ratios, patterns).

What it does:
    Builds a safe metadata payload per column → sends to GPT API →
    parses the structured JSON response → returns a SemanticResult per column.

What it produces (SemanticResult) per column:
    - semantic_type       : classified type (full_name, email, free_text, ...)
    - confidence          : 0.0 to 1.0
    - generatable         : whether synthetic data can be generated for this column
    - generation_strategy : how to generate it (e.g. faker.name())
    - locale              : detected Faker locale (e.g. 'es_ES', 'en_US') or None
    - regex_pattern       : inferred regex pattern for code columns or None
    - pii_detected        : True if column contains Personal Identifiable Information
    - pii_category        : GDPR PII category (direct_identifier, quasi_identifier, ...)
    - notes               : GPT explanation of its reasoning

GDPR PII categories:
    direct_identifier   : uniquely identifies a person (name, email, phone, ID)
    quasi_identifier    : can identify when combined (age, city, zipcode, gender)
    sensitive           : special category under GDPR Art. 9 (health, religion, ...)
    non_pii             : no personal data (product codes, ticket numbers, etc.)

Supported semantic types and generation strategies:
    first_name   → faker.first_name()         + locale detection  [PII: direct]
    last_name    → faker.last_name()           + locale detection  [PII: direct]
    full_name    → faker.name()                + locale detection  [PII: direct]
    email        → faker.email()                                   [PII: direct]
    phone        → faker.phone_number()        + locale detection  [PII: direct]
    address      → faker.address()             + locale detection  [PII: direct]
    country      → faker.country()             + locale detection  [PII: quasi]
    city         → faker.city()                + locale detection  [PII: quasi]
    zipcode      → faker.zipcode()             + locale detection  [PII: quasi]
    code         → regex_pattern generation    + missing_ratio preserved  [non_pii]
    free_text    → not generatable in v1 (flagged)
    unknown      → skipped with warning

Supported Faker locales:
    en_US, en_GB, es_ES, es_MX, fr_FR, de_DE, it_IT,
    pt_BR, pt_PT, nl_NL, pl_PL, ru_RU, ja_JP, zh_CN, ar_AA

Dependencies:
    openai, dataclasses

Part of:
    Synthetic Tabular Data Engine — Auto-Adaptive Diffusion System
"""

import json
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI

from analyzer.metadata_extractor import ColumnMetadata, StringMetadata


# ── Supported semantic types ──────────────────────────────────────────────────
GENERATABLE_TYPES = {
    "first_name",
    "last_name",
    "full_name",
    "email",
    "phone",
    "address",
    "country",
    "city",
    "zipcode",
    "code",          # generatable via regex pattern
}

NON_GENERATABLE_TYPES = {
    "free_text",
    "unknown",
}

GENERATION_STRATEGIES = {
    "first_name" : "faker.first_name()",
    "last_name"  : "faker.last_name()",
    "full_name"  : "faker.name()",
    "email"      : "faker.email()",
    "phone"      : "faker.phone_number()",
    "address"    : "faker.address()",
    "country"    : "faker.country()",
    "city"       : "faker.city()",
    "zipcode"    : "faker.zipcode()",
    "code"       : "regex_pattern",
    "free_text"  : None,
    "unknown"    : None,
}

# Locale-aware types — GPT will attempt to detect the locale for these
LOCALE_AWARE_TYPES = {
    "first_name",
    "last_name",
    "full_name",
    "phone",
    "address",
    "country",
    "city",
    "zipcode",
}

# Supported Faker locales
SUPPORTED_LOCALES = [
    "en_US", "en_GB", "es_ES", "es_MX", "fr_FR", "de_DE",
    "it_IT", "pt_BR", "pt_PT", "nl_NL", "pl_PL", "ru_RU",
    "ja_JP", "zh_CN", "ar_AA",
]


# ── GDPR PII classification ───────────────────────────────────────────────────

# Maps semantic_type → (pii_detected, pii_category)
# Based on GDPR Article 4(1) and Article 9 definitions
PII_CLASSIFICATION: dict[str, tuple[bool, str]] = {
    # Direct identifiers — uniquely identify a natural person
    "full_name"  : (True,  "direct_identifier"),
    "first_name" : (True,  "direct_identifier"),
    "last_name"  : (True,  "direct_identifier"),
    "email"      : (True,  "direct_identifier"),
    "phone"      : (True,  "direct_identifier"),
    "address"    : (True,  "direct_identifier"),

    # Quasi-identifiers — can re-identify when combined with other data
    "country"    : (True,  "quasi_identifier"),
    "city"       : (True,  "quasi_identifier"),
    "zipcode"    : (True,  "quasi_identifier"),

    # Non-PII
    "code"       : (False, "non_pii"),
    "free_text"  : (False, "non_pii"),    # could contain PII but unstructured
    "unknown"    : (False, "non_pii"),
}

# GDPR category descriptions for reporting
PII_CATEGORY_LABELS = {
    "direct_identifier" : "Direct identifier (GDPR Art. 4(1)) — uniquely identifies a person",
    "quasi_identifier"  : "Quasi-identifier — re-identification risk when combined",
    "sensitive"         : "Special category (GDPR Art. 9) — requires explicit consent",
    "non_pii"           : "No personal data detected",
}


# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass
class SemanticResult:
    """Result of GPT semantic classification for a single column."""
    column_name         : str
    semantic_type       : str
    confidence          : float
    generatable         : bool
    generation_strategy : Optional[str]
    locale              : Optional[str]   # Faker locale, e.g. 'es_ES'
    regex_pattern       : Optional[str]   # initial regex from GPT (may be refined by RegexMiner)
    pii_detected        : bool            # True if column contains PII
    pii_category        : str             # direct_identifier | quasi_identifier | sensitive | non_pii
    notes               : str


# ── GPT prompt builder ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a data schema analyst specialized in identifying the semantic meaning
of dataset columns based ONLY on structural metadata — never on actual values.

Your job is to:
1. Classify what kind of real-world information a column contains.
2. For locale-aware types (names, addresses, phones, cities, etc.),
   infer the most likely language/region locale from the column name
   and structural signals.
3. For code columns, infer the most likely regex pattern that describes
   the structure of the values based on length distributions and
   character composition signals.

You must respond ONLY with a valid JSON object. No explanation, no markdown,
no extra text — just the raw JSON.
""".strip()


def _build_user_prompt(meta: ColumnMetadata) -> str:
    """
    Builds the GPT prompt from safe metadata only.
    No real values are ever included.
    """
    sm: StringMetadata = meta.string_metadata

    payload = {
        "column_name"            : meta.column_name,
        "missing_ratio"          : meta.missing_ratio,
        "unique_ratio"           : sm.unique_ratio,
        "avg_char_length"        : sm.avg_char_length,
        "avg_word_count"         : sm.avg_word_count,
        "starts_uppercase_ratio" : sm.starts_uppercase_ratio,
        "has_numbers_ratio"      : sm.has_numbers_ratio,
        "has_special_chars_ratio": sm.has_special_chars_ratio,
        "sample_lengths"         : sm.sample_lengths,
    }

    valid_types  = list(GENERATION_STRATEGIES.keys())
    locale_aware = list(LOCALE_AWARE_TYPES)

    return f"""
Classify the semantic type of this dataset column using ONLY the metadata below.
Do NOT assume any actual data values exist — only use the structural signals provided.

Column metadata:
{json.dumps(payload, indent=2)}

Valid semantic types: {valid_types}

Locale-aware types (infer locale if semantic_type is one of these): {locale_aware}
Supported locales: {SUPPORTED_LOCALES}

For CODE columns: infer the most likely regex pattern from the length distribution
and character composition (has_numbers_ratio, has_special_chars_ratio,
starts_uppercase_ratio, avg_char_length, sample_lengths).

Respond ONLY with this exact JSON structure:
{{
  "semantic_type": "<one of the valid types above>",
  "confidence": <float between 0.0 and 1.0>,
  "generatable": <true or false>,
  "generation_strategy": "<faker method, 'regex_pattern', or null>",
  "locale": "<faker locale string or null if not applicable>",
  "regex_pattern": "<inferred regex string or null if not a code column>",
  "notes": "<brief reasoning based only on the metadata signals>"
}}
""".strip()


# ── Main analyzer ─────────────────────────────────────────────────────────────

class SemanticAnalyzer:
    """
    GPT-powered semantic classifier for string columns.

    Receives only privacy-safe metadata payloads — no real data values.
    Makes one API call per unknown_string column.

    Changes in v3:
        - PII detection: pii_detected + pii_category added to SemanticResult
        - PII classification is LOCAL — no extra API call needed
        - PII is derived deterministically from semantic_type using GDPR rules
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client  = OpenAI(api_key=api_key)
        self.model   = model
        self.results : list[SemanticResult] = []

    def analyze(self, columns: list[ColumnMetadata]) -> list[SemanticResult]:
        unknown_columns = [
            col for col in columns
            if col.detected_type == "unknown_string" and col.string_metadata is not None
        ]

        if not unknown_columns:
            print("✅ No unknown_string columns found. Layer 2 not needed.")
            return []

        print(f"\n🤖 Semantic Analyzer — {len(unknown_columns)} column(s) to classify")
        print("=" * 55)

        self.results = []

        for meta in unknown_columns:
            print(f"\n🔍 Analyzing: {meta.column_name} ...", end=" ")
            result = self._classify_column(meta)
            self.results.append(result)

            pii_icon = "🔴" if result.pii_detected else "🟢"
            print(f"→ {result.semantic_type} (confidence: {result.confidence:.0%})", end="")
            if result.locale:
                print(f" [locale: {result.locale}]", end="")
            if result.regex_pattern:
                print(f" [regex: {result.regex_pattern}]", end="")
            print(f" {pii_icon} {'PII' if result.pii_detected else 'non-PII'}")

        return self.results

    def _classify_column(self, meta: ColumnMetadata) -> SemanticResult:
        try:
            response = self.client.chat.completions.create(
                model    = self.model,
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": _build_user_prompt(meta)},
                ],
                temperature     = 0,
                max_tokens      = 400,
                response_format = {"type": "json_object"},
            )

            raw    = response.choices[0].message.content
            parsed = json.loads(raw)
            return self._build_result(meta.column_name, parsed)

        except Exception as e:
            print(f"\n  ⚠️  API error for '{meta.column_name}': {e}")
            return SemanticResult(
                column_name         = meta.column_name,
                semantic_type       = "unknown",
                confidence          = 0.0,
                generatable         = False,
                generation_strategy = None,
                locale              = None,
                regex_pattern       = None,
                pii_detected        = False,
                pii_category        = "non_pii",
                notes               = f"Classification failed: {str(e)}",
            )

    def _build_result(self, column_name: str, parsed: dict) -> SemanticResult:
        semantic_type = parsed.get("semantic_type", "unknown")

        if semantic_type not in GENERATION_STRATEGIES:
            semantic_type = "unknown"

        generatable = semantic_type in GENERATABLE_TYPES
        strategy    = GENERATION_STRATEGIES.get(semantic_type)

        # Validate locale
        raw_locale = parsed.get("locale")
        locale = raw_locale if raw_locale in SUPPORTED_LOCALES else None

        # Only keep regex for code columns (GPT first pass — may be refined by RegexMiner)
        regex_pattern = None
        if semantic_type == "code":
            regex_pattern = parsed.get("regex_pattern") or None

        # ── PII classification (local, deterministic, no API call) ────────────
        pii_detected, pii_category = PII_CLASSIFICATION.get(
            semantic_type, (False, "non_pii")
        )

        return SemanticResult(
            column_name         = column_name,
            semantic_type       = semantic_type,
            confidence          = float(parsed.get("confidence", 0.0)),
            generatable         = generatable,
            generation_strategy = strategy,
            locale              = locale,
            regex_pattern       = regex_pattern,
            pii_detected        = pii_detected,
            pii_category        = pii_category,
            notes               = parsed.get("notes", ""),
        )

    def summary(self) -> None:
        if not self.results:
            print("No semantic results to display.")
            return

        print("\n🧠 Semantic Analysis Report (Layer 2)")
        print("=" * 55)

        for result in self.results:
            gen_icon = "✅" if result.generatable else "⚠️ "
            pii_icon = "🔴" if result.pii_detected else "🟢"

            print(f"\n{gen_icon} {result.column_name}")
            print(f"   semantic_type : {result.semantic_type}")
            print(f"   confidence    : {result.confidence:.0%}")
            print(f"   generatable   : {result.generatable}")
            print(f"   strategy      : {result.generation_strategy or 'N/A — will be skipped'}")
            if result.locale:
                print(f"   locale        : {result.locale}")
            if result.regex_pattern:
                print(f"   regex_pattern : {result.regex_pattern}")
            print(f"   {pii_icon} pii_detected  : {result.pii_detected}")
            if result.pii_detected:
                label = PII_CATEGORY_LABELS.get(result.pii_category, result.pii_category)
                print(f"   pii_category  : {result.pii_category}")
                print(f"   pii_note      : {label}")
            print(f"   notes         : {result.notes}")

        print("\n" + "=" * 55)

        # PII summary block
        pii_cols = [r for r in self.results if r.pii_detected]
        if pii_cols:
            print(f"\n🔴 GDPR ALERT — {len(pii_cols)} PII column(s) detected:")
            for r in pii_cols:
                label = PII_CATEGORY_LABELS.get(r.pii_category, r.pii_category)
                print(f"   → {r.column_name:<20} [{r.pii_category}]  {label}")
        else:
            print("\n🟢 No PII columns detected.")

        non_gen = [r for r in self.results if not r.generatable]
        if non_gen:
            print(f"\n⚠️  {len(non_gen)} column(s) cannot be generated in v1:")
            for r in non_gen:
                print(f"   → {r.column_name} ({r.semantic_type})")
        else:
            print("\n✅ All columns are generatable.")