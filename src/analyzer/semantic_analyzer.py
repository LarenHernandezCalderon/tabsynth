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
    - notes               : GPT explanation of its reasoning

Supported semantic types and generation strategies:
    first_name   → faker.first_name()
    last_name    → faker.last_name()
    full_name    → faker.name()
    email        → faker.email()
    phone        → faker.phone_number()
    address      → faker.address()
    country      → faker.country()
    city         → faker.city()
    zipcode      → faker.zipcode()
    free_text    → not generatable in v1 (flagged)
    code         → not generatable in v1 (flagged)
    unknown      → skipped with warning

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
}

NON_GENERATABLE_TYPES = {
    "free_text",
    "code",
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
    "free_text"  : None,
    "code"       : None,
    "unknown"    : None,
}


# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass
class SemanticResult:
    """Result of GPT semantic classification for a single column."""
    column_name:         str
    semantic_type:       str
    confidence:          float
    generatable:         bool
    generation_strategy: Optional[str]
    notes:               str


# ── GPT prompt builder ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a data schema analyst specialized in identifying the semantic meaning
of dataset columns based ONLY on structural metadata — never on actual values.

Your job is to classify what kind of real-world information a column contains,
given only statistical and structural signals like average length, word count,
character patterns, and uniqueness ratio.

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

    valid_types = list(GENERATION_STRATEGIES.keys())

    return f"""
Classify the semantic type of this dataset column using ONLY the metadata below.
Do NOT assume any actual data values exist — only use the structural signals provided.

Column metadata:
{json.dumps(payload, indent=2)}

Valid semantic types: {valid_types}

Respond ONLY with this exact JSON structure:
{{
  "semantic_type": "<one of the valid types above>",
  "confidence": <float between 0.0 and 1.0>,
  "generatable": <true or false>,
  "generation_strategy": "<faker method or null>",
  "notes": "<brief reasoning based only on the metadata signals>"
}}
""".strip()


# ── Main analyzer ─────────────────────────────────────────────────────────────

class SemanticAnalyzer:
    """
    GPT-powered semantic classifier for string columns.

    Receives only privacy-safe metadata payloads — no real data values.
    Makes one API call per unknown_string column.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model  = model
        self.results: list[SemanticResult] = []

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
            print(f"→ {result.semantic_type} (confidence: {result.confidence:.0%})")

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
                max_tokens      = 300,
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
                notes               = f"Classification failed: {str(e)}",
            )

    def _build_result(self, column_name: str, parsed: dict) -> SemanticResult:
        semantic_type = parsed.get("semantic_type", "unknown")

        if semantic_type not in GENERATION_STRATEGIES:
            semantic_type = "unknown"

        generatable = semantic_type in GENERATABLE_TYPES
        strategy    = GENERATION_STRATEGIES.get(semantic_type)

        return SemanticResult(
            column_name         = column_name,
            semantic_type       = semantic_type,
            confidence          = float(parsed.get("confidence", 0.0)),
            generatable         = generatable,
            generation_strategy = strategy,
            notes               = parsed.get("notes", ""),
        )

    def summary(self) -> None:
        if not self.results:
            print("No semantic results to display.")
            return

        print("\n🧠 Semantic Analysis Report (Layer 2)")
        print("=" * 55)

        for result in self.results:
            icon = "✅" if result.generatable else "⚠️ "
            print(f"\n{icon} {result.column_name}")
            print(f"   semantic_type : {result.semantic_type}")
            print(f"   confidence    : {result.confidence:.0%}")
            print(f"   generatable   : {result.generatable}")
            print(f"   strategy      : {result.generation_strategy or 'N/A — will be skipped'}")
            print(f"   notes         : {result.notes}")

        print("\n" + "=" * 55)

        non_gen = [r for r in self.results if not r.generatable]
        if non_gen:
            print(f"\n⚠️  {len(non_gen)} column(s) cannot be generated in v1:")
            for r in non_gen:
                print(f"   → {r.column_name} ({r.semantic_type})")