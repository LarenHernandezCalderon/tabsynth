import os
import pandas as pd
from dotenv import load_dotenv
from analyzer.metadata_extractor import MetadataExtractor
from analyzer.semantic_analyzer import SemanticAnalyzer
from analyzer.distribution_profiler import DistributionProfiler

load_dotenv()

df = pd.read_csv("../data/titanic.csv")

# ── Layer 1: local metadata extraction ───────────────────────────────────────
extractor = MetadataExtractor(df)
columns   = extractor.extract()
extractor.summary()

# ── Layer 2: GPT semantic classification (unknown strings only) ───────────────
analyzer = SemanticAnalyzer(api_key=os.getenv("OPENAI_API_KEY"))
results  = analyzer.analyze(columns)
analyzer.summary()

# ── Layer 3: distribution profiling (numerical columns) ───────────────────────
profiler = DistributionProfiler(df, columns)
profiler.profile()
profiler.summary()