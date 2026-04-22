import os
import pandas as pd
from dotenv import load_dotenv

from analyzer.dataset_analyzer import DatasetAnalyzer
from preprocessor.preprocessor import Preprocessor

load_dotenv()

# ── Dataset selection — change path to switch datasets ────────────────────
df = pd.read_csv("../data/titanic.csv")
# df = pd.read_csv("../data/adult.csv")
# df = pd.read_csv("../data/diamonds.csv")

# ── Phase 1: Analysis ─────────────────────────────────────────────────────
profile = DatasetAnalyzer(
    df,
    openai_api_key=os.getenv("OPENAI_API_KEY")
).analyze()

# ── Phase 2: Preprocessing ────────────────────────────────────────────────
prep   = Preprocessor(df, profile)
output = prep.fit_transform()