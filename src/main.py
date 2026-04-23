import os
import pandas as pd
from dotenv import load_dotenv

from analyzer.dataset_analyzer import DatasetAnalyzer
from preprocessor.preprocessor import Preprocessor
from diffusion                  import TabDDPM

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

# ── Phase 3: Diffusion model ──────────────────────────────────────────────
model = TabDDPM.from_preprocessor(output)
model.summary()

model.auto_tune(output.tensor_dataset)

synthetic_tensor = model.generate(n_samples=len(df))
synthetic_df     = prep.inverse_transform(synthetic_tensor)

# ── Quick sanity check ────────────────────────────────────────────────────
print(synthetic_df.head(20))
print(f"\nOriginal shape  : {df.shape}")
print(f"Synthetic shape : {synthetic_df.shape}")
print(f"Columns match   : {list(df.columns) == list(synthetic_df.columns)}")

# ── Save model ────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
model.save("models/tabddpm.pt")