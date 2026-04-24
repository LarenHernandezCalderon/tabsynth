import os
import pandas as pd
from dotenv import load_dotenv

from analyzer.dataset_analyzer import DatasetAnalyzer
from preprocessor.preprocessor import Preprocessor
from diffusion                  import TabDDPM
from generator                  import Generator   

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

# ── Phase 4: Generator ────────────────────────────────────────────────────
gen    = Generator(profile, output, df)
result = gen.generate(synthetic_tensor, prep)

# ── Output ────────────────────────────────────────────────────────────────
print(result.synthetic_df.head(80))
print(f"\nOriginal shape  : {df.shape}")
print(f"Synthetic shape : {result.synthetic_df.shape}")
print(f"Columns match   : {result.columns_match}")
print(f"Dtypes match    : {result.dtypes_match}")

# Save CSV
os.makedirs("../output", exist_ok=True)
result.synthetic_df.to_csv("../output/synthetic.csv", index=False)
print("\nSaved → ../output/synthetic.csv")

# ── Save model ────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
model.save("models/tabddpm.pt")