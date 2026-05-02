import os
import pandas as pd
from dotenv import load_dotenv

from analyzer.dataset_analyzer import DatasetAnalyzer
from preprocessor.preprocessor import Preprocessor
from diffusion                  import TabDDPM
from generator                  import Generator
from generator.calibrator       import Calibrator
from evaluator                  import Evaluator
import torch
import numpy as np
import random

load_dotenv()

# ── Global configuration ──────────────────────────────────────────────────
DATASET = "titanic.csv"
SEED    = 12

# ── Fix random seeds for reproducibility ─────────────────────────────────
print(f"Seed: {SEED} | Dataset: {DATASET}")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Dataset preparation ───────────────────────────────────────────────────

def prepare_dataset(name: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(f"../data/{name}")

    if name == "titanic.csv":
        target = "Survived"

    elif name == "plane.csv":
        df["all_perished"] = (df["fatalities"] >= df["aboard"]).astype(int)
        drop_cols = ["summary", "route", "location"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        target = "all_perished"

    elif name == "housing.csv":
        target = "median_house_value"

    else:
        # Unknown dataset — fall back to last column as target
        target = df.columns[-1]
        print(f"  Warning — unknown dataset, using '{target}' as target_col, for evaluation phase.")

    return df, target


df, target_col = prepare_dataset(DATASET)
print(f"\nDataset : {DATASET}  ({df.shape[0]} rows x {df.shape[1]} columns)")
print(f"Target  : {target_col}\n")

# ── Phase 1 — Dataset analysis ────────────────────────────────────────────
profile = DatasetAnalyzer(df, openai_api_key=os.getenv("OPENAI_API_KEY")).analyze()

# ── Phase 2 — Preprocessing ───────────────────────────────────────────────
prep   = Preprocessor(df, profile)
output = prep.fit_transform()

# ── Phase 3 — Diffusion model training & generation ───────────────────────
model = TabDDPM.from_preprocessor(output, profile=profile)
model.summary()
model.auto_tune(output.tensor_dataset)
synthetic_tensor = model.generate(n_samples=len(df))

# ── Phase 4 — Post-processing & inverse transform ─────────────────────────
gen    = Generator(profile, output, df)
result = gen.generate(synthetic_tensor, prep)

# Calibrate zero-inflated columns to match original null/zero distribution
zero_inflated_originals = set(output.zero_inflated_map.values())
calibrator = Calibrator(real_df=df, zero_inflated=zero_inflated_originals)
result.synthetic_df = calibrator.transform(result.synthetic_df)

print(result.synthetic_df.head(20))
print(f"\nOriginal shape  : {df.shape}")
print(f"Synthetic shape : {result.synthetic_df.shape}")
print(f"Columns match   : {result.columns_match}")
print(f"Dtypes match    : {result.dtypes_match}")

# Save synthetic dataset to disk
os.makedirs("../output", exist_ok=True)
result.synthetic_df.to_csv("../output/synthetic.csv", index=False)
print(f"\nSaved -> ../output/synthetic.csv")

# ── Phase 5 — Evaluation ──────────────────────────────────────────────────
evaluator   = Evaluator()
eval_result = evaluator.evaluate(
    real_df    = df,
    synth_df   = result.synthetic_df,
    target_col = target_col,
    profile    = profile,
)

evaluator.report(eval_result)
evaluator.save_report(eval_result, "../output/evaluation_report.txt")

# ── Save model checkpoint ─────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
model.save(f"models/tabddpm_{DATASET.replace('.csv', '')}_seed{SEED}.pt")
print(f"Model saved to: models/tabddpm_{DATASET.replace('.csv', '')}_seed{SEED}.pt")