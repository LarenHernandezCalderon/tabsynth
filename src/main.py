import os
import pandas as pd
from dotenv import load_dotenv

from analyzer.dataset_analyzer import DatasetAnalyzer
from preprocessor.preprocessor import Preprocessor
from diffusion                  import TabDDPM
from generator                  import Generator
from evaluator                  import Evaluator   # Phase 5

load_dotenv()

# Dataset selection
df = pd.read_csv("../data/titanic.csv")

# Phase 1
profile = DatasetAnalyzer(df, openai_api_key=os.getenv("OPENAI_API_KEY")).analyze()

# Phase 2
prep   = Preprocessor(df, profile)
output = prep.fit_transform()

# Phase 3
model = TabDDPM.from_preprocessor(output)
model.summary()
model.auto_tune(output.tensor_dataset)
synthetic_tensor = model.generate(n_samples=len(df))

# Phase 4
gen    = Generator(profile, output, df)
result = gen.generate(synthetic_tensor, prep)

print(result.synthetic_df.head(20))
print(f"\nOriginal shape  : {df.shape}")
print(f"Synthetic shape : {result.synthetic_df.shape}")
print(f"Columns match   : {result.columns_match}")
print(f"Dtypes match    : {result.dtypes_match}")

os.makedirs("../output", exist_ok=True)
result.synthetic_df.to_csv("../output/synthetic.csv", index=False)
print("\nSaved → ../output/synthetic.csv")

# Phase 5
evaluator   = Evaluator()
eval_result = evaluator.evaluate(
    real_df    = df,
    synth_df   = result.synthetic_df,
    target_col = "Survived",
)

evaluator.report(eval_result)
evaluator.save_report(eval_result, "../output/evaluation_report.txt")

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/tabddpm.pt")