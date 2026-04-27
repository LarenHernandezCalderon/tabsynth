[README.md](https://github.com/user-attachments/files/27113552/README.md)
<div align="center">
  <img src="docs/assets/banner.svg" alt="tabsynth" width="100%"/>
</div>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square" alt="Python"/></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="#"><img src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square" alt="License"/></a>
  <a href="#"><img src="https://img.shields.io/badge/GDPR-compliant-success?style=flat-square" alt="GDPR"/></a>
  <a href="#"><img src="https://img.shields.io/badge/status-production--ready-brightgreen?style=flat-square" alt="Status"/></a>
</p>

<p align="center">
  <b>An auto-adaptive pipeline for generating high-quality synthetic tabular data.</b><br>
  Point it at any CSV — it figures out the rest.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#how-it-works">How It Works</a> ·
  <a href="#results">Results</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="#citation">Citation</a>
</p>

---

## Why tabsynth

Real data is the most valuable and the most restricted asset in any organization. GDPR fines reach €20M, healthcare data can't leave the building, financial records can't be shared with vendors. The standard answer — "just generate synthetic data" — is harder than it sounds: most tools require manual configuration, ignore privacy, or destroy the statistical relationships that make data useful.

tabsynth is built around three principles:

- **Auto-adaptive.** No configuration files. No type hints. The pipeline analyzes the dataset, infers column types, detects PII, finds correlations, and configures the model itself.
- **GDPR-compliant by design.** Direct identifiers (names, emails) are regenerated with Faker — never sampled from real data. Quasi-identifiers go through the diffusion model. The Distance Ratio metric quantifies re-identification risk.
- **Rigorous evaluation.** Six independent dimensions — fidelity, relationships, utility (TSTR), privacy, diversity, detectability — combined into a single reproducible score.

## Demo

<div align="center">
  <img src="docs/assets/pipeline.gif" alt="tabsynth pipeline running on Titanic" width="800"/>
  <br><sub>Full pipeline running on the Titanic dataset (891 rows × 12 columns).</sub>
</div>

## Quick Start

```bash
git clone https://github.com/LarenHernandezCalderon/tabsynth.git
cd tabsynth
pip install -r requirements.txt
```

```python
from tabsynth import SyntheticDataEngine

engine = SyntheticDataEngine.from_csv("data/titanic.csv")
synthetic_df = engine.generate(n_rows=891)
report = engine.evaluate(synthetic_df)

print(f"Overall quality: {report.overall_score:.2f}")
synthetic_df.to_csv("output/synthetic.csv", index=False)
```

That's it. The engine handles type detection, PII handling, transformations, model selection, and evaluation automatically.

## How It Works

The pipeline runs in five sequential phases. Each phase has a single responsibility and a clear contract with the next.

```
┌─────────────┐    ┌──────────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
│  Analyzer   │ →  │ Preprocessor │ →  │ Diffusion │ →  │ Generator │ →  │ Evaluator │
│  (Phase 1)  │    │  (Phase 2)   │    │ (Phase 3) │    │ (Phase 4) │    │ (Phase 5) │
└─────────────┘    └──────────────┘    └───────────┘    └───────────┘    └───────────┘
   types,             tensor             diffusion         type            6-dim
   PII, regex,        [0,1]              diffusion         enforce,        quality
   distributions,     normalized                           Faker,          score
   correlations                                            regex
```

| Phase | Role |
|-------|------|
| **1. Analyzer** | Detects column types, distributions, PII categories (GDPR Art. 4 / Art. 9), correlations (Pearson, Spearman, Cramér's V, η²), and learns regex patterns from the data |
| **2. Preprocessor** | Modular 8-step pipeline: sentinel handling, redundancy removal, missing indicators, zero-inflated splits, transformations, encoding, normalization |
| **3. Diffusion Model** | Custom Gaussian diffusion architecture for mixed tabular data, inspired by TabDDPM (Kotelnikov et al., 2022). Sinusoidal timestep embedding, residual MLP, mixed loss (MSE + BCE + CE), cosine noise schedule, and 3-level auto-tuning |
| **4. Generator** | Type enforcement with discrete snap, gender-aware Faker, regex template generation, null-rate preservation |
| **5. Evaluator** | KS Test, Wasserstein, TVD, Frobenius norm, TSTR with 3 ML models, NND-based privacy, PCA spread |

Full architectural details and design decisions are documented in [`docs/tabsynth_technical_report.pdf`](docs/tabsynth_technical_report.pdf).

## Results

Validated on three public datasets that stress different parts of the pipeline — small dataset with PII and exponential distributions (Titanic), large dataset with strong physical correlations (Diamonds), and mid-size dataset with discrete ordinal scales (Social Media).

| Dataset | Rows | Cols | Fidelity | Relationships | Utility (TSTR) | Privacy | Diversity | **Overall** |
|---------|------|------|----------|---------------|----------------|---------|-----------|-------------|
| **Titanic** | 891 | 12 | 0.68 | 0.93 | 0.93 | 1.00 | 0.86 | **0.86** |
| **Diamonds** | 50,000 | 10 | 0.65 | 0.99 | 1.00 | 1.00 | 0.97 | **0.85** |
| **Social Media** | 1,200 | 13 | 0.51 | 0.94 | 1.00 | 1.00 | 0.81 | **0.83** |

Score interpretation: ≥0.85 production-ready · ≥0.70 suitable for most use cases · ≥0.55 use with caution

The numbers that matter for real-world use are **Utility, Relationships, and Privacy** — they describe whether the synthetic data behaves like the real data and stays GDPR-safe:

- **Utility (TSTR) on Titanic: 0.93** — a model trained on synthetic data and tested on real data has only a **0.7% accuracy gap** averaged across LogisticRegression, RandomForest, and GradientBoosting. Functionally indistinguishable from real data for downstream ML.
- **Relationships ≥ 0.93 on all three datasets** — correlation structure is preserved. On Diamonds, **21/21 critical pairs** stay within tolerance, including the strong physical correlations between `x`, `y`, `z`, `carat` and `price`.
- **Privacy = 1.00 on all three datasets** — Distance Ratio ≥ 1.09 confirms synthetic records are at least as far from real records as real records are from each other (GDPR re-identification standard).

Fidelity is reported for completeness but is the least informative metric of the six: it answers *"can a statistical test detect any difference between distributions?"*, which is a question that grows artificially strict as datasets get larger and is largely orthogonal to whether the data is useful or private. Detailed methodology and per-column breakdowns are covered in the [technical report](docs/tabsynth_technical_report.pdf).

## Architecture

```
tabsynth/
├── analyzer/                  # Phase 1 — 5 detection layers
│   ├── metadata_extractor.py
│   ├── semantic_analyzer.py   # GPT-4 PII classification
│   ├── regex_miner.py
│   ├── distribution_profiler.py
│   ├── correlation_analyzer.py
│   └── dataset_analyzer.py
│
├── preprocessor/              # Phase 2 — modular 8-step pipeline
│   ├── contracts.py           # shared dataclasses + constants
│   ├── preprocessor.py        # orchestrator
│   ├── steps/                 # one file per step
│   │   ├── step0_null_normalizer.py
│   │   ├── step1_column_filter.py
│   │   ├── step2_missing_handler.py
│   │   ├── step3_zero_inflated.py
│   │   ├── step4_numeric_transform.py
│   │   ├── step5_categorical_encoder.py
│   │   ├── step6_normalizer.py
│   │   └── step7_tensor_builder.py
│   └── inverse/inverter.py    # 8-step inverse transformation
│
├── diffusion/                 # Phase 3 — custom diffusion model
│   ├── tabddpm.py             # main orchestrator (TabDDPM-inspired)
│   ├── mlp.py                 # ResidualBlock + SinusoidalEmbedding
│   ├── noise_scheduler.py     # cosine schedule
│   ├── loss.py                # MixedLoss
│   └── tuner.py               # HyperTuner + EarlyStopping
│
├── generator/                 # Phase 4 — tensor → CSV
│   ├── generator.py
│   ├── type_enforcer.py       # discrete snap for ordinals
│   ├── column_builder.py      # gender-aware Faker
│   └── null_validator.py
│
└── evaluator/                 # Phase 5 — 6-dimension scoring
    ├── evaluator.py           # Evaluator A vs B separation
    ├── fidelity.py            # KS + Wasserstein + TVD
    ├── relationships.py       # Pearson/Spearman + Frobenius
    ├── utility.py             # TSTR with 3 models
    ├── detectability.py       # adversarial classifier
    ├── privacy.py             # NND + Distance Ratio
    └── diversity.py           # Variance + PCA + Coverage
```

## Tech Stack

`Python 3.11+` · `PyTorch 2.x` · `pandas` · `scikit-learn` · `scipy` · `Faker` · `OpenAI API` (semantic analysis)

## Documentation

The full technical report — covering architecture, design rationale, evaluation methodology, and empirical results — is available at [`docs/tabsynth_technical_report.pdf`](docs/tabsynth_technical_report.pdf).

## Limitations

Honest about what doesn't work yet:

- **Heavy-tailed distributions** (e.g. `Fare` in Titanic, max 512 with median 14) remain the hardest case. log1p helps but the extreme tail is undersampled with small datasets.
- **Low-cardinality ordinals** (10-value scales like `stress_level` 1–10) currently receive QuantileTransformer, which over-disperses on inversion. Fix identified — applying `transform=none` for ordinals with ≤10 unique values is on the roadmap.
- **CPU training** on large datasets (Adult Income, ~48k rows) takes >1h. CUDA reduces it to ~10min.

The project does not (yet) support: time-series data, multi-table relational schemas, or text generation beyond regex-based codes.

## Roadmap

- [ ] Quantile fix for low-cardinality ordinals (+0.10 expected on Social Media)
- [ ] Hugging Face Spaces interactive demo
- [ ] Optional Microsoft Presidio integration for value-level PII validation
- [ ] Time-series support (longitudinal data)
- [ ] Multi-table relational schemas

## Citation

If you use tabsynth in your research or production work:

```bibtex
@software{hernandez2026tabsynth,
  author  = {Hernández Calderón, Laren},
  title   = {tabsynth: Auto-adaptive synthetic tabular data generation},
  year    = {2026},
  url     = {https://github.com/LarenHernandezCalderon/tabsynth}
}
```

The diffusion architecture is inspired by — but is not a reimplementation of — the original TabDDPM paper:

> Kotelnikov, A., Baranchuk, D., Rubachev, I., & Babenko, A. (2022). *TabDDPM: Modelling Tabular Data with Diffusion Models*. ICML 2023.

Key differences from the original work: this implementation uses a residual MLP with sinusoidal timestep embedding and a mixed Gaussian-only loss (MSE + BCE + CE), rather than the multinomial diffusion process for categorical features described in the paper. The full analysis-preprocessing-evaluation pipeline around the model is original to this project.

## License

MIT — see [`LICENSE`](LICENSE).

## Author

**Laren Hernández Calderón**
[GitHub](https://github.com/LarenHernandezCalderon) · [LinkedIn](https://www.linkedin.com/in/laren-hern%C3%A1ndez-a9037a2a5/)

---

<sub>Built with care. If this project helped you, a ⭐ on the repo means a lot.</sub>
