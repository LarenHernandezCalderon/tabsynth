<div align="center">
  <img src="docs/assets/banner.svg" alt="tabsynth" width="100%"/>
</div>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square" alt="Python"/></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="#"><img src="https://img.shields.io/badge/OpenAI-GPT--4-412991?style=flat-square&logo=openai&logoColor=white" alt="OpenAI"/></a>
  <a href="#"><img src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square" alt="License"/></a>
  <a href="#"><img src="https://img.shields.io/badge/GDPR-compliant-success?style=flat-square" alt="GDPR"/></a>
  <a href="#"><img src="https://img.shields.io/badge/status-prototype-blue?style=flat-square" alt="Status"/></a>
</p>

<p align="center">
  <b>Auto-adaptive synthetic tabular data generation via diffusion models.</b><br>
  Point it at any CSV — it figures out the rest.
</p>

<p align="center">
  <a href="#why">Why</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#how-it-works">How It Works</a> ·
  <a href="#results">Results</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#limitations">Limitations</a>
</p>

---

## Why

Real data is the most valuable and the most restricted asset in any organisation. GDPR fines reach €20M, healthcare data can't leave the building, financial records can't be shared with vendors. The standard answer — "just generate synthetic data" — is harder than it sounds: most tools require manual schema configuration, ignore statistical structure, or break the correlations that make data useful for ML.

**tabsynth** is built around one idea: the system should understand the data before generating anything. A five-layer analysis pipeline inspects every column autonomously — types, distributions, PII categories, inter-column correlations — and uses that understanding to configure a diffusion model that preserves the full joint distribution, not just marginals.

Three design principles guided every decision:

- **Auto-adaptive.** No config files, no type hints, no manual schema. Drop any CSV and the pipeline handles the rest — from detecting that `SibSp` is zero-inflated to inferring that `Name` is a GDPR direct identifier.
- **Correlation-aware training.** Strong column pairs detected in Phase 1 (Pearson, Spearman, Cramér's V, η²) are injected as a `CorrelationLoss` term during diffusion training. The model is explicitly penalised for breaking relationships like `carat ↔ price` or `Pclass ↔ Fare`.
- **GDPR-compliant by design.** Direct identifiers are regenerated with Faker — real values are never sampled, never stored, never sent anywhere. Quasi-identifiers go through the diffusion model. Privacy is quantified via Distance Ratio (NND-based re-identification risk).

---

## Demo

<div align="center">
  <img src="assets/pipeline.gif" alt="tabsynth pipeline" width="820"/>
  <br><sub>Full pipeline on Titanic (891 rows × 12 columns) — analysis, training, generation, evaluation.</sub>
</div>

---

## Quick Start

```bash
git clone https://github.com/LarenHernandezCalderon/synthetic-data-engine.git
cd synthetic-data-engine
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
```

Edit `src/main.py` — change one line:

```python
DATASET = "titanic.csv"   # or "diamonds.csv", "plane.csv", your own CSV
```

Then run:

```bash
cd src
python main.py
```

Output: `output/synthetic.csv` + `output/evaluation_report.txt` + `models/tabddpm_<dataset>_seed<N>.pt`

---

## How It Works

The pipeline runs in five sequential phases. Each phase has a single responsibility and exposes a typed contract (`DatasetProfile`, `PreprocessorOutput`) to the next.

```
┌─────────────┐    ┌──────────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
│  Analyzer   │───▶│ Preprocessor │───▶│ Diffusion │───▶│ Generator │───▶│ Evaluator │
│  (Phase 1)  │    │  (Phase 2)   │    │ (Phase 3) │    │ (Phase 4) │    │ (Phase 5) │
└─────────────┘    └──────────────┘    └───────────┘    └───────────┘    └───────────┘
  5 detection        8-step modular      TabDDPM-        type enforce,    6-dimension
  layers:            tensor pipeline     inspired        Faker, regex,    quality score
  types, PII,        [0,1] normalized    + CorrelLoss    null preserve    + PDF report
  regex, distrib,    + zero-inflated     + auto-tune
  correlations       splits
```

### Phase 1 — Analyzer (5 layers)

The brain of the system. Quality of everything downstream depends directly on what is found here.

**Layer 1 — MetadataExtractor** inspects each column structurally without storing raw values. Text columns produce a `StringMetadata` object capturing signals like average length, capitalisation ratio, numeric character ratio, and special character presence. These signals — never the actual data — are what gets sent to the LLM in Layer 2.

**Layer 2 — SemanticAnalyzer (GPT-4)** classifies columns the MetadataExtractor could not resolve (`unknown_string`). It receives only the aggregated signals, infers semantic type (`full_name`, `email`, `code`, `free_text`…), assigns a GDPR category (direct identifier / quasi-identifier / Article 9 sensitive / non-PII), and infers locale for Faker generation. Real data never leaves the system.

**Layer 2b — RegexMiner** deploys on `code` columns (ticket numbers, IBANs, product codes). It groups strings by structural patterns and synthesises a regular expression covering ≥80% of non-null values. Synthetic codes produced later will pass the same format validators as real ones.

**Layer 3 — DistributionProfiler** runs Shapiro-Wilk normality tests (capped at 5,000 samples), classifies skewness (mild / moderate / severe), detects zero-inflation, and recommends the optimal pre-model transformation: `log1p` for exponential distributions, `QuantileTransformer` for multimodal or unknown shapes, `none` for already-normal data.

**Layer 4 — CorrelationAnalyzer** computes all pairwise dependencies using the method appropriate for each column type pair: Pearson / Spearman for numeric–numeric, Cramér's V for categorical–categorical, η² for numeric–categorical. Pairs above the `strong` threshold (≥0.40) are stored in `DatasetProfile.critical_pairs` and injected into the diffusion training loss.

### Phase 2 — Preprocessor (8 steps)

A modular pipeline where each step is independent and invertible:

| Step | What it does |
|------|-------------|
| 0 — Null Normaliser | Detects sentinel nulls (`-1`, `999`, `N/A`) and replaces with `NaN` |
| 1 — Column Filter | Routes columns: tensor / external generation (Faker, regex) / skip (free_text) / sequential ID |
| 2 — Missing Handler | Creates binary `_missing` indicator flags; imputes with median/mode |
| 3 — Zero-Inflated Split | Splits zero-heavy columns into `_flag` (binary) + `_value` (non-zero distribution) |
| 4 — Numeric Transform | Applies log1p / QuantileTransformer as recommended by Phase 1 |
| 5 — Categorical Encoder | Ordinal and one-hot encoding; preserves category order |
| 6 — Normaliser | MinMax scaling to [0, 1] for all tensor features |
| 7 — Tensor Builder | Assembles typed `TensorDataset` with feature metadata (binary / continuous / ordinal / categorical) |

The full inverse transformation is implemented in `inverter.py` to map generated tensors back to the original schema.

### Phase 3 — Diffusion Model

A custom Gaussian diffusion architecture for mixed tabular data, inspired by [TabDDPM (Kotelnikov et al., 2022)](https://arxiv.org/abs/2209.15421).

**Architecture:** Residual MLP with sinusoidal timestep embeddings. Each ResidualBlock applies LayerNorm → Linear → SiLU → Dropout → Linear with a skip connection. Timestep is embedded via sinusoidal encoding and projected into each block, conditioning the denoising on the noise level.

**Loss:** Mixed `MixedLoss` combining MSE (continuous features), BCE (binary), and CrossEntropy (categorical), weighted by feature type counts. A `CorrelationLoss` term is added for critical pairs detected in Phase 1:

```
total_loss = mixed_loss + corr_weight × correlation_loss
```

where `correlation_loss` penalises deviation of the predicted batch's correlation matrix from the target values at each training step.

**Noise schedule:** Cosine schedule (`noise_scheduler.py`) for smoother variance control than linear schedules.

**Auto-tuning:** `HyperTuner` runs a 5-candidate grid search (50 probe epochs each) across network depth, width, learning rate, and batch size. The winning configuration trains to convergence with patience-based early stopping (default patience=50, max 2000 epochs).

| Candidate | hidden_dims | lr | batch |
|-----------|-------------|-----|-------|
| A — small/fast | [128, 128] | 1e-3 | 256 |
| B — standard | [256, 256] | 1e-3 | 64 |
| C — deeper | [256, 256, 128] | 1e-3 | 64 |
| D — wide + low lr | [512, 256] | 5e-4 | 128 |
| E — large network | [512, 512, 256] | 1e-3 | 64 |

### Phase 4 — Generator

Takes pure Gaussian noise → 1000 reverse diffusion steps → synthetic tensor → inverts all preprocessor transforms → reconstructs original schema:

- **Ordinal columns**: discrete snap to nearest valid integer value
- **PII columns** (`direct_identifier`): Faker with locale inferred in Phase 1 — real values never used
- **Code columns**: regex template generation from RegexMiner pattern
- **Null rates**: preserved per-column from the real dataset
- **Post-hoc calibration**: QuantileTransformer calibration on continuous columns to correct residual distributional drift

### Phase 5 — Evaluator (6 dimensions)

Two evaluators with different roles:

**Evaluator A — Statistical Realism** (counts toward overall score):

| Dimension | Method | Weight |
|-----------|--------|--------|
| Fidelity | KS test + Wasserstein distance + TVD | 30% |
| Relationships | Pearson/Spearman Frobenius norm + pair delta | 25% |
| Utility (TSTR) | Train on Synthetic, Test on Real — 3 ML models | 30% |
| Privacy | NND Distance Ratio (re-identification risk) | 10% |
| Diversity | Variance ratio + PCA spread + category coverage | 5% |

**Evaluator B — Detectability** (diagnostic only, not in overall score): adversarial RandomForest trained to distinguish real vs synthetic. Columns generated externally (Faker, regex, sequential IDs) are excluded — they would trivially inflate detectability and tell us nothing about model quality.

> PII safety: `direct_identifier` columns are dropped from **all** evaluators before any computation. Real names, emails, or IDs never enter any classifier or distance calculation.

---

## Results

Validated on three public datasets covering different challenges: small dataset with PII, mixed types and exponential distributions (Titanic); mid-size aviation dataset with strong physical correlations between continuous variables (Plane Crashes); large dataset with well-defined ordinal categories and strong cardinality (Diamonds, ~54k rows).

| Dataset | Rows | Cols | Fidelity | Relationships | Utility (TSTR) | Privacy | Diversity | **Overall** |
|---------|------|------|----------|---------------|----------------|---------|-----------|-------------|
| **Titanic** | 891 | 12 | 0.78 | 0.84 | 0.81 | 1.00 | 0.96 | **0.84** 🟡 |
| **Plane Crashes** | 2,709 | 8 | 0.80 | 0.98 | 1.00 | 1.00 | 0.98 | **0.93** 🟢 |
| **Diamonds** | ~54k | 10 | — | — | — | — | — | **pending** |

> 🟢 ≥ 0.85 production-ready · 🟡 ≥ 0.70 suitable for most use cases · 🟠 ≥ 0.55 use with caution

All results use `SEED = 12` — fully reproducible. Run `python main.py` with the dataset of your choice.

**What the numbers mean in practice:**

**Utility (TSTR) = 1.00 on Plane Crashes** — three classifiers (LogisticRegression, RandomForest, GradientBoosting) trained on synthetic data and tested on real data achieve **identical accuracy** to models trained on real data. Zero measurable gap.

**Utility (TSTR) = 0.81 on Titanic** — mean accuracy gap of 1.9% across three models on a notoriously hard dataset: 891 rows, 20% missing Age values, PII columns, zero-inflated distributions, and exponential Fare. LogisticRegression trained on synthetic data achieves identical accuracy (80.45%) to training on real data.

**Relationships = 0.98 on Plane Crashes** — **10/10 critical pairs** preserved within tolerance, including the key physical correlation `aboard ↔ fatalities` (Spearman 0.763 real → 0.734 synthetic, delta=0.029). The `CorrelationLoss` term during training is directly responsible for this.

**Privacy = 1.00 on both datasets** — Distance Ratio of 1.71 (Titanic) and 1.20 (Plane) confirms synthetic records are further from real records than real records are from each other. No re-identification risk detected.

**Dataset size effect is visible and expected** — Titanic (891 rows) scores 0.84 GOOD while Plane Crashes (2,709 rows) scores 0.93 EXCELLENT. More data gives the diffusion model more signal to learn from. Diamonds (~54k rows) results pending.

---

## Architecture

```
synthetic-data-engine/
├── src/
│   ├── main.py                        # entry point
│   │
│   ├── analyzer/                      # Phase 1 — 5 detection layers
│   │   ├── metadata_extractor.py      # structural inspection, no raw values stored
│   │   ├── semantic_analyzer.py       # GPT-4 PII classification via signal metadata
│   │   ├── regex_miner.py             # local pattern extraction for code columns
│   │   ├── distribution_profiler.py   # Shapiro-Wilk, skewness, transform recommendation
│   │   ├── correlation_analyzer.py    # Pearson / Spearman / Cramér's V / η²
│   │   └── dataset_analyzer.py        # orchestrator → DatasetProfile
│   │
│   ├── preprocessor/                  # Phase 2 — modular 8-step pipeline
│   │   ├── contracts.py               # shared dataclasses + typed constants
│   │   ├── preprocessor.py            # orchestrator
│   │   ├── steps/
│   │   │   ├── step0_null_normalizer.py
│   │   │   ├── step1_column_filter.py
│   │   │   ├── step2_missing_handler.py
│   │   │   ├── step3_zero_inflated.py
│   │   │   ├── step4_numeric_transform.py
│   │   │   ├── step5_categorical_encoder.py
│   │   │   ├── step6_normalizer.py
│   │   │   └── step7_tensor_builder.py
│   │   └── inverter.py                # full inverse transformation
│   │
│   ├── diffusion/                     # Phase 3 — custom diffusion model
│   │   ├── tabddpm.py                 # main model + training loop
│   │   ├── mlp.py                     # ResidualBlock + SinusoidalEmbedding
│   │   ├── noise_scheduler.py         # cosine schedule
│   │   ├── loss.py                    # MixedLoss + CorrelationLoss
│   │   └── tuner.py                   # HyperTuner + EarlyStopping
│   │
│   ├── generator/                     # Phase 4 — tensor → original schema
│   │   ├── generator.py
│   │   ├── type_enforcer.py           # discrete snap for ordinals/binary
│   │   ├── column_builder.py          # Faker + regex generation
│   │   ├── null_validator.py          # null rate correction
│   │   └── calibrator.py             # post-hoc quantile calibration
│   │
│   └── evaluator/                     # Phase 5 — 6-dimension scoring
│       ├── evaluator.py               # Evaluator A / B orchestration
│       ├── fidelity.py                # KS + Wasserstein + TVD
│       ├── relationships.py           # Pearson/Spearman + Frobenius
│       ├── utility.py                 # TSTR — 3 ML models
│       ├── detectability.py           # adversarial classifier
│       ├── privacy.py                 # NND + Distance Ratio
│       ├── diversity.py               # Variance + PCA + Coverage
│       └── report.py                  # evaluation report writer
│
├── data/                              # input CSVs (not committed)
├── output/                            # synthetic.csv + evaluation_report.txt
├── models/                            # saved .pt checkpoints
├── assets/                            # banner, diagrams
├── docs/                              # technical report PDF
├── requirements.txt
└── .env.example
```

---

## Tech Stack

| Layer | Library |
|-------|---------|
| Deep learning | PyTorch 2.x |
| Data processing | pandas, numpy, scipy |
| ML evaluation | scikit-learn |
| PII generation | Faker |
| Semantic analysis | OpenAI API (GPT-4) |
| Statistical tests | scipy.stats (KS, Shapiro-Wilk, Wasserstein) |

---

## Limitations

Honest about what doesn't work yet:

- **Heavy-tailed distributions** (`Fare` in Titanic, max 512 with median 14) remain the hardest case. `log1p` helps significantly but the extreme tail is undersampled on small datasets — this is a fundamental limitation of the dataset size, not the model.
- **High-cardinality categoricals** (`hour` in Plane Crashes: 30 classes including malformed values like `c:09:00`) degrade TVD scores. The model generates valid-format values but cannot fully replicate a noisy real-world distribution.
- **CPU training time** on large datasets (~54k rows, Diamonds) takes 30–60 min per run. CUDA reduces this to ~5 min.
- **Free-text columns** (`operator`, `summary`, `route`) are skipped in v1 — they are logged in the profile but not generated. NLP generation is on the roadmap.

The project does not yet support: time-series data, multi-table relational schemas, or streaming input.

---

## Roadmap

- [ ] Diamonds results + full 3-dataset table
- [ ] Hugging Face Spaces interactive demo
- [ ] Quantile fix for low-cardinality ordinals (≤10 unique values → `transform=none`)
- [ ] Optional Microsoft Presidio integration for value-level PII validation
- [ ] NLP generation for free-text columns
- [ ] Time-series support
- [ ] Multi-table relational schemas

---

## Citation

If you use this system in research or production:

```bibtex
@software{hernandez2026tabsynth,
  author  = {Hernández Calderón, Laren},
  title   = {tabsynth: Auto-adaptive synthetic tabular data generation via diffusion models},
  year    = {2026},
  url     = {https://github.com/LarenHernandezCalderon/synthetic-data-engine}
}
```

The diffusion architecture is inspired by — but not a reimplementation of — the original TabDDPM paper:

> Kotelnikov, A., Baranchuk, D., Rubachev, I., & Babenko, A. (2022). *TabDDPM: Modelling Tabular Data with Diffusion Models*. ICML 2023.

Key differences from the original: this implementation uses a residual MLP with sinusoidal timestep embeddings and a mixed Gaussian-only loss (MSE + BCE + CE) rather than the multinomial diffusion process for categorical features described in the paper. The five-layer analysis pipeline, correlation-aware training loss, and six-dimension evaluation framework are original to this project.

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Author

**Laren Hernández Calderón**

[GitHub](https://github.com/LarenHernandezCalderon) · [LinkedIn](https://www.linkedin.com/in/laren-hern%C3%A1ndez-a9037a2a5/)

---

<sub>Built with care. If this helped you, a ⭐ on the repo means a lot.</sub>
