<p align="right">
  🌐 <a href="README.md">English</a> · <strong>Español</strong>
</p>

<div align="center">
  <img src="docs/assets/banner.svg" alt="tabsynth" width="100%"/>
</div>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square" alt="Python"/></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="#"><img src="https://img.shields.io/badge/OpenAI-GPT--4-412991?style=flat-square&logo=openai&logoColor=white" alt="OpenAI"/></a>
  <a href="#"><img src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square" alt="Licencia"/></a>
  <a href="#"><img src="https://img.shields.io/badge/GDPR-compliant-success?style=flat-square" alt="GDPR"/></a>
  <a href="#"><img src="https://img.shields.io/badge/estado-prototipo-blue?style=flat-square" alt="Estado"/></a>
</p>

<p align="center">
  <b>Generación auto-adaptativa de datos tabulares sintéticos mediante modelos de difusión.</b><br>
  Apúntalo a cualquier CSV — él se encarga del resto.
</p>

<p align="center">
  <a href="#por-qué">Por qué</a> ·
  <a href="#seguridad">Seguridad</a> ·
  <a href="#inicio-rápido">Inicio Rápido</a> ·
  <a href="#cómo-funciona">Cómo Funciona</a> ·
  <a href="#resultados">Resultados</a> ·
  <a href="#arquitectura">Arquitectura</a> ·
  <a href="#limitaciones">Limitaciones</a>
</p>

---

## Por qué

Los datos reales son el activo más valioso y más restringido de cualquier organización. Las multas del RGPD alcanzan los 20 M€, los datos sanitarios no pueden salir del edificio, los registros financieros no se pueden compartir con proveedores. La respuesta habitual — "simplemente genera datos sintéticos" — es más difícil de lo que parece: la mayoría de las herramientas requieren configuración manual del esquema, ignoran la estructura estadística o rompen las correlaciones que hacen que los datos sean útiles para el ML.

**tabsynth** se construye en torno a una idea: el sistema debe entender los datos antes de generar nada. Un pipeline de análisis de cinco capas inspecciona cada columna de forma autónoma — tipos, distribuciones, categorías de IIP, correlaciones entre columnas — y utiliza ese conocimiento para configurar un modelo de difusión que preserva la distribución conjunta completa, no solo las marginales.

Tres principios de diseño guiaron cada decisión:

- **Auto-adaptativo.** Sin archivos de configuración, sin sugerencias de tipo, sin esquema manual. Carga cualquier CSV y el pipeline se encarga del resto — detectando automáticamente distribuciones con inflación de ceros, identificando identificadores directos del RGPD, infiriendo patrones regex para códigos estructurados y eligiendo la transformación numérica óptima por columna.
- **Entrenamiento consciente de correlaciones.** Los pares de columnas con fuerte dependencia detectados en la Fase 1 (Pearson, Spearman, V de Cramér, η²) se inyectan como término `CorrelationLoss` durante el entrenamiento del modelo de difusión. El modelo es penalizado explícitamente por romper relaciones estadísticamente significativas entre columnas.
- **Cumplimiento del RGPD por diseño.** Los identificadores directos se regeneran con Faker — los valores reales nunca se muestrean, nunca se almacenan, nunca se envían a ningún sitio. Los quasi-identificadores pasan por el modelo de difusión. La privacidad se cuantifica mediante el Distance Ratio (riesgo de re-identificación basado en NND).

---

## 🔒 Seguridad

> **Ningún dato real abandona jamás el entorno seguro. Esto es una garantía arquitectónica, no una política.**

Este es el compromiso de seguridad central del sistema, y determina cada decisión de diseño:

**Lo que siempre permanece local:**
- Los valores brutos de los datos nunca se envían a ningún servicio externo, en ninguna etapa del pipeline.
- El LLM (GPT-4) recibe únicamente metadatos estructurales agregados: longitudes medias, ratios de capitalización, ratios de caracteres numéricos, rangos de cardinalidad. Nunca ve valores reales de las filas.
- El RegexMiner se ejecuta completamente en proceso: analiza los valores reales en memoria para inferir patrones estructurales, y su único output es una cadena regex — una descripción de la estructura, no una copia de los datos.
- Todo el análisis de distribuciones, cálculo de correlaciones y perfilado estadístico ocurre de forma local mediante scipy y pandas.

**Lo que el modelo aprende — y lo que no:**
- El modelo de difusión aprende la distribución estadística conjunta del dataset. No memoriza filas.
- Las columnas `direct_identifier` (nombres, emails, IDs, teléfonos) nunca se pasan al tensor. Se regeneran desde cero usando Faker con el locale inferido — los valores reales se descartan antes de que comience el entrenamiento.
- La privacidad se mide activamente: la métrica Distance Ratio confirma que los registros sintéticos están estadísticamente más lejos de los registros reales que los registros reales entre sí.

**Lo que los evaluadores nunca tocan:**
- Las columnas con IIP se eliminan de todos los evaluadores (fidelidad, relaciones, utilidad, detectabilidad, privacidad) antes de cualquier cómputo. Los nombres o IDs reales nunca entran en ningún clasificador ni cálculo de distancias.

**En resumen:** lo único que abandona el entorno es el dataset sintético — y por construcción, no puede rastrearse hasta ningún individuo real.

---

## Demo

<div align="center">
  <img src="assets/pipeline.gif" alt="tabsynth pipeline" width="820"/>
  <br><sub>Pipeline completo sobre Titanic (891 filas × 12 columnas) — análisis, entrenamiento, generación, evaluación.</sub>
</div>

---

## Inicio Rápido

```bash
git clone https://github.com/LarenHernandezCalderon/synthetic-data-engine.git
cd synthetic-data-engine
pip install -r requirements.txt
cp .env.example .env   # añade tu OPENAI_API_KEY
```

Edita `src/main.py` — cambia una línea:

```python
DATASET = "titanic.csv"   # o "diamonds.csv", "plane.csv", tu propio CSV
```

Luego ejecuta:

```bash
cd src
python main.py
```

Salida: `output/synthetic.csv` + `output/evaluation_report.txt` + `models/tabddpm_<dataset>_seed<N>.pt`

---

## Cómo Funciona

El pipeline se ejecuta en cinco fases secuenciales. Cada fase tiene una única responsabilidad y expone un contrato tipado (`DatasetProfile`, `PreprocessorOutput`) a la siguiente.

```
┌─────────────┐    ┌──────────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
│  Analyzer   │───▶│ Preprocessor │───▶│ Diffusion │───▶│ Generator │───▶│ Evaluator │
│  (Fase 1)   │    │  (Fase 2)    │    │ (Fase 3)  │    │ (Fase 4)  │    │ (Fase 5)  │
└─────────────┘    └──────────────┘    └───────────┘    └───────────┘    └───────────┘
  5 capas de         pipeline tensor     TabDDPM-        tipos, Faker,    puntuación
  detección:         modular de 8        inspirado       regex, nulos     6 dimensiones
  tipos, IIP,        pasos [0,1]         + CorrelLoss    preservados      + informe PDF
  regex, distrib,    + splits            + auto-tune
  correlaciones      zero-inflated
```

### Fase 1 — Analizador (5 capas)

El cerebro del sistema. La calidad de todo lo que sigue depende directamente de lo que se detecta aquí.

**Capa 1 — MetadataExtractor** inspecciona cada columna estructuralmente sin almacenar valores brutos. Las columnas de texto producen un objeto `StringMetadata` que captura señales como longitud media, ratio de capitalización, ratio de caracteres numéricos y presencia de caracteres especiales. Estas señales — nunca los datos reales — son las que se envían al LLM en la Capa 2.

**Capa 2 — SemanticAnalyzer (GPT-4)** clasifica las columnas que el MetadataExtractor no pudo resolver (`unknown_string`). Recibe únicamente las señales agregadas, infiere el tipo semántico (`full_name`, `email`, `code`, `free_text`…), asigna una categoría RGPD (identificador directo / quasi-identificador / sensible Art. 9 / no-IIP), e infiere el locale para la generación con Faker. **Los datos reales nunca abandonan el sistema.**

**Capa 2b — RegexMiner** actúa sobre columnas `code` (números de ticket, códigos de producto, IDs estructurados). Agrupa cadenas por patrones estructurales y sintetiza una expresión regular que cubre ≥80% de los valores no nulos. **Se ejecuta completamente de forma local — no se almacena ni transmite ningún valor, solo el patrón inferido.** Los códigos sintéticos generados después superarán los mismos validadores de formato que los reales.

**Capa 3 — DistributionProfiler** ejecuta tests de normalidad de Shapiro-Wilk (limitados a 5.000 muestras), clasifica la asimetría (leve / moderada / severa), detecta inflación de ceros y recomienda la transformación pre-modelo óptima: `log1p` para distribuciones exponenciales, `QuantileTransformer` para formas multimodales o desconocidas, `none` para datos ya normales.

**Capa 4 — CorrelationAnalyzer** calcula todas las dependencias por pares usando el método apropiado para cada tipo de par de columnas: Pearson / Spearman para numérico–numérico, V de Cramér para categórico–categórico, η² para numérico–categórico. Los pares por encima del umbral `strong` (≥0.40) se almacenan en `DatasetProfile.critical_pairs` y se inyectan en la función de pérdida del entrenamiento de difusión.

### Fase 2 — Preprocesador (8 pasos)

Un pipeline modular donde cada paso es independiente e invertible:

| Paso | Qué hace |
|------|----------|
| 0 — Null Normaliser | Detecta nulos centinela (`-1`, `999`, `N/A`) y los reemplaza con `NaN` |
| 1 — Column Filter | Enruta columnas: tensor / generación externa (Faker, regex) / omitir (free_text) / ID secuencial |
| 2 — Missing Handler | Crea flags binarias `_missing`; imputa con mediana/moda |
| 3 — Zero-Inflated Split | Divide columnas con muchos ceros en `_flag` (binaria) + `_value` (distribución no-cero) |
| 4 — Numeric Transform | Aplica log1p / QuantileTransformer según lo recomendado por la Fase 1 |
| 5 — Categorical Encoder | Codificación ordinal y one-hot; preserva el orden de categorías |
| 6 — Normaliser | Escalado MinMax a [0, 1] para todas las features del tensor |
| 7 — Tensor Builder | Ensambla un `TensorDataset` tipado con metadatos de features (binaria / continua / ordinal / categórica) |

La transformación inversa completa está implementada en `inverter.py` para mapear los tensores generados de vuelta al esquema original.

### Fase 3 — Modelo de Difusión

Una arquitectura de difusión gaussiana personalizada para datos tabulares mixtos, inspirada en [TabDDPM (Kotelnikov et al., 2022)](https://arxiv.org/abs/2209.15421).

**Arquitectura:** MLP residual con embeddings sinusoidales de timestep. Cada ResidualBlock aplica LayerNorm → Linear → SiLU → Dropout → Linear con una conexión de salto. El timestep se embebe mediante codificación sinusoidal y se proyecta en cada bloque, condicionando la denoisificación al nivel de ruido.

**Pérdida:** `MixedLoss` combinada que combina MSE (features continuas), BCE (binarias) y CrossEntropy (categóricas), ponderadas por conteos de tipo de feature. Se añade un término `CorrelationLoss` para los pares críticos detectados en la Fase 1:

```
pérdida_total = mixed_loss + corr_weight × correlation_loss
```

donde `correlation_loss` penaliza la desviación de la matriz de correlación del batch predicho respecto a los valores objetivo en cada paso de entrenamiento.

**Calendario de ruido:** Calendario coseno (`noise_scheduler.py`) para un control de varianza más suave que los calendarios lineales.

**Auto-ajuste:** `HyperTuner` ejecuta una búsqueda en rejilla de 5 candidatos (50 épocas de sondeo cada uno) a lo largo de la profundidad de la red, anchura, tasa de aprendizaje y tamaño de batch. La configuración ganadora entrena hasta convergencia con early stopping basado en paciencia (paciencia por defecto=50, máx. 2000 épocas).

| Candidato | hidden_dims | lr | batch |
|-----------|-------------|-----|-------|
| A — pequeño/rápido | [128, 128] | 1e-3 | 256 |
| B — estándar | [256, 256] | 1e-3 | 64 |
| C — más profundo | [256, 256, 128] | 1e-3 | 64 |
| D — ancho + lr bajo | [512, 256] | 5e-4 | 128 |
| E — red grande | [512, 512, 256] | 1e-3 | 64 |

### Fase 4 — Generador

Toma ruido gaussiano puro → 1000 pasos de difusión inversa → tensor sintético → invierte todas las transformaciones del preprocesador → reconstruye el esquema original:

- **Columnas ordinales**: snap discreto al valor entero válido más cercano
- **Columnas IIP** (`direct_identifier`): Faker con el locale inferido en la Fase 1 — los valores reales nunca se utilizan
- **Columnas de código**: generación por plantilla regex a partir del patrón RegexMiner
- **Tasas de nulos**: preservadas por columna del dataset real
- **Calibración post-hoc**: calibración QuantileTransformer sobre columnas continuas para corregir la deriva distribucional residual

### Fase 5 — Evaluador (6 dimensiones)

Dos evaluadores con roles diferentes:

**Evaluador A — Realismo Estadístico** (cuenta para la puntuación global):

| Dimensión | Método | Peso |
|-----------|--------|------|
| Fidelidad | Test KS + distancia Wasserstein + TVD | 30% |
| Relaciones | Norma de Frobenius Pearson/Spearman + delta de par | 25% |
| Utilidad (TSTR) | Train on Synthetic, Test on Real — 3 modelos ML | 30% |
| Privacidad | NND Distance Ratio (riesgo de re-identificación) | 10% |
| Diversidad | Ratio de varianza + dispersión PCA + cobertura de categorías | 5% |

**Evaluador B — Detectabilidad** (solo diagnóstico, no cuenta en la puntuación global): RandomForest adversarial entrenado para distinguir real vs sintético. Las columnas generadas externamente (Faker, regex, IDs secuenciales) se excluyen — inflarían trivialmente la detectabilidad sin decirnos nada sobre la calidad del modelo.

> Seguridad IIP: las columnas `direct_identifier` se eliminan de **todos** los evaluadores antes de cualquier cómputo. Los nombres, emails o IDs reales nunca entran en ningún clasificador ni cálculo de distancias.

---

## Resultados

Validado en tres datasets públicos que cubren diferentes retos: dataset pequeño con IIP, tipos mixtos y distribuciones exponenciales (Titanic); dataset de aviación de tamaño medio con fuertes correlaciones físicas entre variables continuas (Plane Crashes); dataset geoespacial grande con fuertes correlaciones entre columnas y features exclusivamente continuas (California Housing, ~21k filas).

| Dataset | Filas | Cols | Fidelidad | Relaciones | Utilidad (TSTR) | Privacidad | Diversidad | **Global** |
|---------|-------|------|-----------|------------|-----------------|------------|------------|------------|
| **Titanic** | 891 | 12 | 0.78 | 0.84 | 0.81 | 1.00 | 0.96 | **0.84** 🟡 |
| **Plane Crashes** | 2.709 | 8 | 0.80 | 0.98 | 1.00 | 1.00 | 0.98 | **0.93** 🟢 |
| **California Housing** | 20.640 | 10 | 0.97 | 0.97 | 0.99 | 1.00 | 0.98 | **0.98** 🟢 |

> 🟢 ≥ 0.85 listo para producción · 🟡 ≥ 0.70 válido para la mayoría de casos · 🟠 ≥ 0.55 usar con precaución

Todos los resultados usan `SEED = 12` — totalmente reproducibles. Ejecuta `python main.py` con el dataset de tu elección.

**Lo que significan los números en la práctica:**

**Utilidad (TSTR) = 1.00 en Plane Crashes** — tres clasificadores (LogisticRegression, RandomForest, GradientBoosting) entrenados sobre datos sintéticos y evaluados sobre datos reales alcanzan **exactamente la misma precisión** que los modelos entrenados sobre datos reales. Cero diferencia medible.

**Utilidad (TSTR) = 0.99 en California Housing** — diferencia media de precisión del 0.1% en tres modelos sobre un dataset geoespacial de 20k filas con 8 features continuas fuertemente correlacionadas y distribuciones geográficas bimodales. Equivalencia estadística casi perfecta para propósitos de ML.

**Utilidad (TSTR) = 0.81 en Titanic** — diferencia media de precisión del 1.9% en tres modelos sobre un dataset notoriamente difícil: 891 filas, 20% de valores ausentes en columnas clave, columnas IIP, distribuciones con inflación de ceros y features continuas de cola exponencial. LogisticRegression entrenado sobre datos sintéticos alcanza exactamente la misma precisión (80.45%) que entrenando sobre datos reales.

**Relaciones = 0.98 en Plane Crashes / 0.97 en California Housing** — pares críticos preservados dentro de tolerancia en ambos datasets. El término `CorrelationLoss` durante el entrenamiento es directamente responsable: los pares de columnas fuertemente correlacionados detectados en la Fase 1 se inyectan en el objetivo de entrenamiento, y el modelo es penalizado por romperlos. En California Housing, **35/36 pares** preservados incluyendo la relación clave `median_income ↔ median_house_value` (Pearson 0.688).

**Privacidad = 1.00 en los tres datasets** — Distance Ratio de 1.60 (Housing), 1.71 (Titanic) y 1.20 (Plane) confirma que los registros sintéticos están más lejos de los registros reales que los registros reales entre sí. No se detecta riesgo de re-identificación.

**El efecto del tamaño del dataset es visible y esperado** — Titanic (891 filas) puntúa 0.84 BUENO, Plane Crashes (2.709 filas) puntúa 0.93 EXCELENTE, y California Housing (20.640 filas) puntúa 0.98 EXCELENTE. Más datos dan al modelo de difusión más señal de la que aprender.

---

## Arquitectura

```
synthetic-data-engine/
├── src/
│   ├── main.py                        # punto de entrada
│   │
│   ├── analyzer/                      # Fase 1 — 5 capas de detección
│   │   ├── metadata_extractor.py      # inspección estructural, sin valores brutos almacenados
│   │   ├── semantic_analyzer.py       # clasificación IIP con GPT-4 via metadatos de señales
│   │   ├── regex_miner.py             # extracción local de patrones para columnas de código
│   │   ├── distribution_profiler.py   # Shapiro-Wilk, asimetría, recomendación de transformación
│   │   ├── correlation_analyzer.py    # Pearson / Spearman / V de Cramér / η²
│   │   └── dataset_analyzer.py        # orquestador → DatasetProfile
│   │
│   ├── preprocessor/                  # Fase 2 — pipeline modular de 8 pasos
│   │   ├── contracts.py               # dataclasses compartidos + constantes tipadas
│   │   ├── preprocessor.py            # orquestador
│   │   ├── steps/
│   │   │   ├── step0_null_normalizer.py
│   │   │   ├── step1_column_filter.py
│   │   │   ├── step2_missing_handler.py
│   │   │   ├── step3_zero_inflated.py
│   │   │   ├── step4_numeric_transform.py
│   │   │   ├── step5_categorical_encoder.py
│   │   │   ├── step6_normalizer.py
│   │   │   └── step7_tensor_builder.py
│   │   └── inverter.py                # transformación inversa completa
│   │
│   ├── diffusion/                     # Fase 3 — modelo de difusión personalizado
│   │   ├── tabddpm.py                 # modelo principal + bucle de entrenamiento
│   │   ├── mlp.py                     # ResidualBlock + SinusoidalEmbedding
│   │   ├── noise_scheduler.py         # calendario coseno
│   │   ├── loss.py                    # MixedLoss + CorrelationLoss
│   │   └── tuner.py                   # HyperTuner + EarlyStopping
│   │
│   ├── generator/                     # Fase 4 — tensor → esquema original
│   │   ├── generator.py
│   │   ├── type_enforcer.py           # snap discreto para ordinales/binarias
│   │   ├── column_builder.py          # generación con Faker + regex
│   │   ├── null_validator.py          # corrección de tasa de nulos
│   │   └── calibrator.py             # calibración cuantil post-hoc
│   │
│   └── evaluator/                     # Fase 5 — puntuación en 6 dimensiones
│       ├── evaluator.py               # orquestación Evaluador A / B
│       ├── fidelity.py                # KS + Wasserstein + TVD
│       ├── relationships.py           # Pearson/Spearman + Frobenius
│       ├── utility.py                 # TSTR — 3 modelos ML
│       ├── detectability.py           # clasificador adversarial
│       ├── privacy.py                 # NND + Distance Ratio
│       ├── diversity.py               # Varianza + PCA + Cobertura
│       └── report.py                  # generador de informe de evaluación
│
├── data/                              # CSVs de entrada (no subidos al repo)
├── output/                            # synthetic.csv + evaluation_report.txt
├── models/                            # checkpoints .pt guardados
├── assets/                            # banner, diagramas
├── docs/                              # informe técnico PDF
├── requirements.txt
└── .env.example
```

---

## Stack Tecnológico

| Capa | Librería |
|------|----------|
| Deep learning | PyTorch 2.x |
| Procesamiento de datos | pandas, numpy, scipy |
| Evaluación ML | scikit-learn |
| Generación de IIP | Faker |
| Análisis semántico | OpenAI API (GPT-4) |
| Tests estadísticos | scipy.stats (KS, Shapiro-Wilk, Wasserstein) |

---

## Limitaciones

Siendo honestos sobre lo que aún no funciona bien:

- **Distribuciones de cola pesada** (p.ej. importes financieros o variables con outliers extremos) siguen siendo el caso más difícil. `log1p` ayuda significativamente, pero la cola extrema está submuestreada en datasets pequeños — esta es una limitación fundamental del tamaño del dataset, no del modelo.
- **Categóricas de alta cardinalidad** con valores reales ruidosos o malformados degradan las puntuaciones TVD. El modelo genera valores de formato válido pero no puede replicar completamente una distribución real ruidosa.
- **Tiempo de entrenamiento en CPU** en datasets grandes (~54k filas) tarda 30–60 min por ejecución. CUDA lo reduce a ~5 min.
- **Columnas de texto libre** se omiten en v1 — se registran en el perfil pero no se generan. La generación con NLP está en el roadmap.

El proyecto aún no soporta: datos de series temporales, esquemas relacionales multi-tabla, ni input en streaming.

---

## Roadmap

- [ ] Demo interactiva en Hugging Face Spaces
- [ ] Fix de cuantil para ordinales de baja cardinalidad (≤10 valores únicos → `transform=none`)
- [ ] Integración opcional con Microsoft Presidio para validación de IIP a nivel de valor
- [ ] Generación NLP para columnas de texto libre
- [ ] Soporte para series temporales
- [ ] Esquemas relacionales multi-tabla

---

## Cita

Si utilizas este sistema en investigación o producción:

```bibtex
@software{hernandez2026tabsynth,
  author  = {Hernández Calderón, Laren},
  title   = {tabsynth: Auto-adaptive synthetic tabular data generation via diffusion models},
  year    = {2026},
  url     = {https://github.com/LarenHernandezCalderon/synthetic-data-engine}
}
```

La arquitectura de difusión está inspirada en — pero no es una reimplementación de — el artículo original de TabDDPM:

> Kotelnikov, A., Baranchuk, D., Rubachev, I., & Babenko, A. (2022). *TabDDPM: Modelling Tabular Data with Diffusion Models*. ICML 2023.

Diferencias clave con el original: esta implementación usa un MLP residual con embeddings sinusoidales de timestep y una pérdida mixta solo gaussiana (MSE + BCE + CE) en lugar del proceso de difusión multinomial para features categóricas descrito en el artículo. El pipeline de análisis de cinco capas, la pérdida de entrenamiento consciente de correlaciones y el marco de evaluación de seis dimensiones son originales de este proyecto.

---

## Licencia

MIT — ver [`LICENSE`](LICENSE).

---

## Autora

**Laren Hernández Calderón**

[GitHub](https://github.com/LarenHernandezCalderon) · [LinkedIn](https://www.linkedin.com/in/laren-hern%C3%A1ndez-a9037a2a5/)

---

<sub>Hecho con cuidado. Si te ha sido útil, una ⭐ en el repositorio significa mucho.</sub>