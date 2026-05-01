<p align="right">
  🌐 <a href="README.md">English</a> · <strong>Español</strong>
</p>

<div align="center">
  <img src="docs/assets/banner.png" alt="tabsynth" width="100%"/>
</div>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square" alt="Python"/></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="#"><img src="https://img.shields.io/badge/OpenAI-GPT--4-412991?style=flat-square&logo=openai&logoColor=white" alt="OpenAI"/></a>
  <a href="#"><img src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square" alt="Licencia"/></a>
  <a href="#"><img src="https://img.shields.io/badge/RGPD-compliant-success?style=flat-square" alt="RGPD"/></a>
  <a href="#"><img src="https://img.shields.io/badge/estado-prototipo-blue?style=flat-square" alt="Estado"/></a>
</p>

<p align="center">
  <b>Generación de datos sintéticos tabulares auto-adaptativa mediante modelos de difusión.</b><br>
  Dale cualquier CSV. Él se encarga.
</p>

<p align="center">
  <a href="#por-qué-existe-esto">Por qué existe esto</a> ·
  <a href="#-los-datos-no-salen">Seguridad</a> ·
  <a href="#primeros-pasos">Primeros pasos</a> ·
  <a href="#cómo-funciona">Cómo funciona</a> ·
  <a href="#resultados">Resultados</a> ·
  <a href="#arquitectura">Arquitectura</a> ·
  <a href="#limitaciones">Limitaciones</a>
</p>

---

## Por qué existe esto

Hay un problema que se repite constantemente en proyectos de datos: tienes información valiosa pero no puedes compartirla. Los datos de pacientes no salen del hospital. Los registros financieros no se pueden ceder a un proveedor externo. El equipo de ML necesita datos para entrenar modelos, pero los datos reales están blindados por el RGPD, por contrato o por simple sentido común.

La respuesta habitual es "genera datos sintéticos". Fácil de decir. Difícil de hacer bien.

La mayoría de herramientas existentes tienen el mismo problema: requieren que alguien configure el esquema a mano, column por columna. Que indique qué es un nombre, qué es una categoría, qué transformación aplicar. Y aun así, el resultado suele ser datos que pasan un vistazo rápido pero fallan en cuanto los metes en un modelo — porque se han perdido las correlaciones, las distribuciones no cuadran, o simplemente nadie detectó que esa columna tenía un 70% de ceros.

**tabsynth** parte de otra premisa: el sistema tiene que entender los datos antes de generar nada. No tú. El sistema.

Un pipeline de cinco fases analiza cada columna de forma autónoma, detecta su naturaleza, sus dependencias con otras columnas y sus particularidades estadísticas. Con ese conocimiento configura y entrena un modelo de difusión que no solo replica cada columna por separado, sino la distribución conjunta completa — las relaciones entre variables incluidas.

Tres cosas que lo diferencian:

- **No necesita configuración.** Carga el CSV, ejecuta, obtén datos sintéticos. El sistema detecta automáticamente distribuciones con exceso de ceros, identifica si una columna es un identificador personal del RGPD, extrae patrones de campos estructurados y elige la transformación numérica más adecuada. Todo solo.
- **Las correlaciones se preservan activamente.** Las dependencias fuertes entre columnas no se esperan que "salgan solas" del entrenamiento — se inyectan explícitamente en la función de pérdida. Si dos variables están muy relacionadas en los datos reales, el modelo es penalizado si las rompe.
- **Los datos personales nunca se usan.** Nombres, emails, IDs directos: se generan desde cero con Faker. Los datos reales de esas columnas se descartan antes de que empiece el entrenamiento. No es una opción de configuración. Es cómo funciona el sistema.

---

## 🔒 Los datos no salen

Esto merece su propia sección porque es lo más importante.

En muchos contextos — sanidad, banca, recursos humanos — el simple hecho de que los datos salgan del entorno ya es un problema legal. Por eso **tabsynth está diseñado para que eso no ocurra nunca**, no como una política configurable, sino como consecuencia de cómo está construido.

¿Qué significa en la práctica?

El único servicio externo que interviene es GPT-4, y solo lo hace para clasificar semánticamente columnas de texto ambiguas — decidir si una columna contiene nombres, correos o códigos estructurados. Pero GPT-4 **nunca ve los valores reales**. Recibe únicamente métricas agregadas: longitud media de las cadenas, proporción de mayúsculas, si contiene números, rangos de cardinalidad. Nada más. Los valores reales no salen.

Todo lo demás — análisis de distribuciones, cálculo de correlaciones, detección de patrones en campos de código, entrenamiento del modelo, generación, evaluación — ocurre completamente en local. Sin llamadas externas, sin logs de datos, sin transferencias.

Y cuando el sistema termina, las columnas con datos personales directos (nombres, IDs, correos) no están en el output del modelo de difusión. Se generan aparte con Faker, usando el locale inferido automáticamente. El modelo nunca aprendió de esos valores porque nunca los vio.

Lo que sale del sistema es el dataset sintético. Y ese dataset, por cómo se ha construido, no puede vincularse a ninguna persona real — algo que además se mide con el Distance Ratio, una métrica que confirma que los registros sintéticos están más alejados de los reales que los propios registros reales entre sí.

---

## Demo

<div align="center">
  <img src="assets/pipeline.gif" alt="tabsynth pipeline" width="820"/>
  <br><sub>Pipeline completo con el dataset Titanic (891 filas × 12 columnas) — análisis, entrenamiento, generación y evaluación.</sub>
</div>

---

## Primeros pasos

```bash
git clone https://github.com/LarenHernandezCalderon/synthetic-data-engine.git
cd synthetic-data-engine
pip install -r requirements.txt
cp .env.example .env   # añade tu OPENAI_API_KEY
```

Edita `src/main.py` y cambia el dataset:

```python
DATASET = "titanic.csv"   # o "plane.csv", "housing.csv", tu propio CSV
```

Ejecuta:

```bash
cd src
python main.py
```

Obtendrás tres ficheros: `output/synthetic.csv` con los datos generados, `output/evaluation_report.txt` con el análisis de calidad detallado, y el checkpoint del modelo entrenado en `models/`.

---

## Cómo funciona

El pipeline tiene cinco fases. Cada una tiene una responsabilidad clara y le pasa un contrato tipado a la siguiente — `DatasetProfile` de la Fase 1 al preprocesador, `PreprocessorOutput` al modelo de difusión.

<div align="center">
  <img src="docs/assets/pipeline.png" alt="diagrama del pipeline de tabsynth" width="100%"/>
</div>

### Fase 1 — El analizador (5 capas)

Todo lo que viene después depende de lo que se detecta aquí. Si el análisis es bueno, el resto del pipeline puede hacer su trabajo. Si falla, nada lo compensa.

La primera capa (**MetadataExtractor**) inspecciona cada columna sin guardar valores. Para columnas de texto genera un perfil con señales estructurales — longitud media, proporción de mayúsculas, si tiene números, si tiene caracteres especiales — que describen la forma de los datos sin exponer su contenido.

Cuando esas señales no son suficientes para determinar el tipo de una columna, entra la segunda capa (**SemanticAnalyzer**). Envía las métricas agregadas a GPT-4, que devuelve el tipo semántico (`full_name`, `email`, `code`, `free_text`…), la categoría RGPD y el locale para generación. Nunca los valores reales — solo las señales.

Si la columna resulta ser un campo de código estructurado — números de ticket, referencias de producto, IDs con formato propio — actúa la capa **RegexMiner**, que analiza los patrones localmente y sintetiza una expresión regular. Solo sale el patrón. Los valores se quedan dentro.

La capa de distribuciones (**DistributionProfiler**) determina la forma estadística de cada columna numérica: si es exponencial, bimodal, tiene exceso de ceros, cuántos outliers hay. Con esa información recomienda la transformación más adecuada antes de meterla al modelo.

Y la capa de correlaciones (**CorrelationAnalyzer**) calcula las dependencias entre todos los pares de columnas — Pearson y Spearman para pares numéricos, V de Cramér para categóricos, η² para mixtos. Los pares que superan el umbral de correlación fuerte (≥0.40) se guardan y se usarán más adelante para penalizar al modelo si los rompe.

### Fase 2 — El preprocesador (8 pasos)

Convierte el DataFrame original en un tensor listo para el modelo de difusión. Cada paso es independiente e invertible:

| Paso | Qué hace |
|------|----------|
| 0 — Null Normaliser | Detecta nulos enmascarados como `-1`, `999` o `N/A` y los convierte a `NaN` |
| 1 — Column Filter | Decide qué va al tensor, qué se genera externamente (Faker, regex) y qué se omite |
| 2 — Missing Handler | Crea indicadores binarios `_missing` e imputa con mediana o moda |
| 3 — Zero-Inflated Split | Divide columnas con exceso de ceros en `_flag` (¿tiene valor?) + `_value` (cuánto) |
| 4 — Numeric Transform | Aplica log1p o QuantileTransformer según lo que recomendó la Fase 1 |
| 5 — Categorical Encoder | Codificación ordinal y one-hot respetando el orden de las categorías |
| 6 — Normalizer | Escala todo a [0, 1] con MinMax |
| 7 — Tensor Builder | Construye el `TensorDataset` tipado con metadatos por feature |

La transformación inversa completa está en `inverter.py` — permite reconstruir el DataFrame original a partir del tensor generado.

### Fase 3 — El modelo de difusión

Un modelo de difusión gaussiana para datos tabulares mixtos, inspirado en [TabDDPM (Kotelnikov et al., 2022)](https://arxiv.org/abs/2209.15421) pero con varias diferencias importantes.

La arquitectura es un MLP residual con embeddings sinusoidales de timestep. Cada bloque aplica normalización de capa, una proyección lineal, activación SiLU, dropout y otra proyección, más una conexión de salto. El timestep se proyecta en cada bloque para que el modelo sepa en qué punto del proceso de denoising está.

La función de pérdida combina MSE para features continuas, BCE para binarias y CrossEntropy para categóricas. Y sobre esa base se añade el `CorrelationLoss` — un término que en cada paso de entrenamiento compara la matriz de correlaciones del batch generado con los valores objetivo que detectó la Fase 1, y penaliza al modelo si se aleja de ellos:

```
pérdida_total = mixed_loss + corr_weight × correlation_loss
```

Antes de entrenar, el `HyperTuner` hace una búsqueda rápida entre cinco configuraciones de red (50 épocas de sondeo cada una) y selecciona la que mejor loss obtiene. La configuración ganadora entrena hasta convergencia con early stopping.

| Candidato | Capas ocultas | lr | batch |
|-----------|-------------|-----|-------|
| A — ligero y rápido | [128, 128] | 1e-3 | 256 |
| B — equilibrado | [256, 256] | 1e-3 | 64 |
| C — más profundo | [256, 256, 128] | 1e-3 | 64 |
| D — ancho con lr bajo | [512, 256] | 5e-4 | 128 |
| E — red profunda y ancha | [512, 512, 256] | 1e-3 | 64 |

### Fase 4 — El generador

Parte de ruido gaussiano puro y aplica 1.000 pasos de difusión inversa para obtener el tensor sintético. Luego invierte todas las transformaciones del preprocesador y reconstruye el esquema original:

- Las columnas ordinales se redondean al entero válido más cercano
- Las columnas con datos personales directos se generan con Faker, usando el locale inferido en la Fase 1
- Los campos de código se generan con la expresión regular que extrajo el RegexMiner
- La tasa de nulos de cada columna se preserva tal como estaba en los datos reales
- Una calibración post-hoc corrige la deriva distribucional residual en columnas continuas

### Fase 5 — El evaluador (6 dimensiones)

La calidad del dataset sintético se mide desde seis ángulos distintos. Cinco de ellos contribuyen a la puntuación global:

| Dimensión | Cómo se mide | Peso |
|-----------|--------|------|
| Fidelidad | Test KS + distancia de Wasserstein + TVD | 30% |
| Relaciones | Norma de Frobenius sobre matrices de correlación + delta por par | 25% |
| Utilidad (TSTR) | Entrena en sintético, evalúa en real — 3 modelos ML | 30% |
| Privacidad | Distance Ratio basado en vecinos más cercanos | 10% |
| Diversidad | Ratio de varianza + dispersión PCA + cobertura de categorías | 5% |

El sexto, la **Detectabilidad**, es diagnóstico: un RandomForest adversarial intenta distinguir registros reales de sintéticos. Si lo consigue con facilidad, hay algo que mejorar. No suma a la puntuación global porque las columnas generadas externamente (Faker, regex, IDs) lo inflarían de forma artificial.

Las columnas con datos personales directos se eliminan de todos los evaluadores antes de cualquier cálculo. Ningún valor real entra en ningún clasificador.

---

## Resultados

Tres datasets con retos muy distintos: Titanic, pequeño y lleno de casuísticas (datos personales, nulos, distribuciones con cola, ceros inflados); Plane Crashes, tamaño medio con correlaciones físicas muy marcadas entre variables continuas; y California Housing, un dataset geoespacial grande donde casi todo está correlacionado con todo.

| Dataset | Filas | Cols | Fidelidad | Relaciones | Utilidad | Privacidad | Diversidad | **Global** |
|---------|-------|------|-----------|------------|----------|------------|------------|------------|
| **Titanic** | 891 | 12 | 0.78 | 0.84 | 0.81 | 1.00 | 0.96 | **0.84** 🟡 |
| **Plane Crashes** | 2.709 | 8 | 0.80 | 0.98 | 1.00 | 1.00 | 0.98 | **0.93** 🟢 |
| **California Housing** | 20.640 | 10 | 0.97 | 0.97 | 0.99 | 1.00 | 0.98 | **0.98** 🟢 |

> 🟢 ≥ 0.85 listo para producción · 🟡 ≥ 0.70 válido para la mayoría de casos · 🟠 ≥ 0.55 usar con precaución

Todos los resultados con `SEED = 12`, totalmente reproducibles.

**Lo que estos números significan de verdad:**

En Plane Crashes, los tres clasificadores entrenados con datos sintéticos obtienen exactamente la misma precisión al evaluarlos sobre datos reales que si los hubieran entrenado con datos reales desde el principio. Diferencia cero. En California Housing la diferencia media es del 0.1%. En Titanic, el dataset más difícil de los tres, es del 1.9% — y eso con 891 filas, un 20% de valores ausentes y varias columnas de datos personales.

Las correlaciones se preservan porque el modelo está explícitamente penalizado por romperlas. En California Housing se conservan 35 de 36 pares críticos. En Plane Crashes, todos los pares críticos dentro de tolerancia.

La privacidad da 1.00 en los tres casos. El Distance Ratio confirma que los registros sintéticos están más alejados de los reales que los propios registros reales entre sí. No se detecta riesgo de re-identificación.

Y el patrón es claro: cuantos más datos, mejor aprende el modelo. Titanic con 891 filas llega a 0.84. Housing con 20k filas llega a 0.98.

---

## Arquitectura

```
synthetic-data-engine/
├── src/
│   ├── main.py
│   │
│   ├── analyzer/                      # Fase 1 — análisis autónomo
│   │   ├── metadata_extractor.py      # inspección estructural sin guardar valores
│   │   ├── semantic_analyzer.py       # GPT-4 clasifica por señales, no por valores
│   │   ├── regex_miner.py             # patrones locales para columnas de código
│   │   ├── distribution_profiler.py   # Shapiro-Wilk, asimetría, transformación
│   │   ├── correlation_analyzer.py    # Pearson / Spearman / V de Cramér / η²
│   │   └── dataset_analyzer.py        # orquestador → DatasetProfile
│   │
│   ├── preprocessor/                  # Fase 2 — 8 pasos modulares e invertibles
│   │   ├── contracts.py
│   │   ├── preprocessor.py
│   │   ├── steps/
│   │   │   ├── step0_null_normalizer.py
│   │   │   ├── step1_column_filter.py
│   │   │   ├── step2_missing_handler.py
│   │   │   ├── step3_zero_inflated.py
│   │   │   ├── step4_numeric_transform.py
│   │   │   ├── step5_categorical_encoder.py
│   │   │   ├── step6_normalizer.py
│   │   │   └── step7_tensor_builder.py
│   │   └── inverter.py
│   │
│   ├── diffusion/                     # Fase 3 — modelo de difusión
│   │   ├── tabddpm.py                 # arquitectura + entrenamiento
│   │   ├── mlp.py                     # ResidualBlock + SinusoidalEmbedding
│   │   ├── noise_scheduler.py         # calendario coseno
│   │   ├── loss.py                    # MixedLoss + CorrelationLoss
│   │   └── tuner.py                   # búsqueda de hiperparámetros + early stopping
│   │
│   ├── generator/                     # Fase 4 — tensor → esquema original
│   │   ├── generator.py
│   │   ├── type_enforcer.py
│   │   ├── column_builder.py          # Faker + regex
│   │   ├── null_validator.py
│   │   └── calibrator.py
│   │
│   └── evaluator/                     # Fase 5 — 6 dimensiones de calidad
│       ├── evaluator.py
│       ├── fidelity.py
│       ├── relationships.py
│       ├── utility.py
│       ├── detectability.py
│       ├── privacy.py
│       ├── diversity.py
│       └── report.py
│
├── data/
├── output/
├── models/
├── requirements.txt
└── .env.example
```

---

## Stack tecnológico

| Capa | Librería |
|------|----------|
| Deep learning | PyTorch 2.x |
| Procesamiento de datos | pandas, numpy, scipy |
| Evaluación ML | scikit-learn |
| Generación de datos personales | Faker |
| Análisis semántico | OpenAI API (GPT-4) |
| Tests estadísticos | scipy.stats (KS, Shapiro-Wilk, Wasserstein) |

---

## Limitaciones

Sin rodeos:

- Las distribuciones con colas muy pesadas siguen siendo el caso más difícil. `log1p` ayuda mucho, pero los extremos quedan submuestreados en datasets pequeños. Es una limitación del volumen de datos, no del modelo.
- Las categóricas de alta cardinalidad con valores reales ruidosos degradan el TVD. El modelo genera valores con formato correcto pero no puede replicar perfectamente una distribución real con mucho ruido.
- El entrenamiento en CPU en datasets grandes puede tardar entre 20 y 60 minutos. Con GPU se reduce a unos 5 minutos.
- Las columnas de texto libre no se generan en esta versión — se omiten y se registran en el perfil. La generación con NLP está en el roadmap.

No soporta todavía: series temporales, esquemas relacionales con varias tablas, ni entrada en streaming.

---

## Roadmap

- [ ] Demo interactiva en Hugging Face Spaces
- [ ] Corrección para ordinales de baja cardinalidad (≤10 valores únicos → `transform=none`)
- [ ] Integración opcional con Microsoft Presidio para validación de datos personales a nivel de valor
- [ ] Generación de columnas de texto libre con NLP
- [ ] Soporte para series temporales
- [ ] Esquemas relacionales multi-tabla

---

## Cita

Si usas esto en investigación o producción:

```bibtex
@software{hernandez2026tabsynth,
  author  = {Hernández Calderón, Laren},
  title   = {tabsynth: Auto-adaptive synthetic tabular data generation via diffusion models},
  year    = {2026},
  url     = {https://github.com/LarenHernandezCalderon/synthetic-data-engine}
}
```

La arquitectura está inspirada en TabDDPM, aunque no es una reimplementación:

> Kotelnikov, A., Baranchuk, D., Rubachev, I., & Babenko, A. (2022). *TabDDPM: Modelling Tabular Data with Diffusion Models*. ICML 2023.

Las diferencias principales: MLP residual con embeddings sinusoidales de timestep, función de pérdida mixta gaussiana (MSE + BCE + CE) en lugar del proceso multinomial del original, y el pipeline de análisis de cinco capas, el `CorrelationLoss` y el marco de evaluación de seis dimensiones son originales de este proyecto.

---

## Licencia

MIT — ver [`LICENSE`](LICENSE).

---

## Autora

**Laren Hernández Calderón**

[GitHub](https://github.com/LarenHernandezCalderon) · [LinkedIn](https://www.linkedin.com/in/laren-hern%C3%A1ndez-a9037a2a5/)

---

<sub>Hecho con cuidado. Si te ha resultado útil, una ⭐ en el repositorio significa mucho.</sub>