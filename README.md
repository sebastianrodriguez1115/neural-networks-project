# Predicción de resistencia antimicrobiana con redes neuronales

Proyecto académico: clasificación binaria de resistencia/susceptibilidad antibiótica en bacterias del grupo ESKAPE a partir de secuencias genómicas completas (WGS). El antibiótico entra como feature al modelo (embedding aprendido) y se exploran múltiples arquitecturas sobre representaciones de k-meros del genoma.

**Modelo seleccionado:** HierSet — encoder de conjunto sobre 256 segmentos de histograma k=4 con cross-attention condicionada por antibiótico (F1=0.8900, AUC=0.9368, Recall=0.9088).

---

## Uso e instalación

Ver [`docs/1_environment.md`](docs/1_environment.md) para instrucciones de instalación, comandos CLI y cómo correr los tests.

Para agentes IA: leer [`AGENTS.md`](AGENTS.md) — convenciones, archivos clave y checklist al terminar tareas.

---

## Modelos implementados

| Modelo | Representación | F1 | AUC-ROC | Recall |
|---|---|---|---|---|
| **HierSet** (seleccionado) | 256 segmentos × hist k=4 + set encoder + cross-attention | **0.8900** | **0.9368** | 0.9088 |
| HierSet v2 | Multi-head attention + hist multi-escala (k=3,4,5) por segmento | 0.8895 | 0.9366 | 0.8971 |
| MLP (baseline) | Concat. histogramas globales k=3,4,5 (1344 dims) | 0.8600 | 0.9035 | 0.9165 |
| BiGRU + Attention | Histogramas globales paddeados a `[1024, 3]` | 0.8566 | 0.8998 | 0.9032 |
| MultiBiGRU | 3 streams order-independent (k=3,4,5) con `bin_importance` | 0.8514 | 0.8944 | 0.8925 |
| HierBiGRU | 256 segmentos + BiGRU (sesgo secuencial) | 0.8307 | 0.8539 | 0.8788 |
| Token BiGRU | Secuencia de tokens k=4 submuestreada | — | — | — |

Detalle completo en [`docs/5_experiments.md`](docs/5_experiments.md).

---

## Estructura del proyecto

```
neural-networks-project/
├── main.py                   # CLI (Typer) — punto de entrada
├── AGENTS.md                 # Instrucciones para agentes IA
├── src/
│   ├── bvbrc/                # Descarga desde BV-BRC
│   │   ├── _http.py          #   utilidades HTTP
│   │   ├── amr.py            #   etiquetas AMR (genome_amr)
│   │   └── genomes.py        #   genomas FASTA (genome_sequence)
│   ├── data_pipeline/        # Preprocesamiento
│   │   ├── cleaning.py       #   limpieza y filtrado de labels
│   │   ├── constants.py      #   constantes compartidas (k, dims, seed, …)
│   │   ├── features.py       #   KmerExtractor: histogramas globales,
│   │   │                     #                  segmentados, multi-escala, tokens
│   │   └── pipeline.py       #   orquestación end-to-end
│   ├── models/
│   │   ├── base_dataset.py   #   BaseAMRDataset (común a todos los modelos)
│   │   ├── mlp/              #   AMRMLP
│   │   ├── bigru/            #   AMRBiGRU + Attention
│   │   ├── multi_bigru/      #   AMRMultiBiGRU (3 streams k=3,4,5)
│   │   ├── token_bigru/      #   AMRTokenBiGRU (descartado)
│   │   ├── hier_bigru/       #   AMRHierBiGRU (256 segs + BiGRU)
│   │   ├── hier_set/         #   AMRHierSet ← mejor modelo
│   │   └── hier_set_v2/      #   AMRHierSetV2 (multi-head + multi-escala)
│   ├── train/                # Loop de entrenamiento, early stopping, evaluación
│   │   ├── loop.py
│   │   └── evaluate.py
│   ├── analyze_attention.py  # Análisis post-hoc de pesos de atención (HierSet)
│   └── eda.py                # Análisis exploratorio
├── tests/
│   ├── bvbrc/
│   ├── data_pipeline/
│   ├── models/               # tests por arquitectura
│   └── test_train.py
├── data/
│   ├── raw/                  # genomas FASTA descargados
│   └── processed/            # etiquetas CSV, features (.npy), splits
├── results/                  # métricas, gráficas, checkpoints por modelo
├── scripts/                  # scripts auxiliares de análisis
├── notebooks/                # exploración y visualización
└── docs/
    ├── 1_environment.md      # stack, instalación, CLI
    ├── 2_eda.md              # hallazgos del EDA
    ├── 3_data_pipeline.md    # pipeline de datos
    ├── 4_models.md           # arquitecturas y decisiones de diseño
    ├── 5_experiments.md      # experimentos y criterio de éxito
    ├── PLAN_*.md             # planes de implementación por modelo
    ├── IDEAS_MEJORA_HIERSET.md
    ├── STUDY_GUIDE_DEEP_NN.md
    ├── PROGRESS.md           # estado por fases
    ├── CHANGELOG.md          # registro histórico
    ├── proposal/             # propuesta del proyecto
    └── reference/            # referencia técnica
```
