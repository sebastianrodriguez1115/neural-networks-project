# `src/` — Índice de módulos

## Estructura

```
src/
├── bvbrc/
│   ├── __init__.py
│   ├── _http.py
│   ├── amr.py
│   └── genomes.py
│
├── data_pipeline/
│   ├── __init__.py
│   ├── constants.py
│   ├── cleaning.py
│   ├── features.py
│   └── pipeline.py
│
├── train/
│   ├── __init__.py
│   ├── evaluate.py
│   └── loop.py
│
├── models/
│   ├── base_dataset.py
│   ├── mlp/
│   ├── bigru/
│   ├── multi_bigru/
│   ├── hier_bigru/
│   ├── hier_set/
│   └── hier_set_v2/
│
├── eda.py
└── README.md
```

## Módulos

### `bvbrc/` — Cliente HTTP para la API REST de BV-BRC

Descarga datos de la base de datos BV-BRC (etiquetas AMR y genomas FASTA) para organismos ESKAPE.

- **`_http.py`** — Utilidades HTTP internas: reintentos con backoff exponencial, parseo de `Content-Range` para paginación, constantes compartidas (`PAGE_SIZE`, `MAX_RETRIES`). Módulo privado, no importar desde fuera del paquete.
- **`amr.py`** — `AMRFetcher`: descarga paginada de etiquetas AMR desde el endpoint `genome_amr`. Filtra por evidencia de laboratorio y fenotipos binarios (Resistant/Susceptible). Exporta también `ESKAPE_TAXON_IDS`.
- **`genomes.py`** — `GenomeFetcher` (descarga individual) y `GenomeBatchFetcher` (descarga en lote) de genomas FASTA desde el endpoint `genome_sequence`. Tolerante a fallos individuales; omite genomas ya descargados.

### `data_pipeline/` — Preprocesamiento de datos

Transforma el CSV crudo de etiquetas + archivos FASTA en features `.npy` listos para entrenamiento.

- **`constants.py`** — Constantes compartidas: `KMER_SIZES` (3,4,5), `TOTAL_KMER_DIM` (1344), `BIGRU_PAD_DIM` (1024), `RANDOM_SEED` (42), ratios de split, `BASE_TO_INDEX`. Hier: `HIER_KMER_K=4`, `HIER_N_SEGMENTS=256`, `HIER_KMER_DIM=256` (v1); multi-escala: `HIER_KMER_SIZES=[3,4,5]`, `HIER_KMER_DIM_MULTI=1344` (v2).
- **`cleaning.py`** — `LabelCleaner`: elimina pares contradictorios y duplicados consistentes del CSV de etiquetas. `GenomeFilter`: filtra genomas por disponibilidad de FASTA y longitud mínima (0.5 Mb).
- **`features.py`** — `KmerExtractor`: histogramas de k-meros (k=3,4,5) con rolling hash O(n); genera vector MLP (1344-dim) y matriz BiGRU ([1024, 3]). `to_tiled_histogram_matrix()` para HierSet v1 (256 × 256) y `to_tiled_multiscale_matrix()` para HierSet v2 (256 × 1344). También contiene `split_genomes()` (split estratificado 70/15/15 por genome_id), `normalize_features()` (z-score con estadísticas solo del train set), `build_antibiotic_index()` y `mlp_vector_to_bigru_matrix()`.
- **`pipeline.py`** — `run_pipeline()`: orquestador del preprocesamiento (limpieza → filtro → índice antibióticos → split → k-meros → normalización). `extract_and_save_hier()` y `extract_and_save_hier_multi()` para las features jerárquicas v1 y v2.

### `eda.py` — Análisis exploratorio

`run_eda()` imprime un reporte completo en consola: resumen general, distribución por especie, balance de clases, ranking de antibióticos, calidad de datos, outliers, baseline benchmark (majority class F1=0.7366) y análisis genómico opcional. `export_contradictions()` exporta pares con etiquetas contradictorias a CSV.

### `models/` — Modelos y datasets de PyTorch

Cada subdirectorio contiene `model.py` (arquitectura) y `dataset.py` (subclase de `BaseAMRDataset`):

- **`base_dataset.py`** — `BaseAMRDataset`: carga `splits.csv`, `antibiotic_index.csv` y `train_stats.json`; expone `load_pos_weight()`. Base para todos los datasets específicos de modelo.
- **`mlp/`** — `AMRMLP`: vector 1344-dim → Dense(512)+Dropout → Dense(128)+Dropout → logit.
- **`bigru/`** — `AMRBiGRU`: matriz [1024,3] → BiGRU(128) → BahdanauAttention → logit.
- **`multi_bigru/`** — `AMRMultiBiGRU`: tres streams k=3,4,5 con `KmerStream` (LayerNorm sin affine + `bin_importance` + attention pooling) → fusión softmax condicionada por antibiótico → logit.
- **`hier_bigru/`** — `AMRHierBiGRU`: matriz [HIER_N_SEGMENTS, 256] → BiGRU(2 capas) → BahdanauAttention → logit. (No competitivo frente a HierSet.)
- **`hier_set/`** — `AMRHierSet`: mismo input → proyección por segmento + dropout → cross-attention query-key condicionada en antibiótico → logit. Permutation-invariant. **Mejor modelo del proyecto** (F1=0.8900, AUC=0.9368).
- **`hier_set_v2/`** — `AMRHierSetV2`: input multi-escala `[HIER_N_SEGMENTS, 1344]` (k=3,4,5 concatenados) → proyección por segmento → **multi-head cross-attention (H=4, d_head=32)** → logit. Misma estructura que v1 pero con más información por segmento y atención multi-cabeza. **Resultado negativo**: F1=0.8895, Recall=0.8971, AUC=0.9366 — empata con v1 en F1/AUC y pierde en Recall (< 0.90). v1 se mantiene como mejor modelo. Ver `docs/5_experiments.md` Experimento 7.

### `train/` — Entrenamiento y evaluación

Paquete con dos módulos:

- **`evaluate.py`** — `collect_predictions()` (inferencia sin gradientes), `compute_metrics()` (accuracy, precision, recall, F1, AUC-ROC desde arrays numpy), `find_optimal_threshold()` (umbral que maximiza F1), y `evaluate()` que compone las tres anteriores.
- **`loop.py`** — `set_seed()` (reproducibilidad), `detect_device()` (CUDA → MPS → CPU), `train_epoch()` (una pasada forward+backward), y `train()` como orquestador con AdamW, early stopping y checkpoint sobre val_F1, ReduceLROnPlateau sobre val_F1. Genera: `best_model.pt`, `metrics.json`, `history.csv`, `history.png`.

## Punto de entrada

El CLI está en `main.py` (raíz del proyecto), no en `src/`. Usa Typer y expone los comandos:

| Comando | Módulo que invoca | Qué hace |
|---|---|---|
| `download-amr` | `bvbrc.amr` | Descarga etiquetas AMR de BV-BRC para organismos ESKAPE |
| `download-genomes` | `bvbrc.genomes` | Descarga FASTAs de los genome_id del CSV de etiquetas |
| `eda` | `eda` | Análisis exploratorio completo con reporte en consola |
| `export-contradictions-cmd` | `eda` | Exporta pares con etiquetas contradictorias a CSV |
| `prepare-data` | `data_pipeline.pipeline` | Pipeline completo: limpieza → filtro → split → k-meros → normalización |
| `prepare-hier` | `data_pipeline.pipeline` | Extrae histogramas segmentados (HIER_N_SEGMENTS×256) para HierBiGRU y HierSet |
| `prepare-hier-multi` | `data_pipeline.pipeline` | Extrae histogramas multi-escala segmentados (HIER_N_SEGMENTS×1344, k=3,4,5) para HierSet v2 |
| `train-mlp` | `models.mlp` | Entrena el MLP y evalúa sobre test set |
| `train-bigru` | `models.bigru` | Entrena la BiGRU + Attention |
| `train-multi-bigru` | `models.multi_bigru` | Entrena el encoder multi-stream order-independent |
| `train-hier-bigru` | `models.hier_bigru` | Entrena la HierBiGRU sobre histogramas segmentados |
| `train-hier-set` | `models.hier_set` | Entrena el HierSet (encoder de conjunto) — mejor modelo |
| `train-hier-set-v2` | `models.hier_set_v2` | Entrena el HierSet v2 (multi-head attention + multi-escala) |

## Flujo de datos

```
BV-BRC API  ──→  data/raw/fasta/*.fna     (genomas FASTA)
            ──→  data/processed/amr_labels.csv  (etiquetas AMR)
                        │
                   prepare-data
                        │
                        ├── cleaned_labels.csv
                        ├── antibiotic_index.csv
                        ├── splits.csv
                        ├── mlp/*.npy          (vector 1344-dim por genoma)
                        ├── bigru/*.npy        (matriz [1024,3] por genoma)
                        ├── mlp_mean.npy
                        └── mlp_std.npy
                        │
                   prepare-hier
                        └── hier_bigru/*.npy   (matriz [256, 256] por genoma — HierBiGRU, HierSet)
                        │
                   prepare-hier-multi
                        └── hier_set_v2/*.npy  (matriz [256, 1344] por genoma — HierSet v2)
```
