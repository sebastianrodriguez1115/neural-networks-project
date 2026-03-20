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
├── eda.py
├── models.py
└── train.py
```

## Módulos

### `bvbrc/` — Cliente HTTP para la API REST de BV-BRC

Descarga datos de la base de datos BV-BRC (etiquetas AMR y genomas FASTA) para organismos ESKAPE.

- **`_http.py`** — Utilidades HTTP internas: reintentos con backoff exponencial, parseo de `Content-Range` para paginación, constantes compartidas (`PAGE_SIZE`, `MAX_RETRIES`). Módulo privado, no importar desde fuera del paquete.
- **`amr.py`** — `AMRFetcher`: descarga paginada de etiquetas AMR desde el endpoint `genome_amr`. Filtra por evidencia de laboratorio y fenotipos binarios (Resistant/Susceptible). Exporta también `ESKAPE_TAXON_IDS`.
- **`genomes.py`** — `GenomeFetcher` (descarga individual) y `GenomeBatchFetcher` (descarga en lote) de genomas FASTA desde el endpoint `genome_sequence`. Tolerante a fallos individuales; omite genomas ya descargados.

### `data_pipeline/` — Preprocesamiento de datos

Transforma el CSV crudo de etiquetas + archivos FASTA en features `.npy` listos para entrenamiento.

- **`constants.py`** — Constantes compartidas: `KMER_SIZES` (3,4,5), `TOTAL_KMER_DIM` (1344), `BIGRU_PAD_DIM` (1024), `RANDOM_SEED` (42), ratios de split, `BASE_TO_INDEX`.
- **`cleaning.py`** — `LabelCleaner`: elimina pares contradictorios y duplicados consistentes del CSV de etiquetas. `GenomeFilter`: filtra genomas por disponibilidad de FASTA y longitud mínima (0.5 Mb).
- **`features.py`** — `KmerExtractor`: histogramas de k-meros (k=3,4,5) con rolling hash O(n); genera vector MLP (1344-dim) y matriz BiGRU ([1024, 3]). También contiene `split_genomes()` (split estratificado 70/15/15 por genome_id), `normalize_features()` (z-score con estadísticas solo del train set), `build_antibiotic_index()` y `mlp_vector_to_bigru_matrix()`.
- **`pipeline.py`** — `run_pipeline()`: orquestador que ejecuta los 6 pasos del preprocesamiento en secuencia (limpieza → filtro → índice antibióticos → split → k-meros → normalización).

### `eda.py` — Análisis exploratorio

`run_eda()` imprime un reporte completo en consola: resumen general, distribución por especie, balance de clases, ranking de antibióticos, calidad de datos, outliers, baseline benchmark (majority class F1=0.7366) y análisis genómico opcional. `export_contradictions()` exporta pares con etiquetas contradictorias a CSV.

### `models.py` — Arquitecturas de redes neuronales *(pendiente)*

- **MLP:** input genómico (1344) + antibiotic embedding (dim=49) → Dense(512, ReLU) + Dropout → Dense(128, ReLU) + Dropout → Dense(1)
- **BiGRU + Attention (Var A):** input [batch, 1024, 3] → BiGRU(hidden=128) → Bahdanau attention → context [batch, 256] + antibiotic embedding → Dense → Dense(1)

### `train.py` — Entrenamiento y evaluación *(pendiente)*

Loop de entrenamiento con Adam (lr=0.001), `BCEWithLogitsLoss` con `pos_weight`, early stopping (patience=10 sobre pérdida de validación). Métricas por epoch: loss train/val, accuracy, precision, recall, F1, AUC-ROC.

## Punto de entrada

El CLI está en `main.py` (raíz del proyecto), no en `src/`. Usa Typer y expone los comandos:

| Comando | Módulo que invoca | Qué hace |
|---|---|---|
| `download-amr` | `bvbrc.amr` | Descarga etiquetas AMR de BV-BRC para organismos ESKAPE |
| `download-genomes` | `bvbrc.genomes` | Descarga FASTAs de los genome_id del CSV de etiquetas |
| `eda` | `eda` | Análisis exploratorio completo con reporte en consola |
| `export-contradictions-cmd` | `eda` | Exporta pares con etiquetas contradictorias a CSV |
| `prepare-data` | `data_pipeline.pipeline` | Pipeline completo: limpieza → filtro → split → k-meros → normalización |

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
```
