# CHANGELOG

### 2026-04-16 (sesión 5)

#### Ajuste de umbral de despliegue HierSet (θ=0.40)
- Umbral de clasificación bajado de 0.4939 (óptimo en val F1) a **θ=0.40** en el modelo HierSet ya entrenado.
- Motivación clínica: en AMR un falso negativo (resistente clasificado como susceptible) es más costoso que un falso positivo. Bajar el umbral prioriza recall sobre precision.
- Resultado: Recall 0.9088 → **0.9289** (+0.020), F1 0.8900 → 0.8876 (−0.002), Precision 0.8743 → 0.8498. AUC sin cambio (0.9368).
- Tabla de sensibilidad explorada: θ=0.50→Rec=0.9066, θ=0.45→0.9187, θ=0.40→0.9289, θ=0.35→0.9383, θ=0.30→0.9474.
- Documentado en `docs/5_experiments.md`: experimento 6, tabla consolidada y conclusiones actualizadas.

### 2026-04-15 (sesión 4)

#### Entrenamiento HierBiGRU y HierSet con 256 segmentos (Fase 6.2 y 6.3)
- Ejecutado `prepare-hier --n-jobs -1`: 9060 genomas procesados en ~574 s. Datos en `data/processed/hier_bigru/`.
- **HierBiGRU** (256 segs): F1=0.8307, Recall=0.8788, AUC=0.8539. Early stopping época 43. No cumple criterio de éxito.
  - Las dependencias secuenciales de la BiGRU entre segmentos adyacentes perjudican la señal cuando los genes de resistencia están distribuidos sin orden fijo.
- **HierSet** (256 segs): F1=0.8900, Recall=0.9088, AUC=**0.9368**. Early stopping época 65. **Mejor AUC del proyecto.**
  - El encoder permutation-invariant con cross-attention condicionada por antibiótico supera a todos los modelos anteriores en AUC-ROC.
- Comparación final documentada en `docs/5_experiments.md`: HierSet lidera en F1 y AUC; MLP tiene mayor Recall (0.9165).
- Docs actualizados: `docs/5_experiments.md`, `docs/PROGRESS.md`, `docs/CHANGELOG.md`.

### 2026-04-15 (sesión 3)

#### Optimización de HierSet + fixes de code review (Fase 6)
- **HIER_N_SEGMENTS** aumentado de 64 a **256** (`src/data_pipeline/constants.py`): resolución ~17 kb/segmento, genes de resistencia representan ~5–12% del histograma del segmento (vs ~1–3% con 64 segs). Requiere re-ejecutar `prepare-hier`.
- **HierSet v2** — tres cambios de arquitectura:
  - Regularización: dropout después de `proj`, `weight_decay` default 1e-3 (antes 1e-4).
  - Cross-attention query-key: `score(s,a) = h_s · q_a / √D` donde `q_a = attn_query(ab_emb)`. Reemplaza la concatenación simple (que era shift-invariant en softmax y no condicionaba la atención). Nuevo test: `test_attention_varies_by_antibiotic`.
- **MultiBiGRU** — fix: `LayerNorm(elementwise_affine=False)` para que `bin_importance` sea el único prior por identidad de bin. Reentrenado: F1=0.8514, Recall=0.8925, AUC=0.8944.
- **Fix de test**: `test_tiled_histogram_matrix_no_boundary_kmers` corregido para verificar AAAT (k-mer espurio real) en lugar de AATT.
- **Docs actualizados**: `AGENTS.md`, `docs/1_environment.md`, `src/README.md`, `docs/PROGRESS.md` sincronizados con estado actual del código.
- Resultados viejos (64 segmentos) de `results/hier_bigru/` y `results/hier_set/` eliminados por incompatibilidad de shape.
- **Pendiente:** re-ejecutar `prepare-hier --n-jobs -1` con 256 segmentos, luego entrenar HierBiGRU y HierSet.

### 2026-04-15 (sesión 2)

#### Extracción de histogramas segmentados — `prepare-hier` completado
- Ejecutado `prepare-hier` con `--n-jobs -1` (25 workers, 80% CPUs): 9060 genomas procesados correctamente.
- Datos disponibles en `data/processed/hier_bigru/` — listos para `train-hier-bigru` y `train-hier-set`.
- Log completo guardado en `results/hier_bigru/prepare_hier_output.txt` (directorio eliminado al cambiar a 256 segmentos en sesión 3; log histórico no disponible).

### 2026-04-15

#### Implementación de HierSet — Encoder de conjunto para histogramas segmentados (Fase 6)
- Implementado modelo `AMRHierSet` en `src/models/hier_set/model.py`: trata los HIER_N_SEGMENTS segmentos como un conjunto. Pipeline inicial: LayerNorm(256) → Linear(256→128)+ReLU → attention pooling content-based → concat(ab_emb) → clasificador. (Arquitectura mejorada en sesión 3: cross-attention condicionada por antibiótico.)
- Implementada `HierSetDataset` en `src/models/hier_set/dataset.py`: subclase delgada de `HierBiGRUDataset`, reutiliza los mismos `.npy` de `hier_bigru/`.
- Añadido comando `train-hier-set` en `main.py`.
- 10 tests unitarios en `tests/models/test_hier_set.py`, incluyendo `test_permutation_invariance` (propiedad central) y `test_no_sequential_modules`.
- Añadido `test_hier_set_dataset` en `tests/models/test_datasets.py`.

#### Refactorización completa de MultiBiGRU — Encoder order-independent (Fase 4 revisada)
- Reemplazada la BiGRU por stream por un encoder sin dependencias secuenciales entre bins: `KmerStream` = LayerNorm(seq_len) + Linear(1→128,ReLU) + `bin_importance` (prior por identidad de bin) + attention pooling.
- `bin_importance`: parámetro aprendido por bin que asigna importancia a cada k-mero específico, sin crear dependencias entre bins adyacentes (a diferencia de la BiGRU).
- Fusión con `softmax` en lugar de `sigmoid` independiente: los tres streams compiten, garantizando que ninguno se apague completamente (evita el camino degenerado de ignorar el genoma).
- Añadido `weight_decay` (AdamW) al comando `train-multi-bigru`.
- Input del clasificador reducido de 433 a 177 dims (128 contexto + 49 embedding).
- Tests actualizados: `test_gate_sum_to_one`, `test_gate_varies_by_antibiotic`, `test_no_sequential_modules`, `test_bin_importance_is_per_bin_prior`. 13 tests en `test_multi_bigru.py`.
- Reentrenado en sesión 3 con nueva arquitectura: F1=0.8514, Recall=0.8925, AUC=0.8944.

#### Implementación de HierBiGRU (Fase 6)
- Implementado modelo `AMRHierBiGRU` en `src/models/hier_bigru/model.py`: BiGRU profunda (2 capas) + BahdanauAttention sobre HIER_N_SEGMENTS segmentos de histogramas (k=4, 256 dims).
- Implementada `HierBiGRUDataset` con validación de shape `(HIER_N_SEGMENTS, HIER_KMER_DIM)`.
- Añadidos comandos `prepare-hier` y `train-hier-bigru` en `main.py`.
- 8 tests unitarios en `tests/models/test_hier_bigru.py`.
- **Pendiente:** correr `prepare-hier` (genera `data/processed/hier_bigru/*.npy`, ~9060 genomas, proceso lento) y luego `train-hier-bigru`.

### 2026-04-10

#### Análisis de resultados y cierre de Fase 5 — Token BiGRU
- Documentado análisis completo de por qué el Token BiGRU no supera al MLP en F1: el subsampling uniforme (1 token cada ~1,100 bp) diluye la señal posicional de genes de resistencia (~800–2,000 bp), resultando en cobertura parcial (~0.1% del genoma).
- Hallazgo clave: los histogramas (censo completo de k-meros) superan a las secuencias de tokens (muestra dispersa) porque la "huella dactilar" de k-meros de los ARGs es distintiva incluso sin información posicional.
- Hallazgo positivo: Token BiGRU v2 logró Recall=0.9567, el más alto del proyecto — el modelo es extremadamente sensible a señales de resistencia.
- Evaluación de trabajo futuro: Transformers sparse (Longformer, BigBird) y modelos genómicos pre-entrenados (DNABERT-2, HyenaDNA, Nucleotide Transformer), con análisis de pros/contras y recomendación de Nucleotide Transformer con chunking como siguiente paso.
- Documentación actualizada en `PLAN_TOKEN_BIGRU.md` (secciones "Análisis de resultados y limitaciones" y "Trabajo futuro"), `5_experiments.md` (comparación final consolidada) y `PROGRESS.md`.

#### Finalización del Entrenamiento — Token BiGRU v2 (Fase 5.3)
- Ejecutado el entrenamiento final de la **Token BiGRU v2** con 2 capas, dropout recurrente y optimizador **AdamW**.
- Logrado un **Recall de 0.9567**, el valor más alto de todo el proyecto, superando el objetivo clínico del 90%.
- F1-Score final de 0.8121 y AUC-ROC de 0.8190.
- Los resultados confirman que la arquitectura profunda con regularización desacoplada (AdamW) y subsampling uniforme es altamente sensible a los determinantes genómicos de resistencia.
- Actualizada la documentación consolidada en `docs/5_experiments.md` y `docs/PROGRESS.md`.

#### Ajuste de hiperparámetros — Token BiGRU Iteración 2 (Fase 5.3)
- Reducido `lr` de 0.001 a **0.0005**: el entrenamiento base overfitteo en <13 épocas; lr más bajo da convergencia más gradual.
- Reducido `pos_weight_scale` de 2.5 a **1.6** (pos_weight efectivo ≈ 1.0, balance real): el 2.5 fue heredado del BiGRU sin justificación para este modelo. Recall=0.9066 en Iter. 1 confirma que no se necesita tanta presión hacia positivos.
- Añadido `weight_decay=1e-4` en Adam: regularización L2 para penalizar pesos grandes y reducir capacidad efectiva [Goodfellow16, Cap. 7].
- Añadido parámetro CLI `--weight-decay` en `train-token-bigru` (default 1e-4).
- Historial de hiperparámetros documentado en `docs/PLAN_TOKEN_BIGRU.md` (sección "Historial de hiperparámetros de entrenamiento").

#### Optimización de Arquitectura — Token BiGRU (Fase 5.2)
- Incrementada la profundidad de la BiGRU de 1 a **2 capas** (`GRU_NUM_LAYERS = 2`) para permitir el aprendizaje de representaciones jerárquicas más complejas.
- Activado el **dropout recurrente (0.3)** entre las capas de la BiGRU [Srivastava14] para mejorar la regularización y mitigar el sobreajuste observado en el entrenamiento base.
- Actualizada la documentación técnica y el diagrama Mermaid en `docs/4_models.md` para reflejar la estructura final.

#### Code review y correcciones — Token BiGRU (Fase 5.1 y 5.2)
- Code review completo del código del Token BiGRU; 8 issues encontrados y corregidos.
- **Bugs corregidos:**
  - `pipeline.py` (`_extract_and_save_tokens`): faltaba validación de `n_jobs`, variable `total` no definida en rama paralela, último genoma no logueado si no es múltiplo de 10.
  - `data_pipeline/__init__.py`: la función privada `_extract_and_save_tokens` se exportaba directamente; reemplazado por alias público `extract_and_save_tokens`.
  - `main.py`: import y call site actualizados al alias sin guión bajo.
  - `tests/models/test_token_bigru.py`: test `test_dropout_active_in_train` faltante (especificado en el plan, no implementado); `pytest.raises(Exception)` demasiado amplio → `pytest.raises(RuntimeError)`.
  - `tests/models/test_datasets.py`: faltaban aserciones de rango `[0, TOKEN_PAD_ID]` en `test_token_bigru_dataset`.
- 128 tests pasando tras correcciones.

#### Implementación de Token BiGRU (Fase 5.1 y 5.2)
- Implementado modelo `AMRTokenBiGRU` en `src/models/token_bigru/model.py` que procesa secuencias reales de k-meros (tokens) en lugar de histogramas; importa `BahdanauAttention` de `models.bigru.model`.
- Implementado `TokenBiGRUDataset` en `src/models/token_bigru/dataset.py`, subclase de `BaseAMRDataset`, carga tokens como `LongTensor` desde `data/processed/token_bigru/`.
- Agregada capacidad de tokenización a `KmerExtractor` mediante el método `to_token_sequence()` con **subsampling uniforme (linspace)** para garantizar cobertura genómica global [Haykin, Cap. 1.2].
- Incorporada capa de **embedding de k-meros** (vocabulario=257, dim=64) para mapear tokens discretos a vectores densos [Mikolov13].
- Reutilizado el mecanismo de **atención de Bahdanau** para interpretabilidad posicional [Bahdanau15].
- Añadidas constantes `TOKEN_KMER_K`, `TOKEN_VOCAB_SIZE`, `TOKEN_PAD_ID`, `TOKEN_MAX_LEN`, `TOKEN_EMBED_DIM` a `src/data_pipeline/constants.py`.
- Añadidos comandos CLI `prepare-tokens` (extracción paralela con `n_jobs`) y `train-token-bigru` (gradient clipping 1.0, pos_weight_scale 2.5).
- Añadidos 9 tests unitarios en `tests/models/test_token_bigru.py` y 1 test en `tests/models/test_datasets.py`.
- Documentada la nueva arquitectura en `docs/4_models.md` (Modelo D), `docs/5_experiments.md` (Experimento 4) y `docs/PROGRESS.md` (Fase 5).
- Plan detallado en `docs/PLAN_TOKEN_BIGRU.md`.

### 2026-04-09


#### Arquitectura Experta: Multi-Stream BiGRU (Fase 4.1 y 4.2)
- Ejecutado el entrenamiento del modelo **Multi-Stream BiGRU**, logrando un **AUC-ROC de 0.9038**, el más alto del proyecto.
- El modelo alcanzó un F1 de 0.8596 y un Recall de 0.8950, demostrando la eficacia de eliminar el padding en arquitecturas recurrentes.
- Implementado modelo `AMRMultiBiGRU` en `src/multi_bigru_model.py` con 3 streams independientes.
- Reducido el hidden size por stream a 64 unidades para mantener la eficiencia de parámetros (~233K totales).
- Actualizado `AMRDataset` para segmentar dinámicamente los vectores MLP en tiempo de carga, optimizando el uso de disco.
- Adaptado el training loop y el módulo de evaluación para soportar entradas genómicas estructuradas como tuplas de tensores.
- Añadido comando `train-multi-bigru` a `main.py`.
- Añadidos 6 tests unitarios específicos para validar la independencia de los flujos y las formas de los pesos de atención.

#### Éxito del Plan de Mejora 1 y Finalización de la Fase 3
- Ejecutado el entrenamiento de la **BiGRU V2** con `pos_weight_scale=2.5` y `DROPOUT=0.3`.
- Alcanzado un **Recall de 0.9032** en el conjunto de prueba, cumpliendo el objetivo clínico principal.
- Realizado el análisis de atención de la V2, revelando una focalización del 92% en k-meros cortos.
- Implementado el guardado automático de parámetros en `params.json` para garantizar la trazabilidad.
- Actualizada toda la documentación final (`PROGRESS.md`, `5_experiments.md`, `4_models.md`).

#### Implementación de BiGRU + Attention (Fase 3.1 y 3.2)
- Implementado modelo `AMRBiGRU` en `src/bigru_model.py` con atención aditiva de Bahdanau y 128 unidades ocultas, siguiendo fielmente a [Lugo21].
...
- Actualizado `AMRDataset` en `src/dataset.py` para soportar la carga de matrices 2D (`1024x3`) mediante el parámetro `model_type="bigru"`.
- Incorporado **Gradient Clipping** (`max_grad_norm=1.0`) en el ciclo de entrenamiento (`src/train/loop.py`) para estabilizar el BPTT sobre 1024 timesteps.
- Centralizada la constante `ANTIBIOTIC_EMBEDDING_DIM = 49` en `src/data_pipeline/constants.py`.
- Añadido comando `train-bigru` a `main.py` con soporte para entrenamiento profundo y evaluación final.
- Añadidos 8 tests unitarios para el modelo y 1 test de integración para el gradient clipping (46 tests totales pasando).
- Sincronizada toda la documentación técnica (`docs/1_environment.md`, `docs/3_data_pipeline.md`, `docs/4_models.md`, `docs/5_experiments.md`).

### 2026-03-23

#### Mejora del pipeline y actualización de resultados MLP
- Implementado filtro de frecuencia mínima de antibióticos (`< 20 registros`) en `LabelCleaner` para garantizar generalización y partición correcta (`src/data_pipeline/cleaning.py`).
- Añadido glosario de métricas clínicas y su interpretación biológica en `docs/5_experiments.md`.
- Actualizado el reporte del EDA (`docs/2_eda.md`) con las estadísticas reales del dataset completo (16,571 genomas, 219 descartados por longitud).
- Documentada la justificación de la arquitectura (capas 512, 128) basándose en Haykin y eficiencia computacional en `docs/4_models.md`.
- Re-entrenado el modelo MLP con el dataset filtrado (82,192 registros, 90 antibióticos conservados).
- Métricas actualizadas en `docs/REPORT_SHALLOW_NN.md`: **F1 = 0.8600, Recall = 0.9165, AUC-ROC = 0.9035**.
- Renombrado parámetro CLI `--top-n` a `--top-n-antibiotics` en `main.py` para mayor claridad.
- Creado script `src/plot_roc.py` para generar la curva ROC del modelo.

### 2026-03-22

#### Entrenamiento y evaluación del MLP (Fase 2.3)
- Entrenado MLP sobre dataset completo: 57,088 train / 12,613 val / 12,519 test (782,804 parámetros)
- Early stopping en época 75 (patience=20), mejor modelo guardado en época 71 (val F1=0.8392)
- Métricas en test set: F1=0.8616, Recall=0.9031, AUC-ROC=0.9098 — criterio de éxito cumplido
- Resultados en `results/mlp/`: `best_model.pt`, `metrics.json`, `history.csv`, `history.png`, `OUTPUT.txt`
- Renombrado `OUTPUT.md` → `OUTPUT.txt` para evitar renderizado incorrecto en GitHub

### 2026-03-21

#### Implementación de `src/train.py` (Fase 2.2)
- `set_seed()`: fija semillas en random, numpy, torch, CUDA y cuDNN para reproducibilidad
- `detect_device()`: detección automática CUDA → MPS → CPU
- `train_epoch()`: una pasada forward+backward por el train set, retorna loss promedio
- `evaluate()`: métricas completas (loss, accuracy, precision, recall, F1, AUC-ROC) + búsqueda de umbral óptimo por máximo F1
- `train()`: orquestador con AdamW, early stopping y checkpoint sobre val_F1 (patience), ReduceLROnPlateau sobre val_F1, genera `best_model.pt`, `metrics.json`, `history.csv`, `history.png`
- Comando `train-mlp` agregado en `main.py` con opciones: `--data-dir`, `--output-dir`, `--epochs`, `--batch-size`, `--lr`, `--patience`
- 13 tests nuevos en `tests/test_train.py` (93 en total pasando)
- Docstrings con referencias a Haykin (2009): retropropagación §4.4, mini-batches §4.3, early stopping §4.13, clasificador de Bayes §1.4

#### Actualización de documentación
- `AGENTS.md`: tabla de archivos clave actualizada, nueva sección "Documentos a mantener al hacer cambios"
- `README.md`: árbol corregido (`data_pipeline/`, `dataset.py`, `mlp_model.py`, `tests/data_pipeline/`)
- `src/README.md`: árbol y secciones de módulos actualizados
- `docs/PROGRESS.md`: fases reestructuradas para reflejar entregas (Fase 2: Shallow NN, Fase 3: Deep NN)

#### Reorganización de documentación
- Separado el registro histórico a un nuevo archivo `docs/CHANGELOG.md` fuera de `docs/PROGRESS.md`.
- Actualizadas referencias en `AGENTS.md`.
- Renombrado `docs/6_implementation_plan.md` a `docs/PLAN_MLP.md` para reflejar su alcance específico y alinearse con la convención de nombres.

#### Implementación del ciclo de entrenamiento (Fase 2.2)
- Implementado paquete `src/train/` con `train_epoch`, `evaluate`, `set_seed` y el orquestador principal con early stopping.
- Incorporado cálculo de umbral óptimo (basado en maximizar F1-score) en tiempo de validación.
- Agregado comando `train-mlp` en `main.py` (Typer CLI).
- Añadidos tests para el flujo de entrenamiento en `tests/test_train.py`.

### 2026-03-20

#### Plan de implementación MLP
- Creado `docs/6_implementation_plan.md` con plan detallado para Fase 2 y 3: `AMRDataset`, `AMRMLP`, loop de entrenamiento, CLI, tests y decisiones de diseño consolidadas
- Decisiones clave registradas: pre-carga de vectores en RAM, device agnostic en training loop, `pos_weight` dinámico desde `train_stats.json`, threshold óptimo en `evaluate()`, estructura de outputs en `results/mlp/`

#### Paralelización de extracción de k-meros
- `src/data_pipeline/pipeline.py`: nueva función top-level `_extract_single_genome` (picklable); `_extract_kmers` acepta `n_jobs` con validación; `n_jobs=-1` usa 80% de CPUs (`max(1, int(cpu_count * 0.8))`); modo paralelo usa `as_completed` para logging en tiempo real desde el proceso principal
- `run_pipeline()`: propaga `n_jobs`
- `main.py`: opción `--n-jobs` en `prepare-data`
- `docs/3_data_pipeline.md`: ejemplos `--n-jobs -1` y `--n-jobs 4`
- 3 tests nuevos en `test_pipeline.py`: unit del helper, equivalencia secuencial vs paralelo (bit-idéntico), smoke test `n_jobs=-1`

#### Limpieza de magic numbers en tests
- `tests/data_pipeline/test_pipeline.py`: constantes `_N_GENOMES`, `_ACGT_PATTERN`, imports de `MIN_GENOME_LENGTH`, `TOTAL_KMER_DIM`, `BIGRU_PAD_DIM`, `KMER_SIZES`
- `tests/data_pipeline/test_features.py`: imports de `KMER_DIMS`, `KMER_SIZES`, `TRAIN_RATIO`; `64`/`256`/`1023` → `KMER_DIMS[0]`/`KMER_DIMS[1]`/`BIGRU_PAD_DIM-1`; `0.70` → `TRAIN_RATIO`; `(BIGRU_PAD_DIM, 3)` → `(BIGRU_PAD_DIM, len(KMER_SIZES))`
- `tests/data_pipeline/test_cleaning.py`: `"ACGT" * 125_000` y `500_000` → `MIN_GENOME_LENGTH`
- `tests/bvbrc/test_http.py`: `timeout=120` → `REQUEST_TIMEOUT`
- `src/bvbrc/_http.py`: extraído `REQUEST_TIMEOUT = 120` como constante exportada

#### Documentación: comandos CLI del data pipeline
- `docs/3_data_pipeline.md`: agregada sección `#### Comandos CLI` bajo el paso 2 con `export-contradictions-cmd`
- `docs/3_data_pipeline.md`: agregada sección "Ejecutar el pipeline completo" con `prepare-data`, incluyendo variante de rutas por defecto, rutas personalizadas y lista de outputs generados

### 2026-03-19

#### Mejora al data pipeline: registro de genomas descartados
- `GenomeFilter` ahora expone propiedades `missing` y `short` para acceder a los genome_id descartados
- `_filter_genomes()` en `pipeline.py` guarda `discarded_genomes.csv` en el directorio de salida (columnas: `genome_id`, `reason`)
- Razones registradas: `missing_fasta` (sin archivo FASTA en disco), `below_min_length` (genoma < 0.5 Mb)
- Pendiente: persistir también los fallos de descarga desde `GenomeBatchFetcher` en `src/bvbrc/genomes.py`

### 2026-03-18

#### Code review y bugfixes
- Code review completo del proyecto contra PEP 8 y AGENTS.md (26 issues encontrados)
- **Bugs corregidos:**
  - `eda.py`: eliminado `import numpy` redundante en `_print_baseline_benchmark`
  - `eda.py`: `genome_id` se comparaba como `float` en análisis genómico — corregido a comparación como `str`
  - `eda.py` y `main.py`: agregado `dtype={"genome_id": str}` a todos los `read_csv` (consistencia con `cleaning.py`)
  - `genomes.py`: `except Exception` reducido a `except (RuntimeError, OSError)` para no ocultar bugs
- **Mejoras de robustez:**
  - `features.py`: `zip(strict=True)` en `KmerExtractor.extract()`
  - `features.py`: guard en `to_mlp_vector()` si `extract()` no fue llamado
  - `cleaning.py`: documentado que `GenomeFilter.filter()` es single-use
  - `main.py`: variable `taxon_id` no usada reemplazada por `_`
- **Tests:** mock de `time.sleep` en tests de retries HTTP (~10s → instantáneo)
- **Docstrings con referencias bibliográficas:**
  - `normalize_features()` → Haykin (2009) §4.6 + LeCun et al. (1998) §4.3
  - `split_genomes()` → Haykin (2009) §4.13
- **Convención de idioma actualizada:** docstrings y comentarios ahora en español
- Renombrado `_split_and_log` → `_split_genomes` en `pipeline.py`
- 63 tests pasando

### 2026-03-17

#### Refactor `src/data_pipeline.py` → paquete `src/data_pipeline/`
- Eliminado `data_pipeline.py` y reemplazado por paquete con 4 módulos:
  - `constants.py` — constantes compartidas
  - `cleaning.py` — `LabelCleaner` + `GenomeFilter`
  - `features.py` — `KmerExtractor` + funciones de features + `split_genomes`
  - `pipeline.py` — funciones privadas de orquestación + `run_pipeline`
- `__init__.py` re-exporta toda la API pública — `main.py` sin cambios
- 31 tests nuevos en `tests/data_pipeline/` (63 en total pasando)
- Bugs corregidos durante la implementación:
  - `genome_id` se parseaba como float (`1.10` → `1.1`) — fix: `dtype={"genome_id": str}`
  - Log messages en español — traducidos a inglés
- Renombrado `contradictory_index` → `contradictory_indices` en `eda.py` y `data_pipeline/cleaning.py`
- Agregado docstring a `KmerExtractor._count_kmers` con referencia bibliográfica (Compeau & Pevzner, 2014)

### 2026-03-16 (continued)

#### Refactor Task 6 — Wire up `data_pipeline/__init__.py`
- Completado refactoring del paquete `src/data_pipeline/`: `__init__.py` implementado con re-exports completos de todas las clases, funciones y constantes públicas
- Agregado `__all__` exhaustivo para garantizar que IDE y documentación reflejen API pública
- Verificado: `from data_pipeline import run_pipeline` funciona sin necesidad de specificar módulos internos
- Full test suite pasa: 63 tests en `tests/bvbrc/` (32) + `tests/data_pipeline/` (31)
- Refactor de 6 tareas completado exitosamente

### 2026-03-16

#### Implementación `src/data_pipeline.py`
- Implementado pipeline completo de preprocesamiento en `src/data_pipeline.py`
- Clases: `LabelCleaner` (limpieza de etiquetas), `GenomeFilter` (filtro por longitud), `KmerExtractor` (histogramas con rolling hash O(n))
- Funciones: `split_genomes()`, `normalize_features()`, `mlp_vector_to_bigru_matrix()`, `run_pipeline()`
- Pipeline orquestador refactorizado en 6 funciones privadas (`_clean_labels`, `_filter_genomes`, `_save_antibiotic_index`, `_split_and_log`, `_extract_kmers`, `_normalize_and_save`)
- Agregado comando `prepare-data` a `main.py`
- Ejecutado exitosamente sobre 136 genomas: 162,170 → 1,150 registros; split 95/20/21; pos_weight=0.9590 (subset actual)
- Outputs generados: `cleaned_labels.csv`, `antibiotic_index.csv` (66 antibióticos), `splits.csv`, 136 × `mlp/*.npy` (1344-dim), 136 × `bigru/*.npy` ([1024,3])
- Actualizado `.gitignore`: agregados patrones para `cleaned_labels.csv`, `antibiotic_index.csv`, `splits.csv`, `*.npy`

### 2026-03-15

#### Reorganización de documentación
- Reestructurada carpeta `docs/`: eliminada subcarpeta `implementation/`, archivos numerados directamente en `docs/`
- Renombrados docs para reflejar orden lógico: EDA pasó de `5_` a `2_`, resto renumerado (3, 4, 5)
- `PROGRESS.md` movido a `docs/`; `CHANGELOG.md` fusionado dentro de `PROGRESS.md`
- `docs/usage.md` creado y luego fusionado en `docs/1_environment.md`; `docs/reference/` creado para `bvbrc_api.md`
- `README.md` simplificado: secciones `## Uso` y `## Documentación` reemplazadas por punteros a `docs/`
- `*.egg-info/` agregado a `.gitignore` y removido del tracking de git

#### Mejoras al CLI y docs
- Agregado `help=` explícito a todos los `@app.command()` en `main.py`
- Agregados comentarios descriptivos a cada variante de comando en `docs/1_environment.md` y `docs/3_data_pipeline.md`
- Comandos CLI movidos a los docs de su fase correspondiente
- `docs/2_eda.md` reescrito con hilo narrativo: hallazgos → implicaciones → decisiones (eliminado information dump)

#### Refactoring `src/bvbrc/`
- `amr.py`: `fetch_amr_labels()` refactorizado a clase `AMRFetcher` con variables de instancia para estado de paginación (`_records`, `_offset`, `_total`); `_build_query` como `@staticmethod`; `_fetch_next_page`, `_update_total`, `_is_complete`, `_page_url` como métodos separados
- `genomes.py`: refactorizado a `GenomeFetcher` (descarga individual) y `GenomeBatchFetcher` (descarga en lote); funciones públicas como thin wrappers
- 32 tests actualizados y pasando

#### Datos y EDA
- Descargados 138 genomas a `data/raw/fasta` (30/especie, 12 fallaron por ausencia de secuencia en BV-BRC)
- Detectados 2 genomas casi vacíos: `1352.11605` (3.6 kb) y `1352.11302` (16.9 kb) — *E. faecium*; pendiente filtro en pipeline
- Implementado `export_contradictions()` en `src/eda.py` y comando `export-contradictions-cmd` en `main.py`
- Generado `data/processed/contradictory_labels.csv`: 488 pares contradictorios, 1392 filas; contradicciones correlacionan con diferencia de método de laboratorio (ej. Broth dilution vs Disk diffusion)

---

### 2026-03-04

#### EDA — análisis genómico
- Descargada muestra de 89 genomas (20/especie estratificados por fenotipo) en `data/raw/fasta_sample/`
- Agregada función `_print_genome_analysis()` a `src/eda.py`: longitud, contigs, GC content, bases N, alertas por calidad
- Actualizado comando `eda` en `main.py` con `--genomes-dir` opcional
- Hallazgos: longitud media 4.4 Mb, GC muy variable por especie (32%-66%), sin genomas cortos ni con alto contenido N; 4 genomas muy fragmentados (*E. faecium*)

#### EDA — entregables completados
- Agregadas dos nuevas secciones a `src/eda.py`:
  - `_print_outliers()`: genomas con registros extremos (>mean+3σ), antibióticos con desbalance ≥90%, etiquetas contradictorias
  - `_print_baseline_benchmark()`: majority class global (54.0%) y por antibiótico (F1=0.7366) — piso mínimo para los modelos
- Actualizado `docs/2_eda.md` con entregables faltantes: data dictionary, leakage/confound analysis, outliers, baseline benchmark

---

### 2026-03-03

#### Implementación
- Agregado `typer` como dependencia
- Configurado `pyproject.toml` con `build-system` (setuptools) para que `src/` sea instalable sin `sys.path` hacks
- Implementado `main.py` como CLI con tres comandos: `download-amr`, `download-genomes`, `eda`

#### EDA — hallazgos del dataset ESKAPE
- **162,170 registros**, 16,204 genomas únicos, **96 antibióticos** distintos
- Solo 5 de 6 especies ESKAPE presentes (falta *Enterobacter spp.*)
- Balance global: 54% Resistant / 46% Susceptible → `pos_weight = 0.8522`
- 10,383 registros duplicados (mismo genome_id + antibiotic) — a resolver en pipeline
- **Dim embedding antibiótico: 49** `[min(50, (96 // 2) + 1)]`

#### Implementación `src/bvbrc/`
- Refactorizado `src/bvbrc_client.py` → paquete `src/bvbrc/` con `_http.py`, `amr.py`, `genomes.py`
- Creados 32 unit tests en `tests/bvbrc/` (todos pasando)
- Creado `docs/reference/bvbrc_api.md` con referencia completa de la API REST de BV-BRC

---

### 2026-02-28

#### Proyecto
- Inicializado proyecto con `uv init`, creada estructura de carpetas
- Agregadas dependencias principales
- Revisión adversarial de docs de diseño: decisiones de organismos, antibióticos, BiRNN variantes, formato de almacenamiento, split por `genome_id`, reemplazo de SMOTE por `pos_weight`
