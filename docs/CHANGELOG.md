# CHANGELOG

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
- `train()`: orquestador con Adam, early stopping sobre val loss (patience), checkpoint por mejor val F1, genera `best_model.pt`, `metrics.json`, `history.csv`, `history.png`
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
