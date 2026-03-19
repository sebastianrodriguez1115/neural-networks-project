# PROGRESS.md — Plan de implementación y avance

Estado: `[ ]` pendiente · `[~]` en progreso · `[x]` completado

---

## Fase 0 — Entorno y estructura

- [x] Inicializar proyecto con `uv init`
- [x] Crear estructura de carpetas (`src/`, `data/raw/`, `data/processed/`, `notebooks/`, `results/`)
- [x] Agregar dependencias (`torch`, `biopython`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `typer`)
- [x] Configurar `pyproject.toml` con build-system (setuptools) y pytest

---

## Fase 1 — Data pipeline

### 1.0 EDA (requisito de la clase)
- [x] Implementar `src/eda.py` con análisis exploratorio del CSV de etiquetas
- [x] Correr EDA sobre dataset ESKAPE completo — ver hallazgos en `docs/2_eda.md`
- [x] Documentar decisiones derivadas del EDA (ver sección "Decisiones pendientes" abajo)
- [x] Completar entregables EDA: data dictionary, leakage/confound analysis, outliers, baseline benchmark (F1=0.7366)
- [x] EDA genómico: análisis de 138 genomas (muestra 30/especie) — longitud ~4.4 Mb, GC variable por especie, 2 genomas cortos detectados
- [x] Exportar pares con etiquetas contradictorias: `export-contradictions-cmd` → `data/processed/contradictory_labels.csv` (488 pares, 1392 filas)

### 1.1 Descarga de datos (`src/bvbrc/`)
- [x] `fetch_amr_labels()` implementado y validado: paginación correcta, campos esperados, fenotipos solo R/S
- [x] `download_genome_fasta()` implementado y validado: FASTA descargado y parseable con biopython
- [x] 32 unit tests pasando (`tests/bvbrc/`)

### 1.2 Preprocesamiento de etiquetas (`src/data_pipeline/`)
- [x] Descartar pares contradictorios (488 pares con R y S para mismo genome_id + antibiotic)
- [x] Eliminar duplicados consistentes (genome_id + antibiotic) — conservar primer registro
- [x] Filtrar genomas < 0.5 Mb (2 genomas casi vacíos de *E. faecium*)
- [x] Guardar triples limpios `(genome_id, antibiotic, label)` en CSV

### 1.3 Extracción de k-meros (`src/data_pipeline/`)
- [x] MLP: histogramas k=3,4,5 concatenados → vector 1344-dim, normalizado (media 0, var 1)
- [x] BiRNN Variante A: mismos histogramas paddeados a 1024 y apilados → matriz `[1024, 3]`
- [x] Guardar vectores/matrices como archivos `.npy`

### 1.4 Manejo de desbalance y splits (`src/data_pipeline/`)
- [x] Calcular `pos_weight` para `BCEWithLogitsLoss` — recalculado sobre train set
- [x] Split estratificado 70/15/15 por `genome_id`
- [x] Normalización con estadísticas solo del train set (evitar leakage)

---

## Fase 2 — Modelos (`src/models.py`)

- [ ] **MLP:** input genómico (1344) + antibiotic embedding (dim=49) → Dense(512, ReLU) + Dropout → Dense(128, ReLU) + Dropout → Dense(1)
- [ ] **BiGRU + Attention (Var A):** input `[batch, 1024, 3]` → BiGRU(hidden=128) → Bahdanau attention → context `[batch, 256]` + antibiotic embedding → Dense → Dense(1)

---

## Fase 3 — Entrenamiento (`src/train.py`)

- [ ] Loop de entrenamiento con Adam (lr=0.001), batch size=32, max 100 epochs
- [ ] `BCEWithLogitsLoss` con `pos_weight=0.8522`
- [ ] Early stopping: patience=10 sobre pérdida de validación
- [ ] Registro de métricas por epoch: loss train/val, accuracy, precision, recall, F1, AUC-ROC
- [ ] Guardar mejor checkpoint (mejor F1 en validación)

---

## Fase 4 — Experimentos

- [ ] **Experimento 1:** Entrenar y evaluar MLP baseline
- [ ] **Experimento 2:** Entrenar y evaluar BiGRU + Attention (Var A)
- [ ] Comparación final: métricas en test, matrices de confusión, análisis de pesos de atención
- [ ] Criterio de éxito: F1 ≥ 0.85 y recall ≥ 0.90 en clase resistente

---

## Decisiones

### Resueltas
- [x] Dim embedding antibiótico → **49** `[min(50, (96 // 2) + 1)]` (96 antibióticos en dataset)
- [x] pos_weight → **0.8522** (Susceptible/Resistant, del EDA)
- [x] Early stopping → pérdida de validación, patience=10

### Pendientes
- [ ] *Enterobacter spp.* ausente del dataset — investigar si taxon_id=547 captura bien los datos

### Resueltas (2026-03-15)
- [x] Estrategia para duplicados → descartar pares contradictorios (488), conservar primer registro de duplicados consistentes
- [x] Filtro de longitud mínima → descartar genomas < 0.5 Mb (`1352.11605`: 3.6 kb, `1352.11302`: 16.9 kb)
- [x] Normalización de k-meros → estadísticas calculadas solo sobre train set, aplicadas a val/test (evitar leakage)

---

## Changelog

### 2026-03-18

#### Code review y bugfixes
- Code review completo del proyecto contra PEP 8 y CLAUDE.md (26 issues encontrados)
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
