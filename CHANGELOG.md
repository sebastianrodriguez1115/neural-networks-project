# Changelog

## 2026-03-04

### EDA — análisis genómico
- Descargada muestra de 89 genomas (20/especie estratificados por fenotipo) en `data/raw/fasta_sample/`
- Agregada función `_print_genome_analysis()` a `src/eda.py`: longitud, contigs, GC content, bases N, alertas por calidad
- Actualizado comando `eda` en `main.py` con `--genomes-dir` opcional
- Hallazgos: longitud media 4.4 Mb, GC muy variable por especie (32%-66%), sin genomas cortos ni con alto contenido N; 4 genomas muy fragmentados (*E. faecium*)
- Documentados hallazgos en `docs/implementation/5_eda.md`

### EDA — entregables completados
- Agregadas dos nuevas secciones a `src/eda.py`:
  - `_print_outliers()`: genomas con registros extremos (>mean+3σ), antibióticos con desbalance ≥90%, etiquetas contradictorias
  - `_print_baseline_benchmark()`: majority class global (54.0%) y por antibiótico (F1=0.7366) — piso mínimo para los modelos
- Actualizado `docs/implementation/5_eda.md` con entregables faltantes:
  - **Data dictionary**: descripción de columnas, tipos, rangos y nulos del CSV de etiquetas
  - **Leakage/confound analysis**: features descartadas, confounds identificados (desbalance por especie/antibiótico, etiquetas contradictorias)
  - **Outliers**: 200 genomas con registros extremos, 27 antibióticos con desbalance ≥90%, 488 etiquetas contradictorias
  - **Baseline benchmark**: F1=0.7366 — los modelos deben superar este valor usando información genómica

---

## 2026-03-03 (continuación)

### Implementación
- Agregado `typer` como dependencia
- Configurado `pyproject.toml` con `build-system` (setuptools) para que `src/` sea instalable sin `sys.path` hacks
- Implementado `main.py` como CLI con tres comandos:
  - `download-amr` — descarga etiquetas AMR de BV-BRC
  - `download-genomes` — descarga genomas FASTA a partir del CSV de etiquetas
  - `eda` — análisis exploratorio con resumen general, distribución por especie, balance de clases, top antibióticos y calidad de datos

### EDA — hallazgos del dataset ESKAPE
- **162,170 registros**, 16,204 genomas únicos, **96 antibióticos** distintos
- Solo 5 de 6 especies ESKAPE presentes (falta *Enterobacter spp.*)
- Balance global: 54% Resistant / 46% Susceptible → `pos_weight = 0.8522`
- Desbalance severo por especie: *A. baumannii* (80% R), *S. aureus* (28% R)
- 10,383 registros duplicados (mismo genome_id + antibiotic) — a resolver en pipeline
- 14.4% de registros sin `laboratory_typing_method`
- **Dim embedding antibiótico: 49** `[min(50, (96 // 2) + 1)]`

---

## 2026-03-03

### Documentación
- Creado `CLAUDE.md` con instrucciones para agentes IA (stack, convenciones, archivos clave)
- Creado `PROGRESS.md` con plan de implementación por fases y estado de cada tarea
- Creado `docs/implementation/bvbrc_api.md` con referencia completa de la API REST de BV-BRC (endpoints, RQL, taxon IDs ESKAPE, paginación, recomendaciones de implementación)

### Implementación
- Refactorizado `src/bvbrc_client.py` → paquete `src/bvbrc/` con tres módulos:
  - `_http.py` — constantes y `make_api_request_with_retries` compartidos
  - `amr.py` — `fetch_amr_labels`
  - `genomes.py` — `download_genome_fasta`, `download_multiple_genomes_fasta`
- Agregado `pytest` como dependencia de desarrollo
- Creados 32 unit tests en `tests/bvbrc/` (todos pasando):
  - `test_http.py` — 8 tests para utilidades HTTP y reintentos
  - `test_amr.py` — 13 tests para construcción de queries RQL y descarga paginada
  - `test_genomes.py` — 11 tests para descarga FASTA, omisión de duplicados y manejo de errores

- Implementado `src/bvbrc/` (antes `src/bvbrc_client.py`):
  - `fetch_amr_labels()` — descarga etiquetas AMR con paginación automática (genome_amr endpoint)
  - `download_genome_fasta()` — descarga FASTA de un genoma individual (genome_sequence endpoint)
  - `download_multiple_genomes_fasta()` — descarga en lote con manejo de errores y omisión de archivos ya descargados

---

## 2026-02-28

### Proyecto
- Movidas carpetas `implementation/` y `proposal/` dentro de `docs/`
- Inicializado proyecto con `uv init`
- Creada estructura de carpetas: `src/`, `data/raw/`, `data/processed/`, `notebooks/`, `results/`
- Creados archivos vacíos: `src/bvbrc_client.py`, `src/data_pipeline.py`, `src/models.py`, `src/train.py`
- Agregadas dependencias: `requests`, `pandas`, `numpy`, `scikit-learn`, `torch`, `biopython`, `matplotlib`, `seaborn`
- Actualizado `uv` de `0.8.5` a `0.10.7`

### Revisión adversarial — issues corregidos

### implementation/2_data_pipeline.md
- Corregido typo: `BV-BCR` → `BV-BRC`
- Decisión: organismos → ESKAPE completo, modelo único para todas las especies
- Decisión: antibióticos → todos los disponibles con evidencia de laboratorio (`Laboratory Method`); antibiótico como feature del modelo
- Decisión: BiRNN → dos variantes; Variante A (artículo de referencia, matriz 3×1024) con prioridad; Variante B (secuencia ordenada) como extensión futura
- Decisión: formato de almacenamiento → etiquetas en CSV, k-meros en `.npy`
- Paso 1 actualizado: conservar todos los antibióticos con evidencia de laboratorio, no filtrar por antibiótico específico
- Paso 2 actualizado: resultado son triples `(genome_id, antibiotic, label)`, no pares
- Split actualizado: por `genome_id` para evitar data leakage
- Manejo de desbalance: reemplazado SMOTE por class weights (`pos_weight` en `BCEWithLogitsLoss`)

### implementation/3_models.md
- Arquitecturas actualizadas para reflejar decisiones del doc 2 (Variante A/B, antibiótico como feature)
- Decisión: GRU (vs LSTM) → GRU, resultados equivalentes pero más simple
- Decisión: 1 capa BiGRU (artículo de referencia)
- Decisión: atención global aditiva / Bahdanau (artículo de referencia)
- Decisión: hidden size RNN → 128 (artículo de referencia)
- Orientación matriz corregida: `[batch, 1024, 3]` (1024 timesteps × 3 features)
- Embedding dim k-meros (100) movido a Variante B donde aplica; marcado como TBD
- Pendiente: dimensión embedding antibiótico → regla empírica `min(50, (num_antibióticos // 2) + 1)` una vez se conozca el número de antibióticos en BV-BRC

### implementation/4_experiments.md
- Experimento 2 actualizado: BiRNN Variante A (matriz 3×1024)
- Agregado Experimento 3: BiRNN Variante B (opcional)
- Decisión: métrica principal → F1; AUC-ROC como secundaria
- Decisión: max 100 epochs, patience 10; nota pendiente sobre costo de calcular F1 por epoch
- Decisión: k fijo en k=3,4,5
- Comparación final actualizada: "todos los modelos implementados"
- Agregada semilla aleatoria fija: `random_seed = 42`
