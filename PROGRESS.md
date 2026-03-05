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
- [x] Correr EDA sobre dataset ESKAPE completo — ver hallazgos en `docs/implementation/5_eda.md`
- [x] Documentar decisiones derivadas del EDA (ver sección "Decisiones pendientes" abajo)
- [x] Completar entregables EDA: data dictionary, leakage/confound analysis, outliers, baseline benchmark (F1=0.7366)
- [x] EDA genómico: análisis de 89 genomas (muestra 20/especie) — longitud ~4.4 Mb, GC variable por especie, calidad buena

### 1.1 Descarga de datos (`src/bvbrc/`)
- [x] `fetch_amr_labels()` implementado y validado: paginación correcta, campos esperados, fenotipos solo R/S
- [x] `download_genome_fasta()` implementado y validado: FASTA descargado y parseable con biopython
- [x] 32 unit tests pasando (`tests/bvbrc/`)

### 1.2 Preprocesamiento de etiquetas (`src/data_pipeline.py`)
- [ ] Eliminar duplicados (genome_id + antibiotic) — estrategia: conservar primer registro
- [ ] Guardar triples limpios `(genome_id, antibiotic, label)` en CSV

### 1.3 Extracción de k-meros (`src/data_pipeline.py`)
- [ ] MLP: histogramas k=3,4,5 concatenados → vector 1344-dim, normalizado (media 0, var 1)
- [ ] BiRNN Variante A: mismos histogramas paddeados a 1024 y apilados → matriz `[1024, 3]`
- [ ] Guardar vectores/matrices como archivos `.npy`

### 1.4 Manejo de desbalance y splits (`src/data_pipeline.py`)
- [ ] Calcular `pos_weight` para `BCEWithLogitsLoss` → ya conocido: **0.8522** (del EDA)
- [ ] Split estratificado 70/15/15 por `genome_id`

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
- [ ] Estrategia para duplicados en pipeline → pendiente confirmar (propuesta: conservar primer registro)
- [ ] *Enterobacter spp.* ausente del dataset — investigar si taxon_id=547 captura bien los datos
