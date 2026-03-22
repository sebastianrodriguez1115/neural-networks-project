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

## Fase 2 — Shallow NN: MLP (Entrega 2)

### 2.1 Dataset y modelo
- [x] `AMRDataset` (`src/dataset.py`): pre-carga vectores `.npy` en RAM, devuelve `(genome_vector, antibiotic_idx, label)`, expone `load_pos_weight()` desde `train_stats.json`
- [x] `AMRMLP` (`src/mlp_model.py`): input genómico (1344) + antibiotic embedding (dim=49) → Dense(512, ReLU) + Dropout(0.3) → Dense(128, ReLU) + Dropout(0.3) → Dense(1); factory method `from_antibiotic_index()`

### 2.2 Entrenamiento (`src/train/`)
- [x] `set_seed(42)` para reproducibilidad
- [x] `train_epoch` (o similar) → loss promedio
- [x] `evaluate(model, loader, criterion, device)` → loss, accuracy, precision, recall, F1, AUC-ROC + threshold óptimo
- [x] `train(...)`: Adam lr=0.001, batch size=32, max 100 epochs, `BCEWithLogitsLoss` con `pos_weight` dinámico desde `train_stats.json`, early stopping patience=10 sobre val loss, checkpoint por mejor val F1
- [x] Outputs en `results/mlp/`: `best_model.pt`, `metrics.json`, `history.csv`, `history.png`
- [x] Comando `train-mlp` en `main.py`
- [x] Tests unitarios (`tests/test_train.py` y `tests/test_mlp.py`)

### 2.3 Experimento y reporte
- [ ] Entrenar y evaluar MLP sobre dataset completo
- [ ] Métricas en test set: accuracy, precision, recall, F1, AUC-ROC, matriz de confusión
- [ ] Criterio de éxito: F1 ≥ 0.85 y recall ≥ 0.90 en clase resistente
- [ ] Reporte de entrega (shallow NN)

---

## Fase 3 — Deep NN: BiGRU + Attention (Entrega final)

### 3.1 Modelo
- [ ] `AMRBiGRU` (`src/bigru_model.py`): input `[batch, 1024, 3]` → BiGRU(hidden=128) → Bahdanau attention → context `[batch, 256]` + antibiotic embedding → Dense → Dense(1)

### 3.2 Entrenamiento
- [ ] Adaptar `AMRDataset` para cargar matrices BiGRU (`bigru/*.npy`)
- [ ] Comando `train-bigru` en `main.py`

### 3.3 Comparación y reporte final
- [ ] Entrenar y evaluar BiGRU + Attention sobre dataset completo
- [ ] Comparación MLP vs BiGRU: métricas en test, matrices de confusión, análisis de pesos de atención
- [ ] Criterio de éxito: F1 ≥ 0.85 y recall ≥ 0.90 en clase resistente
- [ ] Reporte final consolidado

---

## Decisiones

### Resueltas
- [x] Dim embedding antibiótico → **49** `[min(50, (96 // 2) + 1)]` (96 antibióticos en dataset)
- [x] pos_weight → **0.8522** (Susceptible/Resistant, del EDA)
- [x] Early stopping → pérdida de validación, patience=10

### Pendientes
- [ ] *Enterobacter spp.* ausente del dataset — investigar si taxon_id=547 captura bien los datos
