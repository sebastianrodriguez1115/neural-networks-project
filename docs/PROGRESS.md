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
- [x] Filtrar antibióticos con baja frecuencia (< 20 registros) para asegurar generalización
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
- [x] `train(...)`: AdamW lr=0.001, batch size=32, max 100 epochs, `BCEWithLogitsLoss` con `pos_weight` dinámico desde `train_stats.json`, early stopping y checkpoint sobre val_F1 (patience configurable), ReduceLROnPlateau sobre val_F1
- [x] Outputs en `results/mlp/`: `best_model.pt`, `metrics.json`, `history.csv`, `history.png`
- [x] Comando `train-mlp` en `main.py`
- [x] Tests unitarios (`tests/test_train.py` y `tests/test_mlp.py`)

### 2.3 Experimento y reporte
- [x] Entrenar y evaluar MLP sobre dataset completo (57036 train / 12628 val / 12528 test, 43 épocas, early stopping)
- [x] Métricas en test set: accuracy=0.8157, precision=0.8100, recall=0.9165, F1=0.8600, AUC-ROC=0.9035
- [x] Criterio de éxito: F1=0.8600 ≥ 0.85 ✓ y recall=0.9165 ≥ 0.90 ✓
- [x] Reporte de entrega (shallow NN)

---

## Fase 3 — Deep NN: BiGRU + Attention (Entrega final)

### 3.1 Modelo
- [x] `AMRBiGRU` (`src/bigru_model.py`): input `[batch, 1024, 3]` → BiGRU(hidden=128) → Bahdanau attention → context `[batch, 256]` + antibiotic embedding → Dense → Dense(1)
- [x] Soporte para Gradient Clipping [Pascanu13] en el loop de entrenamiento.
- [x] Optimización del modelo (V2): Ajuste de Dropout (0.3) y función de pérdida asimétrica.

### 3.2 Entrenamiento
- [x] Adaptar `AMRDataset` para cargar matrices BiGRU (`bigru/*.npy`)
- [x] Comando `train-bigru` en `main.py`
- [x] 46 unit tests pasando (`tests/test_dataset.py`, `tests/test_bigru.py`, `tests/test_train.py`)

### 3.3 Comparación y reporte final
- [x] Entrenar y evaluar BiGRU + Attention sobre dataset completo
- [x] Comparación MLP vs BiGRU: métricas en test, matrices de confusión, análisis de pesos de atención
- [x] Criterio de éxito: F1 ≥ 0.85 y recall ≥ 0.90 en clase resistente (Cumplido con BiGRU V2)
- [x] Reporte final consolidado (en `docs/5_experiments.md` y `results/bigru_v2/OUTPUT.txt`)

---

## Fase 4 — Arquitectura Experta: Multi-Stream BiGRU

### 4.1 Modelo
- [x] `AMRMultiBiGRU` (`src/multi_bigru_model.py`): 3 streams (k=3,4,5) → BiGRU(hidden=64) → Attention → Concatenación → Dense → Dense(1)
- [x] Soporte para inputs tipo tupla en training loop y evaluación.

### 4.2 Entrenamiento
- [x] Segmentación dinámica de vectores MLP en `AMRDataset`.
- [x] Comando `train-multi-bigru` en `main.py`.
- [x] 46 unit tests pasando (incluyendo `tests/test_multi_bigru.py`).

### 4.3 Experimento final
- [x] Entrenar y evaluar Multi-Stream BiGRU
- [x] Comparación definitiva: MLP vs BiGRU vs Multi-Stream BiGRU
- [x] Reporte de interpretabilidad multiescala (en `results/multi_bigru/OUTPUT.txt`)

---

## Fase 5 — Token BiGRU: Secuenciación Real de k-meros

### 5.1 Pipeline de tokenización
- [x] `KmerExtractor.to_token_sequence()`: extracción de tokens (IDs de 2 bits) con subsampling uniforme (linspace) para cobertura global [Haykin, Cap. 1.2].
- [x] Comando `prepare-tokens` en `main.py`: extracción paralela y guardado en `data/processed/token_bigru/`.
- [x] `AMRDataset` modificado para cargar tokens como tensores `long` [Mikolov13].

### 5.2 Modelo
- [x] `AMRTokenBiGRU` (`src/models/token_bigru/model.py`): input `[batch, 4096]` (IDs) → Embedding(257, 64) → BiGRU(128, layers=2, dropout=0.3) → Bahdanau attention → context `[batch, 256]` + antibiotic embedding → MLP → Logit.
- [x] `TokenBiGRUDataset` (`src/models/token_bigru/dataset.py`): subclase de `BaseAMRDataset`, carga tokens como `LongTensor`.
- [x] Reutilización de `BahdanauAttention` de `models.bigru.model`.
- [x] 10 unit tests pasando (`tests/models/test_token_bigru.py` + `tests/models/test_datasets.py`). 128 totales.

### 5.3 Entrenamiento y evaluación

**Iteración 1** — configuración base (lr=0.001, pos_weight_scale=2.5, GRU 1 capa, sin weight decay):
- [x] Comando `train-token-bigru` en `main.py` con gradient clipping (1.0) y pos_weight_scale (2.5).
- [x] Entrenar y evaluar Token BiGRU — F1=0.8165, Recall=0.9066, AUC=0.8251. Early stopping en época 13.
- [x] Diagnóstico: overfitting severo desde época 1. train_loss=0.32 vs val_loss=1.02 en época 13; mejor val F1 (0.809) en época 1.

**Iteración 2** — corrección de overfitting:
- [x] Cambios respecto a iter. 1: GRU 2 capas + dropout recurrente (0.3) entre capas, lr=0.0005, pos_weight_scale=1.6, weight_decay=1e-4. Parámetros: 537K (vs 240K en iter. 1).
- [x] Re-entrenamiento y evaluación: F1=0.8121, **Recall=0.9567**, AUC=0.8190.
- [x] Análisis de limitaciones: el subsampling uniforme (1 token/~1,100 bp) diluye la señal posicional de genes de resistencia (~800-2,000 bp), explicando por qué los histogramas (censo completo) superan a los tokens (muestra dispersa) en F1. Documentado en `PLAN_TOKEN_BIGRU.md`.
- [x] Análisis de trabajo futuro: evaluación de Transformers sparse y modelos genómicos pre-entrenados (DNABERT-2, HyenaDNA, Nucleotide Transformer). Documentado en `PLAN_TOKEN_BIGRU.md` y `5_experiments.md`.

---

---

## Fase 6 — Arquitecturas Jerárquicas y Revisión de MultiBiGRU

### 6.1 Refactorización MultiBiGRU (encoder order-independent)
- [x] Reemplazar BiGRU por stream por encoder sin dependencias secuenciales: `KmerStream` = LayerNorm(elementwise_affine=False) + proyección element-wise + `bin_importance` + attention pooling
- [x] Fusión entre streams con `softmax` (en lugar de `sigmoid` independiente) para garantizar competencia entre streams
- [x] Añadir `weight_decay` (AdamW) al comando `train-multi-bigru`
- [x] Actualizar tests: `test_gate_sum_to_one`, `test_no_sequential_modules`, `test_bin_importance_is_per_bin_prior`
- [x] Reentrenar MultiBiGRU con nueva arquitectura — F1=0.8514, Recall=0.8925, AUC=0.8944 (2026-04-15)

### 6.2 HierBiGRU — Cobertura total del genoma con histogramas segmentados
- [x] `KmerExtractor.to_tiled_histogram_matrix()`: divide el genoma en HIER_N_SEGMENTS segmentos, histograma k=4 (256 dims) por segmento
- [x] `extract_and_save_hier()`: extracción paralela con `n_jobs`, guarda `.npy` en `hier_bigru/`
- [x] Comando `prepare-hier` en `main.py`
- [x] `AMRHierBiGRU`: BiGRU profunda (2 capas) + BahdanauAttention sobre HIER_N_SEGMENTS segmentos
- [x] `HierBiGRUDataset`: carga matrices `[HIER_N_SEGMENTS, 256]` con validación de shape
- [x] Comando `train-hier-bigru` en `main.py`
- [x] 8 tests unitarios en `tests/models/test_hier_bigru.py`
- [x] Correr `prepare-hier` con HIER_N_SEGMENTS=256 — 9060 genomas en ~574 s, datos en `data/processed/hier_bigru/`
- [x] Entrenar HierBiGRU — F1=0.8307, Recall=0.8788, AUC=0.8539. Early stopping época 43. No cumple criterio.

### 6.3 HierSet — Encoder de conjunto para histogramas segmentados
- [x] `AMRHierSet`: cross-attention query-key condicionada en el antibiótico (score(s,a) = h_s·q_a/√D) + proyección con dropout (sin dependencias secuenciales entre segmentos). Permutation-invariant.
- [x] `HierSetDataset`: subclase de `HierBiGRUDataset`, reutiliza mismos `.npy`
- [x] Comando `train-hier-set` en `main.py`
- [x] 11 tests unitarios en `tests/models/test_hier_set.py` (incluye `test_permutation_invariance` y `test_attention_varies_by_antibiotic`)
- [x] Entrenar HierSet — F1=0.8900, Recall=0.9088, AUC=0.9368. Early stopping época 65. **Mejor AUC del proyecto.**

### 6.4 Comparación final
- [x] Comparar: MLP vs BiGRU vs MultiBiGRU vs HierBiGRU vs HierSet — ver `docs/5_experiments.md`
- [x] Criterio de éxito: AUC-ROC ≥ 0.900 en algún modelo jerárquico — **cumplido por HierSet (AUC=0.9368)**

### 6.5 Documentación de ideas para la siguiente iteración de HierSet
- [x] Escribir ideas de mejora arquitectónica para HierSet en `docs/IDEAS_MEJORA_HIERSET.md`

---

## Fase 7 — HierSet v2: multi-head attention + histogramas multi-escala

Plan detallado en `plan_hier_set_v2.md`.

### 7.1 Pipeline de datos multi-escala
- [x] `HIER_KMER_SIZES`, `HIER_KMER_DIMS`, `HIER_KMER_DIM_MULTI`, `HIER_KMER_OFFSETS` en `src/data_pipeline/constants.py`
- [x] `KmerExtractor.to_tiled_multiscale_matrix()`: divide el genoma en HIER_N_SEGMENTS segmentos, concatena histogramas k=3,4,5 → shape (256, 1344). Separador `N * (max_k - 1)` = 4 N's.
- [x] `extract_and_save_hier_multi()` en `src/data_pipeline/pipeline.py`: extracción paralela con `n_jobs`, guarda `.npy` en `hier_set_v2/`
- [x] Comando `prepare-hier-multi` en `main.py`
- [x] 5 tests en `tests/data_pipeline/test_features.py` (shape, suma por fila, offsets por escala, empty fasta, no boundary k-mers)

### 7.2 Modelo AMRHierSetV2
- [x] `src/models/hier_set_v2/model.py`: multi-head cross-attention (H=4, d_head=32) vía `torch.einsum`, input 1344 dims/segmento, `_attention_weights` shape `[B, H, S]` para interpretabilidad.
- [x] `src/models/hier_set_v2/dataset.py`: `HierSetV2Dataset` carga desde `hier_set_v2/` con validación de shape `(256, 1344)`
- [x] 13 tests en `tests/models/test_hier_set_v2.py` (forward shape, attention shape+softmax por cabeza, permutation-invariance, atención varía por antibiótico, dataset con fixtures)
- [x] Comando `train-hier-set-v2` en `main.py` (mismos hiperparámetros que `train-hier-set` para comparación justa)

### 7.3 Experimento y evaluación
- [x] Ejecutar `prepare-hier-multi --n-jobs -1` sobre los 9060 genomas
- [x] Entrenar `train-hier-set-v2` y comparar contra HierSet v1 (F1=0.8900, AUC=0.9368)
- [x] Objetivo: F1 ≥ 0.8950, AUC ≥ 0.9400, Recall ≥ 0.9000 → **No cumplido** (F1=0.8895, AUC=0.9366, Recall=0.8971 — resultado negativo)
- [x] Actualizar `docs/5_experiments.md` (Experimento 7) con resultados

### 7.4 Análisis post-hoc de umbrales por antibiótico
- [x] Script `scripts/per_antibiotic_threshold.py` — evalúa umbrales por antibiótico offline sobre v1 sin reentrenar
- [x] Resultado negativo: ΔF1 = −0.0048 (0.8900 → 0.8852) por sobreajuste del umbral al val set
- [x] Documentado en `docs/5_experiments.md` (sección "Análisis post-hoc")

---

## Decisiones

### Resueltas
- [x] Dim embedding antibiótico → **49** `[min(50, (96 // 2) + 1)]` (96 antibióticos en dataset)
- [x] pos_weight → **0.8522** (Susceptible/Resistant, del EDA)
- [x] Early stopping → pérdida de validación, patience=10
- [x] *Enterobacter spp.* ausente del dataset — excluido por requerir doble consulta (primero genomas, luego AMR) dado que la API de BV-BRC no soporta filtrar por linaje en el endpoint de AMR, añadiendo complejidad innecesaria.

### Pendientes
- [ ] Descargar `docs/reference/schuster1997_birnn.pdf` — Schuster & Paliwal (1997), "Bidirectional Recurrent Neural Networks", IEEE. Requiere credenciales universitarias. URL: https://ieeexplore.ieee.org/document/650093
