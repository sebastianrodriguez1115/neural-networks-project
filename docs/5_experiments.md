# Experiments

## Condiciones compartidas (para comparación justa)
- Mismo dataset y mismos splits (70/15/15 estratificado)
- Misma función de pérdida: `BCEWithLogitsLoss` con `pos_weight` calculado sobre el set de entrenamiento.
- Mismo optimizador: Adam (lr=0.001)
- Misma estrategia de desbalance de clases (basada en el ratio S/R del EDA).
- Semilla aleatoria fija (`42`) para reproducibilidad total [Haykin, §4.4].
- **Gradient Clipping:** `max_grad_norm=1.0` (opcional en MLP, obligatorio en BiGRU [Pascanu13]).

## Experimento 1 — MLP baseline (Shallow NN)
- **Modelo:** `AMRMLP` (512 → 128 → 1).
- **Entrada:** Vector concatenado de 1344 dims.
- **Comando:**
  ```bash
  uv run python main.py train-mlp --output-dir results/mlp --batch-size 32
  ```
- **Resultados obtenidos:** F1=0.8600, Recall=0.9165, AUC-ROC=0.9035.
- **Observación:** Superó el criterio de éxito (F1 ≥ 0.85).

## Experimento 2 — BiGRU + Attention (Deep NN)
- **Modelo:** `AMRBiGRU` (BiGRU 128 + Bahdanau Attention).
- **Entrada:** Matriz 1024×3 (representación distribuida [Lugo21]).
- **Comando (Mejorado):**
  ```bash
  uv run python main.py train-bigru --output-dir results/bigru_v2 --batch-size 128 --pos-weight-scale 2.5
  ```
- **Resultados obtenidos (V2):** F1=0.8566, **Recall=0.9032**, AUC-ROC=0.8998.
- **Observación:** Mediante el escalado del `pos_weight` (2.5x) y la reducción de Dropout (0.3), se logró superar el umbral clínico de sensibilidad.
- **Análisis de Atención:** El 92% de la energía se concentra en k=3 y k=4, validando que las frecuencias de bajo nivel dominan la señal de resistencia en ESKAPE.

## Experimento 5 — HierBiGRU (Cobertura total con histogramas segmentados)
- **Modelo:** `AMRHierBiGRU` (BiGRU profunda 2 capas + BahdanauAttention sobre HIER_N_SEGMENTS=256 segmentos).
- **Entrada:** Matriz [256, 256] — 256 segmentos de histograma k=4 (256 dims) con cobertura total del genoma.
- **Comando:**
  ```bash
  uv run python main.py train-hier-bigru
  ```
- **Resultados:** F1=0.8307, Recall=0.8788, AUC-ROC=0.8539. Early stopping época 43.
- **Observación:** No supera el criterio de éxito (F1 < 0.85, Recall < 0.90). La BiGRU impone dependencias secuenciales artificiales entre segmentos adyacentes del tensor, lo que perjudica la señal cuando los genes de resistencia están distribuidos a lo largo del genoma sin un orden fijo.

## Experimento 6 — HierSet (Encoder de conjunto, permutation-invariant)
- **Modelo:** `AMRHierSet` — cross-attention query-key condicionada en el antibiótico (`score(s,a) = h_s · q_a / √D`), permutation-invariant sobre los 256 segmentos.
- **Entrada:** Matriz [256, 256] — mismos segmentos que HierBiGRU (reutiliza los mismos `.npy`).
- **Optimizaciones vs. arquitectura base:** dropout tras proyección, `weight_decay=1e-3`, cross-attention (en lugar de la concatenación shift-invariant).
- **Comando:**
  ```bash
  uv run python main.py train-hier-set
  ```
- **Resultados (umbral 0.4939, óptimo sobre val):** F1=0.8900, Recall=0.9088, AUC-ROC=**0.9368**. Early stopping época 65.
- **Umbral de despliegue ajustado a θ=0.40:** Bajar el umbral de 0.4939 a 0.40 incrementa el recall en +0.020 con una caída de F1 de solo −0.002, lo que es clínicamente favorable (en AMR un falso negativo es más costoso que un falso positivo). Resultados con θ=0.40: **F1=0.8876, Recall=0.9289**, Precision=0.8498, AUC=0.9368.
- **Observación:** **Mejor modelo del proyecto** — mayor F1, mayor AUC y recall clínico ≥ 0.929 con θ=0.40. La eliminación de dependencias secuenciales (set encoder) permite que la atención condicionada por antibiótico identifique libremente los segmentos más relevantes, sin el sesgo de vecindad de la BiGRU.

## Experimento 3 — Multi-Stream BiGRU (Arquitectura Experta, order-independent)
- **Modelo:** `AMRMultiBiGRU` — 3 streams order-independent (`KmerStream`: LayerNorm(elementwise_affine=False) + proyección element-wise + `bin_importance` + attention pooling). Fusión softmax condicionada por antibiótico.
- **Entrada:** Segmentación dinámica de vectores MLP (k=3, 4 y 5 por separado).
- **Comando:**
  ```bash
  uv run python main.py train-multi-bigru
  ```
- **Resultados obtenidos (nueva arquitectura, 2026-04-15):** F1=**0.8514**, Recall=0.8925, AUC-ROC=**0.8944**. Early stopping época 35.
- **Observación:** La refactorización a encoder order-independent redujo ligeramente F1 y AUC respecto a la versión anterior (F1=0.8596, AUC=0.9038, arquitectura BiGRU). La nueva arquitectura es interpretable vía `bin_importance`, pero no alcanza AUC ≥ 0.900.

## Experimento 4 — Token BiGRU (Arquitectura Secuencial Real)
- **Modelo:** `AMRTokenBiGRU` (Embedding + BiGRU + Attention).
- **Entrada:** Secuencia de 4096 tokens (k=4) extraídos con subsampling uniforme.

### Iteración 1 — configuración base (lr=0.001, pos_weight_scale=2.5, GRU 1 capa)
- **Nota:** resultados históricos — `results/token_bigru/` eliminado (modelo descartado por diseño deficiente).
- **Resultados:** F1=0.8165, Recall=0.9066, AUC-ROC=0.8251. Early stopping en época 13.
- **Diagnóstico:** Overfitting severo. train_loss cayó a 0.32 mientras val_loss subió a 1.02; el mejor val F1 (0.809) fue en época 1. La red memorizó el training set sin generalizar.

### Iteración 2 — corrección de overfitting (lr=0.0005, pos_weight_scale=1.6, GRU 2 capas + dropout recurrente, weight_decay=1e-4)
- **Motivación:** Iter. 1 overfitteo inmediatamente. Se añadió dropout recurrente (requiere GRU de 2 capas), se bajó el lr, se migró a **AdamW** para un mejor manejo del decaimiento de pesos y se ajustó el pos_weight_scale a **1.6** para un balance neutro.
- **Nota:** resultados históricos — `results/token_bigru_v2/` eliminado (modelo descartado por diseño deficiente).
- **Resultados:** F1=0.8121, **Recall=0.9567**, AUC-ROC=0.8190. Early stopping en época 13 (mejor val_loss en época 3).
- **Interpretación:** La arquitectura profunda con regularización agresiva logró el **Recall más alto de todo el proyecto (95.6%)**. Aunque el F1 global es ligeramente inferior al MLP, su capacidad para capturar casi todos los casos de resistencia lo convierte en el modelo más seguro desde una perspectiva clínica.

---

## Resultados Consolidados

| Modelo | F1-Score | Recall | AUC-ROC | Éxito (F1 ≥ 0.85 y Rec ≥ 0.90) |
| :--- | :---: | :---: | :---: | :---: |
| **HierSet (θ=0.40)** | **0.8876** | **0.9289** | **0.9368** | ✓ |
| HierSet (θ=0.4939, óptimo) | 0.8900 | 0.9088 | 0.9368 | ✓ |
| MLP (Baseline) | 0.8600 | 0.9165 | 0.9035 | ✓ |
| BiGRU (v2) | 0.8566 | 0.9032 | 0.8998 | ✓ |
| MultiBiGRU (order-independent) | 0.8514 | 0.8925 | 0.8944 | ~✓ |
| HierBiGRU (256 segs) | 0.8307 | 0.8788 | 0.8539 | ✗ |

## Comparación final

### Hallazgos principales

1. **HierSet (256 segmentos) es el mejor modelo del proyecto** (F1=0.8876, AUC=0.9368, Recall=0.9289 con θ=0.40). La representación geográfica del genoma — histogramas locales por segmento — combinada con un encoder permutation-invariant y atención condicionada por antibiótico supera a todos los modelos basados en histogramas globales. El AUC-ROC de 0.9368 es el más alto registrado en el proyecto. El umbral de despliegue se fijó en θ=0.40 (vs. θ=0.4939 óptimo en F1) para priorizar recall clínico (+0.020 recall, −0.002 F1).

2. **El MLP con histogramas globales es el mejor modelo entre los que usan representación plana** (F1=0.8600, AUC=0.9035, Recall=0.9165). La "huella dactilar" de k-meros de los genes de resistencia es suficientemente distintiva sin necesidad de localización posicional. Supera a los modelos BiGRU sobre histogramas globales, confirmando que la composición global es una representación muy informativa.

3. **HierBiGRU no supera el criterio de éxito** (F1=0.8307, Recall=0.8788). A pesar de usar los mismos 256 segmentos que HierSet, las dependencias secuenciales artificiales de la BiGRU entre segmentos adyacentes perjudican la señal cuando los genes de resistencia están distribuidos sin orden fijo en el genoma.

4. **El encoder de conjunto (HierSet) supera al encoder secuencial (HierBiGRU)** sobre los mismos datos jerárquicos, validando que la invarianza a la permutación es la propiedad correcta para este problema: los genes de resistencia no tienen una posición fija en el genoma.

### Comparación de arquitecturas

| Representación | Cobertura | Sesgo secuencial | F1 | Recall | AUC-ROC |
|---|---|---|---|---|---|
| **Hist. segmentado → HierSet** | **Total (256 segs)** | **Ninguno** | **0.8900** | 0.9088 | **0.9368** |
| Histograma global (MLP) | Total | Ninguno | 0.8600 | **0.9165** | 0.9035 |
| Histograma global → BiGRU | Total | Pseudo (orden k-meros) | 0.8566 | 0.9032 | 0.8998 |
| Histograma global → MultiBiGRU | Total | Por escala de k | 0.8514 | 0.8925 | 0.8944 |
| Hist. segmentado → HierBiGRU | Total (256 segs) | Sí (entre segs) | 0.8307 | 0.8788 | 0.8539 |

### Trabajo futuro

Para capturar la información posicional real de los genomas (promotores, operones, IS elements flanqueantes), se requieren arquitecturas capaces de procesar secuencias del orden de cientos de miles a millones de nucleótidos. Las opciones más prometedoras son los **modelos de lenguaje genómico pre-entrenados** (DNABERT-2, HyenaDNA, Nucleotide Transformer), que ya codifican la estructura del DNA en sus embeddings y solo necesitan fine-tuning para la tarea de AMR. Ver análisis detallado en `docs/PLAN_TOKEN_BIGRU.md`, sección "Trabajo futuro".

## Criterio de éxito
- F1 ≥ 0.85 y recall ≥ 0.90 para la clase resistente, sin sobreajuste evidente
- **Cumplido por:** HierSet θ=0.40 (F1=0.89, Rec=0.93, AUC=0.94), MLP (F1=0.86, Rec=0.92), BiGRU v2 (F1=0.86, Rec=0.90)
- **No cumplido por:** HierBiGRU (F1=0.83, Rec=0.88), MultiBiGRU (F1=0.85, Rec=0.89 — borderline), Token BiGRU (F1=0.81 — descartado)
- **Modelo seleccionado:** HierSet con θ=0.40 — mejor AUC, mejor recall clínico (0.9289) y F1 ≥ 0.88

## Decisiones pendientes
- [x] Métrica principal de selección del mejor modelo → F1 (recall es crítico en AMR); AUC-ROC como métrica secundaria
- [x] Número de epochs y criterio de early stopping → max 100 epochs, patience 10 (early stopping sobre **val_F1**; checkpointing sobre mejor val F1)
  - Nota: un *step* es una actualización de pesos con un mini-batch; un *epoch* es una pasada completa por el dataset. El artículo de referencia reporta steps (2800–3600), pero como nuestro dataset tendrá un tamaño diferente, trabajamos en epochs para que el criterio sea independiente del tamaño del dataset.
- [x] Variación de k → usar k=3,4,5 fijo (el artículo de referencia probó k=6 sin mejora, k=3,4,5 está justificado)

---

## Glosario de métricas de desempeño

Dada la naturaleza crítica de la predicción de RAM, se emplean las siguientes métricas de clasificación binaria para evaluar los modelos:

| Métrica | Definición matemática | Interpretación en Microbiología |
|---|---|---|
| **Recall** (Sensibilidad) | $TP / (TP + FN)$ | **Métrica prioritaria**. Representa la capacidad del modelo para detectar todos los casos resistentes. Un recall alto (ej. >0.90) asegura que casi ninguna bacteria resistente sea clasificada erróneamente como susceptible. |
| **Precision** | $TP / (TP + FP)$ | Representa la confiabilidad de una alerta de resistencia. Una precisión alta indica que si el modelo dice que una bacteria es resistente, es muy probable que lo sea, evitando el uso innecesario de antibióticos de reserva. |
| **F1-Score** | $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$ | Media armónica entre precisión y recall. Es el indicador global de la calidad del modelo, especialmente útil cuando las clases están desbalanceadas. Nuestro éxito se define por un **F1 ≥ 0.85**. |
| **Accuracy** | $(TP + TN) / Total$ | Porcentaje global de aciertos. Aunque es útil, puede ser engañoso si hay mucho desbalance; por ello, no es nuestra métrica principal de decisión. |
| **AUC-ROC** | Área bajo la curva ROC | Mide la capacidad de discriminación del modelo (qué tan bien separa resistentes de susceptibles) independientemente del umbral de decisión elegido. |

*Abreviaturas: TP (True Positive/Resistente correcto), TN (True Negative/Susceptible correcto), FP (False Positive/Falsa alarma), FN (False Negative/Resistente omitido).*
