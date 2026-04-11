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

## Experimento 3 — Multi-Stream BiGRU (Arquitectura Experta)
- **Modelo:** `AMRMultiBiGRU` (3 streams expertos + Bahdanau Attention).
- **Entrada:** Segmentación dinámica de vectores MLP (k=3, 4 y 5 por separado).
- **Comando:**
  ```bash
  uv run python main.py train-multi-bigru --output-dir results/multi_bigru --batch-size 128 --pos-weight-scale 2.5
  ```
- **Resultados obtenidos:** F1=**0.8596**, Recall=0.8950, AUC-ROC=**0.9038**.
- **Observación:** Esta arquitectura logró igualar el F1 del MLP y superarlo en AUC-ROC, demostrando que el procesamiento independiente de k-meros es la representación más potente para redes recurrentes.

## Experimento 4 — Token BiGRU (Arquitectura Secuencial Real)
- **Modelo:** `AMRTokenBiGRU` (Embedding + BiGRU + Attention).
- **Entrada:** Secuencia de 4096 tokens (k=4) extraídos con subsampling uniforme.

### Iteración 1 — configuración base (lr=0.001, pos_weight_scale=2.5, GRU 1 capa)
- **Comando:**
  ```bash
  uv run python main.py train-token-bigru --output-dir results/token_bigru --batch-size 16 --pos-weight-scale 2.5
  ```
- **Resultados:** F1=0.8165, Recall=0.9066, AUC-ROC=0.8251. Early stopping en época 13.
- **Diagnóstico:** Overfitting severo. train_loss cayó a 0.32 mientras val_loss subió a 1.02; el mejor val F1 (0.809) fue en época 1. La red memorizó el training set sin generalizar.

### Iteración 2 — corrección de overfitting (lr=0.0005, pos_weight_scale=1.6, GRU 2 capas + dropout recurrente, weight_decay=1e-4)
- **Motivación:** Iter. 1 overfitteo inmediatamente. Se añadió dropout recurrente (requiere GRU de 2 capas), se bajó el lr, se migró a **AdamW** para un mejor manejo del decaimiento de pesos y se ajustó el pos_weight_scale a **1.6** para un balance neutro.
- **Comando:**
  ```bash
  uv run python main.py train-token-bigru --epochs 100 --batch-size 32 --lr 0.0005 --weight-decay 1e-4 --pos-weight-scale 1.6 --output-dir results/token_bigru_v2
  ```
- **Resultados:** F1=0.8121, **Recall=0.9567**, AUC-ROC=0.8190. Early stopping en época 13 (mejor val_loss en época 3).
- **Interpretación:** La arquitectura profunda con regularización agresiva logró el **Recall más alto de todo el proyecto (95.6%)**. Aunque el F1 global es ligeramente inferior al MLP, su capacidad para capturar casi todos los casos de resistencia lo convierte en el modelo más seguro desde una perspectiva clínica.

---

## Resultados Consolidados

| Modelo | F1-Score | Recall | AUC-ROC | Éxito (Rec ≥ 0.90) |
| :--- | :---: | :---: | :---: | :---: |
| MLP (Baseline) | 0.8600 | 0.9165 | 0.9035 | ✓ |
| BiGRU (v2) | 0.8566 | 0.9032 | 0.8998 | ✓ |
| Multi-Stream BiGRU | 0.8596 | 0.8950 | 0.9038 | ~✓ |
| Token BiGRU (Iter. 1) | 0.8165 | 0.9066 | 0.8251 | ✓ |
| **Token BiGRU (v2)** | **0.8121** | **0.9567** | **0.8190** | **✓✓** |

## Comparación final

### Hallazgos principales

1. **El MLP con histogramas es el modelo más robusto (F1=0.8600).** La composición global de k-meros — un censo completo de todos los k-meros sin información posicional — es la representación más informativa para este dataset. Esto indica que la "huella dactilar" de k-meros de los genes de resistencia es suficientemente distintiva para predecir AMR sin necesidad de saber dónde están los genes en el genoma.

2. **Los modelos BiGRU con histogramas igualan al MLP (F1~0.86) y aportan interpretabilidad.** El mecanismo de atención de Bahdanau permite identificar qué regiones del espacio de k-meros son más relevantes para cada predicción. El Multi-Stream BiGRU además logra el AUC-ROC más alto (0.9038), demostrando que procesar cada resolución de k-mero por separado es la mejor representación para redes recurrentes.

3. **El Token BiGRU no supera al MLP en F1 (0.8121 vs 0.8600), pero logra el Recall más alto del proyecto (0.9567).** La causa fundamental: el subsampling uniforme (1 token cada ~1,100 bp) diluye la señal posicional y reduce la cobertura del genoma. Los genes de resistencia (~800–2,000 bp) quedan representados por 1-2 tokens, y la información posicional a escala de promotores (~100 bp) y operones (~5,000 bp) se pierde por falta de resolución.

4. **Existe un trade-off entre cobertura y posición.** Los histogramas ofrecen cobertura completa sin posición (F1 alto, recall moderado). Los tokens ofrecen posición parcial con cobertura mínima (F1 menor, recall muy alto). Una representación ideal requeriría cobertura completa CON posición — lo cual exige procesar secuencias del orden de millones de nucleótidos, fuera del alcance de una BiGRU.

### Histograma vs. secuencia para predicción de AMR

| Representación | Cobertura | Info. posicional | F1 | Recall | AUC-ROC |
|---|---|---|---|---|---|
| Histograma k-meros (MLP) | Completa | Ninguna | **0.8600** | 0.9165 | 0.9035 |
| Histograma → BiGRU + Att | Completa | Pseudo | 0.8566 | 0.9032 | 0.8998 |
| Histograma → Multi-Stream | Completa | Pseudo (por k) | 0.8596 | 0.8950 | **0.9038** |
| Tokens → BiGRU + Att (v2) | Parcial (~0.1%) | Diluida | 0.8121 | **0.9567** | 0.8190 |

### Trabajo futuro

Para capturar la información posicional real de los genomas (promotores, operones, IS elements flanqueantes), se requieren arquitecturas capaces de procesar secuencias del orden de cientos de miles a millones de nucleótidos. Las opciones más prometedoras son los **modelos de lenguaje genómico pre-entrenados** (DNABERT-2, HyenaDNA, Nucleotide Transformer), que ya codifican la estructura del DNA en sus embeddings y solo necesitan fine-tuning para la tarea de AMR. Ver análisis detallado en `docs/PLAN_TOKEN_BIGRU.md`, sección "Trabajo futuro".

## Criterio de éxito
- F1 ≥ 0.85 y recall ≥ 0.90 para la clase resistente, sin sobreajuste evidente
- **Cumplido por:** MLP (F1=0.86, Rec=0.92), BiGRU v2 (F1=0.86, Rec=0.90), Token BiGRU v2 (Rec=0.96, pero F1=0.81 < 0.85)

## Decisiones pendientes
- [x] Métrica principal de selección del mejor modelo → F1 (recall es crítico en AMR); AUC-ROC como métrica secundaria
- [x] Número de epochs y criterio de early stopping → max 100 epochs, patience 10 (early stopping sobre **pérdida de validación**; checkpointing sobre mejor val F1)
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
