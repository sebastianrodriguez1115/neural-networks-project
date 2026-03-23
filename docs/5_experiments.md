# Experiments

## Condiciones compartidas (para comparación justa)
- Mismo dataset y mismos splits (70/15/15 estratificado)
- Misma función de pérdida: Binary Cross-Entropy
- Mismo optimizador: Adam
- Misma estrategia de desbalance de clases
- Semilla aleatoria fija (`random_seed = 42`) en numpy, PyTorch y scikit-learn para reproducibilidad

## Experimento 1 — MLP baseline
- Entrenar el MLP sobre el vector de k-meros concatenado
- Registrar: accuracy, precision, recall, F1, AUC-ROC en el conjunto de prueba
- Registrar: curvas de pérdida (train vs validation) para detectar sobreajuste

## Experimento 2 — BiRNN + Attention (Variante A, artículo de referencia)
- Entrenar la BiRNN sobre la matriz 3×1024 de histogramas de k-meros
- Registrar las mismas métricas que el Experimento 1
- Registrar: pesos de atención para analizar qué k (3, 4 o 5) es más informativo

## Experimento 3 — BiRNN + Attention (Variante B, secuencia ordenada, opcional)
- Entrenar la BiRNN sobre la secuencia ordenada de k-meros si el tiempo lo permite
- Registrar las mismas métricas que los experimentos anteriores
- Registrar: pesos de atención para analizar qué regiones del genoma son más informativas

## Comparación final
- Comparar todos los modelos implementados en el conjunto de prueba bajo las mismas condiciones
- Analizar la matriz de confusión de cada uno, con énfasis en falsos negativos (resistentes clasificados como susceptibles)
- Reportar si la complejidad adicional de la BiRNN se justifica con una mejora significativa en F1 y recall

## Criterio de éxito
- F1 ≥ 0.85 y recall ≥ 0.90 para la clase resistente, sin sobreajuste evidente

## Decisiones pendientes
- [x] Métrica principal de selección del mejor modelo → F1 (recall es crítico en AMR); AUC-ROC como métrica secundaria
- [x] Número de epochs y criterio de early stopping → max 100 epochs, patience 10 (early stopping sobre F1 de validación)
  - Nota pendiente: hacer early stopping sobre F1 requiere evaluación completa por epoch (costoso). Alternativa más estándar: early stopping sobre pérdida de validación y usar F1 solo para selección final. Evaluar en implementación.
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
