# Models

## Modelo A — MLP (línea de base superficial)

**Nota sobre profundidad:** La arquitectura usa 2 capas ocultas (512 → 128), lo cual
técnicamente supera la definición mínima de "deep" (>1 capa oculta). Sin embargo, la
consideramos superficial en el contexto de este proyecto por dos razones: (1) la propuesta
define "superficial" en contraste con la BiGRU+Attention, que posee capas recurrentes,
mecanismo de atención y mayor capacidad de modelar dependencias secuenciales; (2) en
la literatura de deep learning, redes de 2-3 capas densas se consideran shallow frente a
arquitecturas con decenas o cientos de capas. La segunda capa oculta (128) cumple un rol
de compresión progresiva — reduce la dimensionalidad antes de la capa de salida — y no
introduce la complejidad arquitectónica que distingue a un modelo profundo.

**Entradas:**
- Vector de histograma de k-meros concatenado (1344 dimensiones, normalizado)
- Antibiótico como índice entero → embedding aprendido (dim TBD)

**Arquitectura:**
```mermaid
graph TD
    GIn["Genomic Input<br/>(1344)"]
    AIn["Antibiotic Index"]
    Emb["Antibiotic Embedding"]
    Cat["Concatenate"]
    L1["Dense(512, ReLU)<br/>+ Dropout"]
    L2["Dense(128, ReLU)<br/>+ Dropout"]
    L3["Dense(1, Sigmoid)"]

    GIn --> Cat
    AIn --> Emb
    Emb --> Cat
    Cat --> L1
    L1 --> L2
    L2 --> L3
```

![Arquitectura MLP](imagenes/model_architecture.png)

### Justificación de la arquitectura

La elección de los tamaños de las capas (**1393 → 512 → 128 → 1**) responde a un diseño de **compresión progresiva** (embudo) fundamentado en los siguientes principios:

1. **Capacidad y Generalización (Haykin, Cap. 4.11):** El tamaño de las capas determina la capacidad de la red para extraer estadísticas de orden superior. Un tamaño de 512 neuronas en la primera capa es suficiente para procesar la entrada dispersa de 1393 dimensiones (k-meros + embedding) sin incurrir en una explosión de parámetros que lleve a la memorización del ruido (overfitting).
2. **Jerarquía de Características:** Según Haykin (Cap. 4.13), el uso de dos capas ocultas permite aprender representaciones jerárquicas de forma más eficiente que una sola capa ancha. La capa de 128 neuronas actúa como un cuello de botella (*bottleneck*) que obliga a la red a sintetizar la información más relevante para la resistencia antes de la clasificación final.
3. **Eficiencia Computacional:** Se utilizan potencias de 2 (**512, 128**) para aprovechar las optimizaciones de hardware en GPU (CUDA/cuDNN), que están diseñadas para procesar bloques de datos alineados con estas dimensiones, acelerando el entrenamiento.
4. **Regularización:** Esta arquitectura, combinada con una tasa de **Dropout de 0.3**, garantiza que la capacidad de la red esté equilibrada con respecto al tamaño del dataset (Fase 1), siguiendo la recomendación de Haykin de mantener una relación saludable entre el número de ejemplos y el número de pesos libres.

**Función de pérdida:** Binary Cross-Entropy
**Optimizador:** Adam
**Regularización:** Dropout (tasa 0.3), Early Stopping

#### Comandos CLI

```bash
# Entrenar MLP con hiperparámetros por defecto:
uv run python main.py train-mlp

# Personalizar entrenamiento:
uv run python main.py train-mlp --epochs 50 --batch-size 64 --lr 0.0005 --patience 5

# Especificar rutas de datos y resultados:
uv run python main.py train-mlp --data-dir data/processed --output-dir results/mlp_exp1
```

---

## Modelo B — BiGRU + Attention (modelo profundo)

### Arquitectura — Basada en [Lugo21]

**Entradas:**
- Matriz de histogramas de k-meros (1024×3): k=3,4,5 cada uno paddeado a 1024 → `[batch, 1024, 3]`. Representación distribuida invariante al orden de nodos en FASTA [Lugo21, p. 647].
- Antibiótico como índice entero → embedding aprendido (dim 49).

**Arquitectura:**
```mermaid
graph TD
    GIn["Genomic Input<br/>[batch, 1024, 3]"]
    AIn["Antibiotic Index<br/>[batch]"]
    Emb["Antibiotic Embedding<br/>(49)"]
    RNN["BiGRU<br/>(hidden=128, bidirectional)"]
    Att["Bahdanau Attention<br/>(dim=128)"]
    Cat["Concatenate<br/>[batch, 305]"]
    L1["Dense(128, ReLU)<br/>+ Dropout(0.3)"]
    L2["Dense(1, Logit)"]

    GIn --> RNN
    RNN -->|"outputs [batch, 1024, 256]"| Att
    Att -->|"context [batch, 256]"| Cat
    AIn --> Emb
    Emb -->|"embedding [batch, 49]"| Cat
    Cat --> L1
    L1 --> L2
```

![Arquitectura BiRNN Variante A](imagenes/birnn_a_arch.png)

### Justificación de la arquitectura

1. **BiGRU (128 unidades):** Basada en [Lugo21]. La bidireccionalidad [Schuster97] captura contexto en ambas direcciones de la secuencia genómica. Las GRU [Cho14] resuelven el problema de dependencias a largo plazo de forma más simple que las LSTM.
2. **Atención Aditiva (Bahdanau):** Implementa el mecanismo de [Bahdanau15] para comprimir los 1024 timesteps en un solo vector de contexto, permitiendo al modelo "enfocarse" en los k-meros más informativos.
3. **Regularización (Dropout 0.3):** Aunque [Lugo21] sugiere 0.5, se redujo a **0.3** tras observar oscilaciones excesivas en el entrenamiento. Un valor de 0.3 es más apropiado para el tamaño de nuestra cabeza clasificadora (305→128→1) [Srivastava14] y mejora la estabilidad de la convergencia.
4. **Gradient Clipping (max_grad_norm=1.0):** Necesario para prevenir el problema de **gradientes explosivos** [Pascanu13] durante la retropropagación a través del tiempo (BPTT) [Haykin, Cap. 15.3] sobre secuencias largas.

### Justificación del Manejo de Desbalance (pos_weight)

El modelo utiliza una variante de **Entropía Cruzada Binaria Ponderada** (Weighted BCE). El parámetro `pos_weight` se calcula dinámicamente como el ratio $N_{susceptible} / N_{resistente}$ y se escala por un factor de **2.5**. Esta decisión se fundamenta en:

1.  **Teoría de la Decisión de Bayes (Haykin, Cap. 1.4):** El umbral óptimo de decisión depende de la relación de costos entre errores. Dado que en AMR un Falso Negativo (FN) es críticamente más peligroso que un Falso Positivo (FP), escalamos la pérdida para penalizar asimétricamente los FN.
2.  **Aprendizaje Sensible al Costo (Cost-Sensitive Learning):** El factor de 2.5 define matemáticamente que omitir un organismo resistente es **2.5 veces más costoso** para el modelo que una falsa alarma. Esto desplaza la frontera de decisión para maximizar el **Recall clínico**.
3.  **Calibración por Objetivos (King & Zeng, 2001):** El valor 2.5 se determinó mediante validación empírica como el multiplicador necesario para satisfacer la restricción técnica de **Recall ≥ 0.90** sin degradar excesivamente la precisión global.

**Función de pérdida:** Binary Cross-Entropy con pesos (`pos_weight`) para desbalance.
**Optimizador:** Adam (lr=0.001) [Kingma15].
**Evaluación:** Umbral calibrado en validación para maximizar F1.

#### Comandos CLI

```bash
# Entrenar BiGRU con hiperparámetros por defecto:
uv run python main.py train-bigru

# Personalizar entrenamiento:
uv run python main.py train-bigru --epochs 50 --batch-size 16 --lr 0.0005
```

---

## Hiperparámetros finales

| Hiperparámetro | MLP | BiGRU | MultiBiGRU | HierBiGRU | HierSet |
|---|---|---|---|---|---|
| Embedding antibiótico | 49 | 49 | 49 | 49 | 49 |
| Hidden size RNN/proj | — | 128 | 64/stream | 128 | 128 |
| Dropout | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 |
| Learning rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| Optimizador | AdamW | AdamW | AdamW | AdamW | AdamW |
| weight_decay | — | — | 1e-4 | — | 1e-3 |
| Gradient clipping | — | 1.0 | — | 1.0 | — |
| pos_weight_scale | 1.0 | 2.5 | 2.5 | 2.5 | 2.5 |
| Early stopping patience | 10 | 10 | 15 | 15 | 15 |
| **F1 (test)** | 0.8600 | 0.8566 | 0.8514 | 0.8307 | **0.8900** |
| **AUC-ROC (test)** | 0.9035 | 0.8998 | 0.8944 | 0.8539 | **0.9368** |
| **Recall (test)** | 0.9165 | 0.9032 | 0.8925 | 0.8788 | 0.9088 |

**Modelo seleccionado: HierSet** — mejor F1 y AUC-ROC del proyecto.

---

## Modelo C — Multi-Stream (Encoder Order-Independent)

### Arquitectura

Cada histograma de k-meros (k=3,4,5) se procesa con un `KmerStream` independiente que no introduce dependencias secuenciales entre bins. La fusión aprende qué escala de k-meros es más diagnóstica para cada antibiótico.

```mermaid
graph TD
K3["k=3 [batch, 64, 1]"]
K4["k=4 [batch, 256, 1]"]
K5["k=5 [batch, 1024, 1]"]
AB["Antibiotic Index"]

S3["KmerStream(64)<br/>LayerNorm → Linear(1→128) → bin_importance → attention"]
S4["KmerStream(256)<br/>LayerNorm → Linear(1→128) → bin_importance → attention"]
S5["KmerStream(1024)<br/>LayerNorm → Linear(1→128) → bin_importance → attention"]

Gate["stream_gate: Linear(49→3)<br/>softmax → gates [batch, 3]"]
Fused["Weighted sum [batch, 128]"]
Cat["Concatenate [batch, 177]"]
L1["Dense(128, ReLU) + Dropout(0.3)"]
L2["Dense(1, Logit)"]

K3 --> S3
K4 --> S4
K5 --> S5
AB --> Emb["Antibiotic Embedding(49)"]
Emb --> Gate
S3 --> Fused
S4 --> Fused
S5 --> Fused
Gate --> Fused
Fused --> Cat
Emb --> Cat
Cat --> L1 --> L2
```

### Justificación de la arquitectura

1.  **Encoder sin dependencias secuenciales:** La BiGRU original trataba los índices de bins como una secuencia, creando dependencias artificiales (bin_i influye sobre bin_{i+1} a través del estado oculto). El `KmerStream` proyecta cada bin de forma independiente — la representación de un k-mero no depende de qué otros k-meros lo precedieron en el tensor [Goodfellow16, Cap. 10].
2.  **`bin_importance`:** Prior aprendido por identidad de bin. A diferencia del sesgo secuencial, asigna un escalar fijo a cada k-mero específico sin crear dependencias entre bins adyacentes. Biológicamente válido: el k-mero "ACGT" siempre mapea al mismo bin.
3.  **Fusión condicionada por antibiótico con softmax [Ngiam11]:** Los gates de fusión usan `softmax` para que los tres streams compitan — garantiza que al menos un stream permanezca activo y el modelo no pueda ignorar el genoma apoyándose solo en el prior del antibiótico.
4.  **Regularización con AdamW [Loshchilov19]:** `weight_decay=1e-4` aplicado con AdamW para regularización L2 desacoplada.

#### Comandos CLI

```bash
# Entrenar Multi-Stream:
uv run python main.py train-multi-bigru
```

---

## Modelo D — Token BiGRU *(descartado)*

> **Estado:** descartado por diseño deficiente. F1=0.8121 (iter. 2), por debajo del umbral de éxito. El subsampling uniforme (1 token/~1,100 bp) diluye la señal de genes de resistencia (~800–2,000 bp). Ver análisis en `docs/PLAN_TOKEN_BIGRU.md`. No se recomienda usar `train-token-bigru`.

### Arquitectura

Esta arquitectura devuelve la RNN a su uso idiomatíco: procesar una **secuencia real de tokens discretos** (k-meros) preservando su orden y contexto posicional en el genoma [Cho14; Mikolov13].

```mermaid
graph TD
    TIn["Token Sequence<br/>[batch, 4096]<br/>(int64)"]
    AIn["Antibiotic Index<br/>[batch]"]

    EMB_T["Kmer Embedding<br/>(257, 64)<br/>padding_idx=256"]
    EMB_A["Antibiotic Embedding<br/>(n_antibiotics, 49)"]

    GRU["BiGRU<br/>(input=64, hidden=128,<br/>layers=2, bidirectional,<br/>dropout=0.3)"]

    ATT["BahdanauAttention<br/>(hidden=256, att=128)"]

    CAT["Concatenate<br/>[batch, 305]"]
    L1["Linear(305, 128)<br/>ReLU + Dropout(0.3)"]
    L2["Linear(128, 1)"]

    TIn --> EMB_T
    EMB_T -->|"[batch, 4096, 64]"| GRU
    GRU -->|"[batch, 4096, 256]"| ATT
    ATT -->|"context [batch, 256]"| CAT
    AIn --> EMB_A
    EMB_A -->|"[batch, 49]"| CAT
    CAT --> L1
    L1 --> L2
```

### Justificación de la arquitectura

1.  **Tokenización y Embedding [Mikolov13]:** En lugar de un histograma (bag-of-words), el genoma se representa como una secuencia de IDs de k-meros (k=4, vocab=256). La capa de embedding mapea estos símbolos a vectores densos de 64 dimensiones donde el modelo puede aprender relaciones biológicas entre k-meros.
2.  **Subsampling Uniforme [Haykin, Cap. 1.2]:** Para manejar genomas de millones de bases, se seleccionan 4096 tokens equidistantes. Esto garantiza una cobertura global del genoma completo sin el sesgo posicional de truncar la secuencia [Lugo21].
3.  **BiGRU Profunda (2 capas) [Cho14; Schuster97]:** La bidireccionalidad permite capturar contexto genómico en ambas direcciones. El uso de **2 capas recurrentes** permite que el modelo aprenda jerarquías de características más complejas (motivos locales -> regiones funcionales) y habilita el uso de **dropout recurrente** entre capas [Srivastava14], fundamental para regularizar secuencias largas de 4096 tokens.
4.  **Interpretabilidad Posicional [Bahdanau15]:** Los pesos de atención `[batch, 4096]` indican qué regiones específicas del genoma (muestreadas) son determinantes para la prediccion, permitiendo mapear la "atención" del modelo de vuelta a coordenadas genómicas reales.

#### Comandos CLI

```bash
# Preparar secuencias de tokens (k=4, max_len=4096):
uv run python main.py prepare-tokens --n-jobs -1

# Entrenar Token BiGRU:
uv run python main.py train-token-bigru --batch-size 32
```

---

## Modelo E — HierBiGRU (Cobertura total con histogramas segmentados)

### Arquitectura

Divide el genoma concatenado en HIER_N_SEGMENTS segmentos contiguos y calcula el histograma k=4 de cada segmento, garantizando 100% de cobertura. Una BiGRU profunda procesa la secuencia de histogramas y la atención de Bahdanau identifica los segmentos más diagnósticos.

```mermaid
graph TD
    IN["[batch, HIER_N_SEGMENTS, 256]<br/>HIER_N_SEGMENTS segmentos × histograma k=4"]
    GRU["BiGRU(hidden=128, layers=2, dropout=0.3)<br/>→ [batch, HIER_N_SEGMENTS, 256]"]
    ATT["BahdanauAttention(dim=128)<br/>→ context [batch, 256]"]
    EMB["Antibiotic Embedding(49)"]
    CAT["Concatenate [batch, 305]"]
    L1["Linear(305→128) + ReLU + Dropout(0.3)"]
    L2["Linear(128→1)"]

    IN --> GRU --> ATT --> CAT
    EMB --> CAT --> L1 --> L2
```

**Limitación conocida:** la BiGRU asume que los segmentos adyacentes en el tensor son biológicamente relacionados. En ensamblajes draft el orden de contigs es arbitrario, por lo que algunas adyacencias entre segmentos son artefactos del ensamblador.

**Resultados:** F1=0.8307, Recall=0.8788, AUC-ROC=0.8539. Early stopping época 43. No supera el criterio de éxito — la BiGRU introduce dependencias secuenciales artificiales entre segmentos que perjudican la señal cuando los genes de resistencia están distribuidos sin orden fijo.

#### Comandos CLI

```bash
# Preparar histogramas segmentados (una sola vez, para HierBiGRU y HierSet):
uv run python main.py prepare-hier --n-jobs -1

# Entrenar HierBiGRU:
uv run python main.py train-hier-bigru
```

---

## Modelo F — HierSet (Encoder de conjunto sobre segmentos)

### Arquitectura

Mismos datos que HierBiGRU (`hier_bigru/*.npy`), pero trata los HIER_N_SEGMENTS segmentos como un conjunto: proyección independiente por segmento + cross-attention query-key condicionada en el antibiótico. Es permutation-invariant sobre los segmentos que recibe.

```mermaid
graph TD
    IN["[batch, HIER_N_SEGMENTS, 256]<br/>HIER_N_SEGMENTS segmentos × histograma k=4"]
    NORM["LayerNorm(256) por segmento"]
    PROJ["Linear(256→128) + ReLU + Dropout(0.3)<br/>(independiente por segmento)"]
    EMB["Antibiotic Embedding(49)"]
    QUERY["Linear(49→128) → query [batch, 128]"]
    ATT["Cross-attention: score(s,a) = h_s·q_a/√D<br/>softmax → weighted sum → [batch, 128]"]
    CAT["Concatenate [batch, 177]"]
    L1["Linear(177→128) + ReLU + Dropout(0.3)"]
    L2["Linear(128→1)"]

    IN --> NORM --> PROJ --> ATT
    EMB --> QUERY --> ATT
    ATT --> CAT
    EMB --> CAT --> L1 --> L2
```

**Propiedad clave:** sin dependencias secuenciales entre segmentos — la representación del segmento i no depende del segmento j. La atención es condicionada por el antibiótico (cross-attention query-key), permitiendo que el modelo atienda distintas regiones del genoma según el antibiótico consultado.

**Alcance:** permutation-invariant sobre los HIER_N_SEGMENTS inputs tal como los recibe del pipeline. `prepare-hier` construye esos segmentos concatenando contigs en orden FASTA — HierSet no añade sesgo secuencial adicional, pero no elimina el que ya codifican las features.

**Resultados:** F1=**0.8900**, Recall=0.9088, AUC-ROC=**0.9368**. Early stopping época 65. **Mejor modelo del proyecto.**

#### Comandos CLI

```bash
# (Reutiliza los mismos .npy de prepare-hier)
uv run python main.py train-hier-set
```

