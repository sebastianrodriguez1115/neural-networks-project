# IDEAS_MEJORA_HIERSET.md — Ideas de mejora para HierSet

## Objetivo

Dejar por escrito posibles mejoras para `AMRHierSet`, el mejor modelo actual del
proyecto (`F1=0.8900`, `Recall=0.9088`, `AUC-ROC=0.9368`), priorizando cambios
arquitectónicos con buena relación impacto/riesgo.

La idea no es "hacerlo más grande" sin criterio, sino atacar los límites reales
del diseño actual.

> **Nota:** Dos ideas ya tienen plan de implementación concreto en
> `plan_hier_set_v2.md`: **multi-head cross-attention** (H=4) e **histogramas
> multi-escala** (k=3,4,5 por segmento). Este documento cubre las ideas
> restantes que podrían explorarse después de v2 o de forma independiente.

---

## Estado actual del modelo

`HierSet` recibe una matriz `[HIER_N_SEGMENTS, 256]` por genoma:

- `HIER_N_SEGMENTS = 256`
- Cada fila es un histograma de frecuencias relativas de k-meros `k=4`
- El encoder actual es:
  `LayerNorm(256) -> Linear(256→128) -> ReLU -> Dropout -> cross-attention condicionada en antibiótico -> clasificador`

Fortalezas ya demostradas:

- Elimina las dependencias secuenciales artificiales de `HierBiGRU`
- La atención sí depende del antibiótico consultado
- Es permutation-invariant sobre los segmentos recibidos
- Supera a `MLP`, `BiGRU`, `MultiBiGRU` y `HierBiGRU`

Limitaciones estructurales del diseño actual:

- Cada segmento se comprime con una sola capa lineal; la representación local es superficial.
- El antibiótico condiciona el pooling, pero no condiciona la representación del segmento.
- Todo el genoma se resume en un único vector de contexto; eso puede perder evidencia distribuida.
- Los inputs son histogramas normalizados por segmento; el modelo ve composición agregada, no orden interno ni abundancia absoluta.

---

## Diagnóstico

La mejora de `HierSet` sobre `HierBiGRU` sugiere que la hipótesis correcta no es
"el genoma segmentado necesita una secuencia", sino:

> la señal de resistencia está distribuida en regiones locales, pero el orden
> lineal entre segmentos no es confiable como sesgo inductivo fuerte.

Por eso las mejoras más prometedoras no van por RNNs más profundas, sino por:

- mejor representación por segmento;
- mejor interacción segmento-antibiótico;
- pooling menos restrictivo que una sola atención softmax;
- combinación explícita de señal local y señal global.

---

## Prioridades

### 1. Condicionar el encoder de segmento por antibiótico

#### Problema

Hoy la representación `h_s` de cada segmento es antibiótico-agnóstica. El
antibiótico solo entra cuando se calculan los scores de atención y en el
clasificador final.

Eso limita al modelo a responder:

- "¿qué segmentos parecen importantes en general para este genoma?"

cuando lo deseable es:

- "¿cómo debe representarse este segmento cuando la pregunta es sobre este antibiótico específico?"

#### Idea

Aplicar modulación condicional tipo FiLM o adapter:

```text
z_s = MLP(segment_s)
gamma, beta = f(ab_emb)
h_s = gamma * z_s + beta
```

Variantes:

- `FiLM` simple sobre la salida de la proyección
- `FiLM` por bloque si el encoder de segmento gana profundidad
- concatenación `segment ++ ab_emb` dentro del encoder, manteniendo el mismo encoder para todos los segmentos

#### Impacto esperado

Alto. Es la mejora más alineada con la naturaleza multi-antibiótico del
problema.

#### Riesgo

Bajo a medio. Aumenta parámetros, pero no cambia el pipeline ni rompe la
invarianza a permutación.

---

### 2. Reemplazar el pooling de una sola query por pooling multi-slot

> **Parcialmente cubierto:** `plan_hier_set_v2.md` implementa multi-head
> attention (H=4), que es la variante más simple de multi-slot. Las ideas
> más avanzadas de esta sección (queries aprendidos por antibiótico, top-k
> attentive pooling) quedan como extensiones futuras.

#### Problema

El modelo actual resume todo el genoma en un solo contexto:

```text
alpha = softmax(scores)
context = sum(alpha_s * h_s)
```

Ese diseño fuerza a mezclar toda la evidencia en un único vector. Si la
resistencia depende de varias regiones distintas, el modelo puede sub-resumir la
señal.

#### Idea

Usar 2 a 4 "slots" o cabezas de pooling, todas condicionadas por el antibiótico:

```text
query_1(a), query_2(a), ..., query_m(a)
context_i = attention(h, query_i)
context = concat(context_1, ..., context_m)
```

Alternativas razonables:

- multi-head attention ligera;
- pooling con varios queries aprendidos por antibiótico;
- `top-k attentive pooling` con `k` pequeño.

#### Impacto esperado

Alto. Permite capturar mecanismos múltiples sin imponer secuencia.

#### Riesgo

Medio. Sube algo la complejidad del clasificador y puede requerir más
regularización.

---

### 3. Sustituir softmax puro por atención con gating tipo MIL

#### Problema

`softmax` obliga a repartir masa total 1 entre segmentos. Eso es útil para
interpretabilidad, pero puede ser restrictivo cuando la evidencia está repartida
en muchos segmentos moderadamente relevantes.

#### Idea

Pasar a un esquema estilo Multiple Instance Learning:

```text
u_s = tanh(W_h h_s + W_a a)
g_s = sigmoid(V_h h_s + V_a a)
score_s = w^T (u_s ⊙ g_s)
alpha = softmax(score_s)
```

O bien una mezcla explícita:

- `attentive context`
- `mean pooling`
- `top-k pooling`

concatenados antes del clasificador.

#### Impacto esperado

Medio a alto. Especialmente útil si la resistencia no vive en un solo "hot spot".

#### Riesgo

Medio. Hay más grados de libertad en el pooling y puede afectar la
interpretabilidad si no se documenta bien.

---

### 4. Hacer el encoder por segmento más profundo

#### Problema

La proyección actual:

```text
Linear(256→128) -> ReLU
```

es probablemente demasiado simple para modelar combinaciones no lineales entre
los bins del histograma.

#### Idea

Cambiar a una MLP residual por segmento:

```text
LayerNorm(256)
Linear(256→192)
GELU/ReLU
Dropout
Linear(192→128)
```

Variantes:

- `256→128→128`
- residual si entrada y salida coinciden tras una proyección
- `GLU` o `GEGLU` si se quiere una versión más expresiva

#### Impacto esperado

Medio. Probablemente ayuda, pero por sí sola no corrige el principal cuello de
botella del modelo.

#### Riesgo

Bajo. Es barata de implementar y probar.

---

### 5. Híbrido local + global (`HierSet + MLP`)

#### Problema

`HierSet` es fuerte para evidencia localizada; `MLP` es fuerte para composición
global del genoma. Hoy se elige uno u otro sesgo.

#### Idea

Combinar ambas vistas:

- rama local: `HierSet` sobre segmentos
- rama global: `MLP` sobre el vector concatenado k=3,4,5
- fusión final con el embedding del antibiótico

Esquema:

```text
context_local = HierSetEncoder(segments, ab)
context_global = MLPEncoder(global_hist)
logit = Classifier([context_local, context_global, ab_emb])
```

#### Impacto esperado

Medio a alto. Esta mezcla ataca un límite real del diseño actual: el pooling por
segmentos puede perder señal difusa o distribuida por todo el genoma.

#### Riesgo

Bajo a medio. Requiere tocar dataset/modelo para consumir dos vistas del genoma.

---

### 6. Self-attention entre segmentos sin recurrencia

#### Problema

Actualmente los segmentos no interactúan entre sí antes del pooling. El modelo
elige segmentos importantes, pero no puede razonar sobre relaciones entre
segmentos.

#### Idea

Añadir 1 o 2 bloques ligeros de self-attention entre segmentos antes del pooling:

```text
segment encoder -> set attention block(s) -> conditioned pooling -> classifier
```

Opciones:

- Set Transformer ligero
- self-attention con 1-2 heads y `D_MODEL` moderado
- attention + residual + FFN, sin positional encoding fuerte

#### Impacto esperado

Medio. Puede capturar co-ocurrencias entre regiones sin caer en el sesgo
secuencial de la BiGRU.

#### Riesgo

Medio a alto. Es la mejora más costosa y la que más fácilmente podría
sobreajustar.

---

### 7. Segment Dropout (regularización de conjuntos)

#### Problema

El modelo puede depender excesivamente de un subconjunto pequeño de segmentos,
especialmente si pocos segmentos contienen genes de resistencia. Esto reduce la
robustez ante genomas con cobertura incompleta o regiones de baja calidad.

#### Idea

Durante entrenamiento, eliminar aleatoriamente un 10-20% de los segmentos del
input antes de la atención:

```python
if self.training:
    mask = torch.rand(B, S, device=h.device) > segment_drop_rate
    # Poner -inf en scores de segmentos dropeados (antes de softmax)
    scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
```

Esto obliga al modelo a encontrar patrones de resistencia redundantes en
múltiples regiones del genoma.

#### Impacto esperado

Bajo a medio. Es data augmentation que respeta la simetría permutation-invariant
del modelo.

#### Riesgo

Muy bajo. Sin parámetros adicionales. Un solo hiperparámetro (`segment_drop_rate`)
que se puede desactivar poniendo 0.

---

### 8. Key/Value Projections separadas

**Referencia:** Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.

#### Problema

En la atención actual, `h` cumple doble función como key Y value:

```text
scores = h · query / √D      ← h es la "key"
context = Σ α_i · h_i        ← h es el "value"
```

Esto fuerza a que lo que determina *cuánta atención recibe* un segmento sea
exactamente lo mismo que *qué información aporta* al contexto.

#### Idea

Agregar proyecciones separadas para keys y values:

```python
self.W_K = nn.Linear(D_MODEL, D_MODEL, bias=False)
self.W_V = nn.Linear(D_MODEL, D_MODEL, bias=False)

keys = self.W_K(h)            # determina cuánta atención recibe
values = self.W_V(h)           # determina qué información aporta
scores = keys · query / √D
context = Σ α_i · values_i
```

Un segmento puede ser "importante para decidir" (key alta) pero contribuir
información diferente al contexto (value que enfatiza otros aspectos).

#### Impacto esperado

Bajo a medio. Mejora probable marginal, pero es una corrección teórica limpia.

#### Riesgo

Bajo. Es una generalización estricta — con W_K = W_V = I, se recupera el modelo
actual. Agrega ~33K parámetros (2 × 128 × 128).

---

### 9. Residual Connection en la atención

#### Problema

Si la atención se concentra excesivamente en pocos segmentos, el contexto pierde
información global del genoma. Toda la composición general se descarta.

#### Idea

Agregar una conexión residual desde la media de los segmentos al contexto:

```python
context_attn = (alpha.unsqueeze(-1) * h).sum(dim=1)   # atención
context_mean = h.mean(dim=1)                            # media simple
context = context_attn + context_mean                   # residual
```

La media actúa como un "respaldo" que garantiza que la composición general del
genoma siempre esté presente.

#### Impacto esperado

Bajo. Puede ayudar si la señal global (composición GC, frecuencia de codones)
complementa la señal local de genes de resistencia.

#### Riesgo

Muy bajo. Sin parámetros adicionales. Puede diluir la señal de atención si el
modelo necesita concentrarse fuertemente en pocos segmentos — verificar con
ablación.

---

### 10. Label Smoothing

**Referencia:** Szegedy, C. et al. (2016). *Rethinking the Inception Architecture
for Computer Vision*. CVPR.

#### Problema

Las etiquetas de resistencia AMR provienen de tests fenotípicos (MIC) con
umbrales de corte clínicos (breakpoints CLSI). Hay incertidumbre inherente
cerca del breakpoint — un organismo con MIC justo en el umbral podría
clasificarse como R o S dependiendo del laboratorio. Targets binarios duros
(0/1) no reflejan esta incertidumbre.

#### Idea

Suavizar los targets:

```python
# target_smooth = target * (1 - epsilon) + epsilon / 2
# Con epsilon = 0.1: 0 → 0.05, 1 → 0.95
```

Le dice al modelo "no estés 100% seguro de ninguna etiqueta", lo cual mejora
calibración y reduce overfitting.

#### Impacto esperado

Bajo a medio. Principalmente mejora calibración de probabilidades, lo cual
beneficia AUC-ROC.

#### Riesgo

Bajo. Modificación en `src/train/loop.py`, no en la arquitectura. Compatible
con `BCEWithLogitsLoss` ajustando targets. Puede afectar la calibración del
umbral óptimo — re-evaluar threshold en validación.

---

### 11. Explorar HIER_N_SEGMENTS

#### Problema

`HIER_N_SEGMENTS = 256` fue elegido sin búsqueda exhaustiva. La granularidad
del segmento afecta directamente la calidad de los histogramas y la resolución
espacial.

#### Contexto cuantitativo

| N_SEGMENTS | Bases/segmento (genoma 3Mbp) | Carácter |
|------------|------------------------------|----------|
| 64         | ~47 Kbp                      | Histogramas estables, baja resolución. Un segmento contiene docenas de genes. |
| 128        | ~23 Kbp                      | Compromiso intermedio. |
| **256**    | **~12 Kbp**                  | **Actual.** Buenos histogramas, resolución moderada. |
| 512        | ~6 Kbp                       | Resolución cercana a gen individual (~1 Kbp). Histogramas más ruidosos. |

#### Idea

Probar N=128 y N=512 con el mismo modelo. No requiere cambios de código: solo
modificar `HIER_N_SEGMENTS` en `constants.py` y re-ejecutar `prepare-hier` +
entrenamiento.

**Nota con multi-escala:** Si se combina con histogramas k=3,4,5, la
sensibilidad al ruido en segmentos pequeños (N=512) podría ser mayor para k=5,
donde el vocab es 1024 y un segmento de 6 Kbp solo produce ~6000 k-meros para
repartir entre 1024 bins.

#### Impacto esperado

Variable. Es puramente un experimento de hiperparámetros.

#### Riesgo

Bajo. Si no mejora, se mantiene 256.

---

## Qué no priorizar primero

### Aumentar solo `D_MODEL`

Puede ayudar marginalmente, pero no ataca el cuello principal. Antes de subir
dimensión conviene mejorar:

- el encoder;
- el pooling;
- la interacción con el antibiótico.

### Volver a introducir secuencia fuerte

La evidencia actual va en la dirección opuesta: `HierBiGRU` rindió peor, lo que
apoya la idea de que el orden entre segmentos no es una señal fiable.

### Cambiar `HIER_N_SEGMENTS` como primera medida

Pasar de 64 a 256 sí tenía una hipótesis clara. Ir más allá antes de mejorar el
encoder parece prematuro. Más segmentos también significan histogramas más
ruidosos y re-ejecutar `prepare-hier`. Sin embargo, vale la pena probar N=128 y
N=512 como experimento de hiperparámetros *después* de las mejoras
arquitectónicas (ver idea 11).

---

## Orden sugerido de experimentos

### Paso 0 — HierSet v2 (ya planificado)

Multi-head attention (H=4) + histogramas multi-escala (k=3,4,5).
Ver `plan_hier_set_v2.md`. Ejecutar primero y evaluar resultados.

### Paso 1 — Mejoras de costo muy bajo sobre v2

Aplicables como ablaciones rápidas sin cambios estructurales mayores:

1. Segment Dropout (idea 7) — 0 parámetros extra, un hiperparámetro
2. Label Smoothing (idea 10) — cambio en el loop, no en el modelo
3. Residual Connection (idea 9) — 0 parámetros extra
4. N_SEGMENTS search (idea 11) — solo regenerar datos y reentrenar

### Paso 2 — Mejoras arquitectónicas moderadas

1. FiLM Conditioning (idea 1) — ~12.5K params
2. Key/Value Projections (idea 8) — ~33K params
3. Encoder más profundo por segmento (idea 4)

### Paso 3 — Cambios de paradigma

1. Gated Attention MIL (idea 3)
2. Híbrido HierSet + MLP (idea 5)
3. Self-attention entre segmentos / Set Transformer (idea 6)

---

## Variante concreta de `HierSet v3` (más allá del plan v2)

> **v2 ya está planificado** en `plan_hier_set_v2.md` con multi-head attention
> + multi-escala (k=3,4,5). Esta sección describe una hipotética v3 que
> combinaría las mejores ideas de este documento.

Si hubiera que elegir una sola variante razonable para implementar después de v2:

```text
segment [256, 1344] -> LayerNorm(1344)
        -> Linear(1344→192)
        -> GELU
        -> Dropout
        -> Linear(192→128)
        -> FiLM(ab_emb)                               ← idea 1
        -> segment dropout (10%)                       ← idea 7
        -> gated multi-head attention (4 heads, MIL)   ← ideas 2+3
        -> residual + mean pooling                     ← idea 9
        -> concat([context, ab_emb])
        -> classifier
```

Motivo:

- mantiene la idea central de conjunto;
- aprovecha los histogramas multi-escala de v2;
- mejora representación local (encoder más profundo);
- mejora la interacción segmento-antibiótico (FiLM);
- permite resumir más de una región relevante (multi-head);
- regularización por segment dropout;
- sigue siendo implementable sin tocar `prepare-hier-multi`.

---

## Protocolo de evaluación recomendado

Para evitar conclusiones engañosas, las mejoras de `HierSet` deberían compararse
con este orden de prioridad:

1. `AUC-ROC`
2. Curva Precision-Recall / `AUPRC` si se agrega
3. `F1`
4. `Recall` a threshold calibrado

Notas:

- `Recall` aislado puede mejorar solo bajando el umbral; eso no implica mejor arquitectura.
- Los cambios de arquitectura deben juzgarse por la calidad del ranking (`AUC`) y por la frontera Precision-Recall.
- Si el objetivo clínico prioriza sensibilidad, conviene reportar también un punto operativo del tipo:
  `max recall sujeto a precision >= 0.85`.

---

## Conclusión

`HierSet` ya validó la hipótesis importante del proyecto: para histogramas
segmentados de genomas draft, tratar los segmentos como conjunto funciona mejor
que imponerles una secuencia.

El siguiente paso concreto es **v2** (`plan_hier_set_v2.md`): multi-head
attention + histogramas multi-escala. Si v2 mejora sobre v1, las ideas de este
documento (especialmente segment dropout, FiLM y gated attention) son
extensiones naturales.

El salto cualitativo probablemente no vendrá de volver a RNNs, sino de hacer
tres cosas mejor:

- representar mejor cada segmento (encoder más profundo, multi-escala);
- condicionar antes y más fuerte en el antibiótico (FiLM);
- resumir múltiples fuentes de evidencia sin colapsarlas en un único contexto
  (multi-head, gated attention, residual).
