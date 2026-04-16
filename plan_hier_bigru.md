# Plan: Hierarchical BiGRU (histogramas segmentados)

## Contexto

El Token BiGRU fue diseñado para capturar el orden de las bases del genoma, pero su implementación actual usa `numpy.linspace` para subsampling uniforme: de ~2M k-meros, solo selecciona 4096 (~0.2% del genoma). Los genes de resistencia son un porcentaje pequeño del genoma y pueden perderse completamente con este subsampling.

La solución es un **Hierarchical BiGRU** que divida el genoma en 64 segmentos geográficos contiguos, calcule el histograma de k-meros de cada segmento (cobertura del 100%, sin subsampling), y procese la secuencia de histogramas con una BiGRU + atención. La atención aprende a enfocarse en los segmentos que contienen genes de resistencia.

Se usa histogramas por segmento (no tokens) porque garantiza cobertura total. Los tokens solo cubren un fragmento del genoma sin importar el tamaño de ventana, no resuelven el problema.

---

## Arquitectura

```
Input: [batch, 64, 256]   ← 64 segmentos × histograma k=4 normalizado

BiGRU(input=256, hidden=128, layers=2, bidireccional, dropout=0.3)
  → [batch, 64, 256]

BahdanauAttention(hidden=256, attn_dim=128)   ← reutilizada de bigru/model.py
  → context [batch, 256]  +  alpha [batch, 64]

Concat(antibiotic_embedding[49])
  → [batch, 305]

Linear(305→128) + ReLU + Dropout(0.3) + Linear(128→1)
  → logit [batch, 1]
```

Parámetros estimados: ~372K (comparable al Token BiGRU).

---

## Archivos a crear

| Archivo | Descripción |
|---|---|
| `src/models/hier_bigru/__init__.py` | Vacío |
| `src/models/hier_bigru/model.py` | `AMRHierBiGRU` con factory method |
| `src/models/hier_bigru/dataset.py` | `HierBiGRUDataset(BaseAMRDataset)` |

---

## Archivos a modificar

### 1. `src/data_pipeline/constants.py`
Agregar al final:
```python
HIER_KMER_K = 4               # k-mero para cada segmento (vocab=256)
HIER_KMER_DIM = 4 ** HIER_KMER_K  # 256
HIER_N_SEGMENTS = 64          # segmentos geográficos del genoma
```

### 2. `src/data_pipeline/features.py`
Agregar método a `KmerExtractor`:
```python
def to_tiled_histogram_matrix(
    self, k=HIER_KMER_K, n_segments=HIER_N_SEGMENTS
) -> numpy.ndarray:  # shape (64, 256), float32
```
Lógica:
1. `_read_sequences()` → concatenar todos los contigs en un string
2. Dividir `total_len` en `n_segments` segmentos iguales (último absorbe remanente)
3. Por segmento: rolling hash → histograma → dividir por suma (frecuencia relativa)
4. Retornar matriz float32 `(n_segments, 4^k)`

Nota: si un segmento suma 0 (solo Ns), dejar en ceros.

### 3. `src/data_pipeline/pipeline.py`
Agregar (siguiendo el patrón de `_extract_and_save_tokens`):
```python
def _extract_single_genome_hier(genome_id, fasta_dir):  # picklable, nivel módulo
    ...

def extract_and_save_hier(genome_ids, fasta_dir, output_dir, n_jobs=1):
    # Guarda en data/processed/hier_bigru/{genome_id}.npy
    ...
```

### 4. `main.py`
Agregar imports:
```python
from models.hier_bigru.dataset import HierBiGRUDataset
from models.hier_bigru.model import AMRHierBiGRU
from data_pipeline.pipeline import extract_and_save_hier
from data_pipeline.constants import HIER_KMER_K, HIER_N_SEGMENTS
```

Agregar dos comandos:
- `prepare_hier()` — extrae y guarda los `.npy` de histogramas segmentados
- `train_hier_bigru()` — entrena el modelo con los defaults de abajo

---

## Hiperparámetros

| Parámetro | Valor | Razón |
|---|---|---|
| `HIER_N_SEGMENTS` | 64 | ~31K bp/segmento → cada gen de resistencia (500-3000 bp) queda dentro de 1-2 segmentos |
| `HIER_KMER_K` | 4 | vocab=256, balance entre especificidad y densidad estadística por segmento |
| `GRU_HIDDEN` | 128 | Consistente con todos los modelos del proyecto [Lugo21] |
| `GRU_LAYERS` | 2 | Necesario para activar dropout recurrente [Srivastava14] |
| `lr` | 0.001 | Default del proyecto para BiGRUs |
| `patience` | 15 | Ligeramente mayor que otros modelos; input más rico puede tardar más en converger |
| `pos_weight_scale` | 2.5 | Mismo que BiGRU base |
| `weight_decay` | 1e-4 | Mismo que Token BiGRU |
| `max_grad_norm` | 1.0 | Estándar para RNNs [Pascanu13] |

---

## Orden de implementación

1. Constantes en `constants.py`
2. Método `to_tiled_histogram_matrix()` en `features.py`
3. Funciones de pipeline en `pipeline.py`
4. `hier_bigru/__init__.py`, `dataset.py`, `model.py`
5. Comandos `prepare_hier` y `train_hier_bigru` en `main.py`

Nota: `prepare-hier` **no requiere re-ejecutar** `prepare-data` — lee directamente los FASTAs.

---

## Verificación

```bash
# 1. Extraer features
python main.py prepare-hier

# 2. Verificar shape de un .npy generado
python -c "
import numpy as np
m = np.load('data/processed/hier_bigru/<genome_id>.npy')
assert m.shape == (64, 256)
assert m.sum(axis=1).max() <= 1.001  # filas normalizadas
print('OK:', m.shape, m.dtype)
"

# 3. Sanity check del forward pass
python -c "
import torch
from src.models.hier_bigru.model import AMRHierBiGRU
model = AMRHierBiGRU(n_antibiotics=96)
genome = torch.randn(4, 64, 256)
ab_idx = torch.randint(0, 96, (4,))
logits = model(genome, ab_idx)
assert logits.shape == (4, 1)
print('Forward OK:', logits.shape)
"

# 4. Entrenar
python main.py train-hier-bigru

# 5. Comparar con modelos existentes
# → results/hier_bigru/metrics.json vs results/multi_bigru/metrics.json
```

Criterio de éxito: AUC-ROC ≥ 0.900 (nivel de multi_bigru y MLP).
Criterio de mejora: AUC-ROC > 0.904.
