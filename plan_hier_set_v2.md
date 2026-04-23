# Plan de Mejora — HierSet v2: Multi-Head Attention + Histogramas Multi-Escala

## Contexto y Motivación

El modelo HierSet actual (F1=0.89, AUC=0.9368) es el mejor del proyecto. Este plan propone dos mejoras ortogonales para intentar superar esa marca:

1. **Multi-Head Cross-Attention:** El modelo actual usa una sola cabeza de atención — un solo patrón de pesos sobre los 256 segmentos por cada antibiótico. Esto obliga al modelo a comprimir múltiples mecanismos de resistencia (genes de resistencia, bombas de eflujo, modificación del target) en una sola distribución de atención. Con H cabezas, cada una puede especializarse en detectar un mecanismo distinto.

2. **Histogramas multi-escala (k=3,4,5) por segmento:** HierSet solo usa k=4 (256 dims por segmento). El MLP ya demostró que la información multi-escala es valiosa (k=3 para composición nucleotídica general, k=5 para motivos específicos). Extender cada segmento a k=3+4+5 (1344 dims) da al modelo más información por segmento sin cambiar la lógica de atención.

Las dos mejoras son independientes entre sí: multi-head cambia cómo se agrega la información, multi-escala cambia qué información entra. Se implementan juntas en un solo modelo actualizado.

### Métricas de éxito

| Métrica | HierSet v1 (actual) | Objetivo v2 |
|---------|---------------------|-------------|
| F1      | 0.8900              | ≥ 0.8950    |
| AUC-ROC | 0.9368              | ≥ 0.9400    |
| Recall  | 0.9088              | ≥ 0.9000    |

Si v2 no supera v1, se reporta como resultado negativo — el modelo actual se mantiene.

---

## Referencias bibliográficas

| Ref. | Cita | Relevancia |
|------|------|------------|
| [Vaswani17] | Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS. | Multi-head attention: múltiples subespacios de atención permiten capturar distintos tipos de relación. |
| [Ilse18] | Ilse, M. et al. (2018). *Attention-based Deep Multiple Instance Learning*. ICML. | Formulación MIL con attention pooling para instancias en una bolsa — análogo a segmentos en un genoma. |
| [Lee19] | Lee, J. et al. (2019). *Set Transformer*. ICML. | Arquitecturas de atención para conjuntos. Contexto teórico para permutation-invariant encoders. |
| [Haykin09] | Haykin, S. (2009). *Neural Networks and Learning Machines*, 3ª ed. | Representación del conocimiento (Cap. 7), regularización (Cap. 4.14). |

---

## Arquitectura actual vs. propuesta

### HierSet v1 (actual)

```
Input [batch, 256, 256]                    ← 256 segmentos × k=4
  → LayerNorm(256)
  → Linear(256→128) + ReLU + Dropout(0.3)  ← proyección independiente
  → Cross-attention (1 cabeza):
      query = Linear(49→128)(ab_emb)
      scores = h · query / √128
      context = softmax(scores) · h        ← [batch, 128]
  → Concat(context, ab_emb)               ← [batch, 177]
  → Linear(177→128) + ReLU + Dropout(0.3)
  → Linear(128→1)
```

### HierSet v2 (propuesta)

```
Input [batch, 256, 1344]                   ← 256 segmentos × k=3+4+5
  → LayerNorm(1344)
  → Linear(1344→128) + ReLU + Dropout(0.3) ← proyección independiente
  → Multi-Head Cross-attention (H=4, d_head=32):
      query = Linear(49→128)(ab_emb)          → reshape [batch, 4, 32]
      h reshaped                              → [batch, 256, 4, 32]
      scores_per_head = h · query / √32       → [batch, 4, 256]
      context_per_head = softmax(scores) · h  → [batch, 4, 32]
      context = concat(heads)                 → [batch, 128]
  → Concat(context, ab_emb)                   ← [batch, 177]
  → Linear(177→128) + ReLU + Dropout(0.3)
  → Linear(128→1)
```

**Parámetros adicionales:** Solo cambia la proyección de entrada (de 256×128 a 1344×128 = +139K params). La multi-head attention no agrega parámetros respecto a single-head (misma `Linear(49→128)`, solo cambia el cómputo).

---

## Fases de implementación

### Fase A — Pipeline de datos: histogramas multi-escala

**Objetivo:** Generar archivos .npy con shape `(256, 1344)` en vez de `(256, 256)`.

#### A.1. Nuevas constantes (`src/data_pipeline/constants.py`)

Agregar:

```python
# Histogramas multi-escala por segmento (k=3,4,5)
HIER_KMER_SIZES = [3, 4, 5]
HIER_KMER_DIM_MULTI = sum(4**k for k in HIER_KMER_SIZES)  # 64 + 256 + 1024 = 1344
```

No modificar las constantes existentes (`HIER_KMER_K`, `HIER_KMER_DIM`) — las usa HierBiGRU y HierSet v1.

#### A.2. Nuevo método en `KmerExtractor` (`src/data_pipeline/features.py`)

Agregar método `to_tiled_multiscale_matrix`:

```python
def to_tiled_multiscale_matrix(
    self,
    kmer_sizes: list[int] = HIER_KMER_SIZES,
    n_segments: int = HIER_N_SEGMENTS,
) -> numpy.ndarray:
    """Divide el genoma en n_segments y calcula histogramas k=3,4,5 por segmento.

    Extiende to_tiled_histogram_matrix con múltiples escalas de k-meros.
    Cada segmento produce un vector de 1344 dims (64 + 256 + 1024) que
    captura composición nucleotídica (k=3), motivos cortos (k=4) y
    motivos largos (k=5) simultáneamente.

    Retorna:
        numpy.ndarray de shape (n_segments, 1344) con dtype float32.
    """
```

**Lógica interna:**
1. Concatenar contigs con separador `N * (max(kmer_sizes) - 1)` = `NNNN` (4 N's para k=5).
2. Calcular `total_len` y `segment_size = total_len // n_segments`.
3. Para cada segmento, para cada k en `kmer_sizes`: calcular histograma con `_count_kmers`, normalizar, y concatenar.
4. Retornar matriz `(n_segments, 1344)`.

**Nota sobre el separador:** Actualmente `to_tiled_histogram_matrix` usa `N * (k-1)`. Con múltiples k, el separador debe ser `N * (max_k - 1)` = 4 N's para que ningún k-mero cruce fronteras entre contigs. Esto es una mejora respecto a v1 donde k=4 usaba solo 3 N's.

#### A.3. Nuevo worker en pipeline (`src/data_pipeline/pipeline.py`)

```python
def _extract_single_genome_hier_multi(
    genome_id: str, fasta_dir: Path
) -> tuple[str, numpy.ndarray]:
    """Extrae histogramas multi-escala segmentados."""
    extractor = KmerExtractor(fasta_dir / f"{genome_id}.fna")
    return genome_id, extractor.to_tiled_multiscale_matrix()
```

Y nueva función pública `extract_and_save_hier_multi` que guarda en `data/processed/hier_set_v2/`:

```python
def extract_and_save_hier_multi(
    genome_ids: list[str],
    fasta_dir: Path,
    output_dir: Path,
    n_jobs: int = 1,
) -> None:
    """Extrae histogramas multi-escala segmentados y los guarda en hier_set_v2/."""
```

Misma estructura paralela que `extract_and_save_hier`. El directorio de salida es `hier_set_v2/` para no sobreescribir los datos de v1.

#### A.4. Nuevo comando CLI (`main.py`)

```python
@app.command(help="Extrae histogramas multi-escala segmentados (k=3,4,5) para HierSet v2.")
def prepare_hier_multi(
    data_dir: Path = ...,
    fasta_dir: Path = ...,
    n_jobs: int = ...,
):
```

Invocación: `uv run python main.py prepare-hier-multi --n-jobs -1`

#### A.5. Tests (`tests/data_pipeline/test_features.py`)

Agregar test para `to_tiled_multiscale_matrix`:
- Verificar shape de salida: `(HIER_N_SEGMENTS, 1344)`
- Verificar que cada fila suma ~3.0 (tres histogramas normalizados a 1.0 cada uno)
- Verificar que los primeros 64 dims corresponden a k=3, los siguientes 256 a k=4, los últimos 1024 a k=5

---

### Fase B — Modelo: Multi-Head Cross-Attention + input multi-escala

**Objetivo:** Nuevo modelo `AMRHierSetV2` que acepta input `(batch, 256, 1344)` y usa multi-head attention.

#### B.1. Nuevos archivos del modelo

Crear directorio `src/models/hier_set_v2/` con:

| Archivo | Contenido |
|---------|-----------|
| `__init__.py` | (vacío) |
| `model.py` | `AMRHierSetV2` |
| `dataset.py` | `HierSetV2Dataset` — carga desde `hier_set_v2/` con shape `(256, 1344)` |

#### B.2. Modelo (`src/models/hier_set_v2/model.py`)

Hiperparámetros:

```python
D_MODEL = 128
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS  # 32
CLASSIFIER_HIDDEN = 128
DROPOUT = 0.3
```

Clase `AMRHierSetV2(nn.Module)`:

```python
def __init__(self, n_antibiotics: int) -> None:
    # Embedding de antibiótico
    self.antibiotic_embedding = nn.Embedding(n_antibiotics, ANTIBIOTIC_EMBEDDING_DIM)

    # Normalización del histograma multi-escala por segmento
    self.norm = nn.LayerNorm(HIER_KMER_DIM_MULTI)  # 1344

    # Proyección independiente por segmento
    self.proj = nn.Linear(HIER_KMER_DIM_MULTI, D_MODEL)  # 1344 → 128
    self.proj_dropout = nn.Dropout(DROPOUT)

    # Multi-head cross-attention: una sola Linear genera la query,
    # que se reshapea a [batch, N_HEADS, D_HEAD]
    self.attn_query = nn.Linear(ANTIBIOTIC_EMBEDDING_DIM, D_MODEL, bias=False)

    # Clasificador (misma arquitectura que v1)
    self.classifier = nn.Sequential(
        nn.Linear(D_MODEL + ANTIBIOTIC_EMBEDDING_DIM, CLASSIFIER_HIDDEN),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(CLASSIFIER_HIDDEN, 1),
    )

    # Pesos de atención para interpretabilidad
    self._attention_weights: torch.Tensor | None = None

@classmethod
def from_antibiotic_index(cls, path: str) -> "AMRHierSetV2":
    """Factory method para instanciación desde la CLI."""
    return cls(n_antibiotics=len(pandas.read_csv(path)))
```

**Forward — cambio clave (multi-head attention):**

```python
def forward(self, genome, antibiotic_idx):
    # 1-2. Normalización y proyección (igual que v1 pero con 1344 dims de entrada)
    x = self.norm(genome)                                    # [batch, S, 1344]
    h = self.proj_dropout(F.relu(self.proj(x)))              # [batch, S, 128]

    # 3. Multi-head cross-attention
    B, S, _ = h.shape
    ab_emb = self.antibiotic_embedding(antibiotic_idx)       # [batch, 49]
    query = self.attn_query(ab_emb)                          # [batch, 128]

    # Reshape para H cabezas
    query = query.view(B, N_HEADS, D_HEAD)                   # [batch, 4, 32]
    h_heads = h.view(B, S, N_HEADS, D_HEAD)                  # [batch, S, 4, 32]

    # Scores por cabeza: [batch, 4, S]
    scores = torch.einsum('bshd,bhd->bhs', h_heads, query) / math.sqrt(D_HEAD)
    alpha = F.softmax(scores, dim=2)                         # [batch, 4, S]

    # Contexto por cabeza: [batch, 4, 32]
    context_heads = torch.einsum('bhs,bshd->bhd', alpha, h_heads)

    # Concatenar cabezas: [batch, 128]
    context = context_heads.reshape(B, D_MODEL)

    # Guardar pesos de atención (promedio de cabezas para interpretabilidad)
    self._attention_weights = alpha.mean(dim=1).detach()     # [batch, S]

    # 4. Fusión y clasificación
    return self.classifier(torch.cat([context, ab_emb], dim=1))
```

**Justificación de H=4:**
- `D_MODEL=128` se divide limpiamente en 4 cabezas de 32 dims.
- 4 cabezas es un buen punto de partida dado que hay ~4 mecanismos principales de resistencia AMR (enzimas inactivadoras, modificación del target, bombas de eflujo, impermeabilidad).
- No agrega parámetros respecto a v1: la misma `Linear(49→128)` se usa para la query, solo cambia el reshape y el cómputo de scores.

#### B.3. Dataset (`src/models/hier_set_v2/dataset.py`)

```python
class HierSetV2Dataset(BaseAMRDataset):
    """Dataset para AMRHierSetV2. Carga histogramas multi-escala."""

    def _load_genome_data(self, data_dir, split_ids):
        hier_dir = data_dir / "hier_set_v2"
        # Carga .npy con shape (HIER_N_SEGMENTS, HIER_KMER_DIM_MULTI)
        # Validación: shape == (256, 1344)
```

#### B.4. Tests (`tests/models/test_hier_set_v2.py`)

- Test de forward pass con input sintético `(2, 256, 1344)` → output `(2, 1)`
- Test de que `_attention_weights` tiene shape `(2, 256)` (promedio de cabezas)
- Test del dataset con datos mock

---

### Fase C — Entrenamiento y evaluación

#### C.1. Comando CLI (`main.py`)

```python
@app.command(help="Entrena el HierSet v2 (multi-head + multi-escala).")
def train_hier_set_v2(
    data_dir: Path = ...,
    output_dir: Path = typer.Option(Path("results/hier_set_v2"), ...),
    epochs: int = typer.Option(100, ...),
    batch_size: int = typer.Option(32, ...),
    lr: float = typer.Option(0.001, ...),
    patience: int = typer.Option(15, ...),
    lr_patience: int = typer.Option(5, ...),
    pos_weight_scale: float = typer.Option(2.5, ...),
    weight_decay: float = typer.Option(1e-3, ...),
):
```

Mismos hiperparámetros de entrenamiento que v1 para una comparación justa.

#### C.2. Entrenamiento

```bash
# 1. Generar datos multi-escala (una sola vez):
uv run python main.py prepare-hier-multi --n-jobs -1

# 2. Entrenar:
uv run python main.py train-hier-set-v2
```

#### C.3. Evaluación

Comparar métricas de test contra v1. Reportar en `docs/4_models.md` y `docs/CHANGELOG.md`.

---

## Archivos a crear/modificar

| Archivo | Acción | Fase |
|---------|--------|------|
| `src/data_pipeline/constants.py` | Modificar — agregar `HIER_KMER_SIZES`, `HIER_KMER_DIM_MULTI` | A.1 |
| `src/data_pipeline/features.py` | Modificar — agregar `to_tiled_multiscale_matrix` | A.2 |
| `src/data_pipeline/pipeline.py` | Modificar — agregar `_extract_single_genome_hier_multi`, `extract_and_save_hier_multi` | A.3 |
| `src/data_pipeline/__init__.py` | Modificar — exportar `extract_and_save_hier_multi` | A.3 |
| `main.py` | Modificar — agregar `prepare-hier-multi`, `train-hier-set-v2` | A.4, C.1 |
| `tests/data_pipeline/test_features.py` | Modificar — agregar test de `to_tiled_multiscale_matrix` | A.5 |
| `src/models/hier_set_v2/__init__.py` | Nuevo | B.1 |
| `src/models/hier_set_v2/model.py` | Nuevo — `AMRHierSetV2` | B.2 |
| `src/models/hier_set_v2/dataset.py` | Nuevo — `HierSetV2Dataset` | B.3 |
| `tests/models/test_hier_set_v2.py` | Nuevo | B.4 |

**Archivos que NO se modifican:** Todo lo existente de HierSet v1 permanece intacto como baseline.

---

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| Overfitting por más parámetros en la proyección (1344×128 vs 256×128) | Mismo dropout=0.3, mismo weight_decay=1e-3. Monitorear brecha train/val loss. Si crece, considerar aumentar weight_decay a 5e-3. |
| Histogramas k=5 ruidosos en segmentos cortos (cada segmento ≈ 10-20 Kbp, k=5 tiene vocab=1024) | La normalización a frecuencia relativa mitiga esto. Si los k=5 dominan por ser 1024 dims vs 64+256, la proyección lineal aprenderá a escalarlos. |
| `prepare-hier-multi` tarda mucho más que `prepare-hier` (3 histogramas vs 1) | Usar `--n-jobs -1` para paralelizar. El cuello de botella es I/O de FASTA, no el cómputo de histogramas — el overhead debería ser ~2x, no 3x. |
| Multi-head no mejora (H=1 era suficiente) | Con H=4 y d_head=32, el modelo v2 es una generalización estricta de v1 — si H=1 era óptimo, el modelo puede aprender a usar solo una cabeza efectiva. No hay pérdida de capacidad. |

---

## Orden de ejecución

1. **Fase A** (pipeline de datos) — primero, porque genera los .npy que la Fase B necesita
2. **Fase B** (modelo) — se puede implementar en paralelo con A.4-A.5
3. **Tests** — ejecutar tests unitarios de A.5 y B.4 antes de entrenar
4. **Fase C** — entrenar y evaluar
5. **Documentación** — actualizar `docs/4_models.md`, `docs/CHANGELOG.md`, `AGENTS.md`, `docs/PROGRESS.md`
