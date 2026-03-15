# Entorno y uso

## Stack
- **Lenguaje:** Python 3.10+
- **Framework de deep learning:** PyTorch
- **Gestor de entorno:** uv
- **Librerías principales:**
  - `biopython` — parseo de archivos FASTA
  - `numpy`, `pandas` — manipulación de datos
  - `scikit-learn` — métricas, splits estratificados, class weights
  - `torch` — modelos y entrenamiento
  - `matplotlib` / `seaborn` — visualización de resultados

## Estructura de carpetas

```
proyecto_redes_neuronales/
├── src/
│   ├── bvbrc/            # Descarga de datos desde BV-BRC
│   │   ├── _http.py      #   Utilidades HTTP compartidas
│   │   ├── amr.py        #   Etiquetas AMR (genome_amr endpoint)
│   │   └── genomes.py    #   Genomas FASTA (genome_sequence endpoint)
│   ├── data_pipeline.py  # Preprocesamiento: etiquetas, k-meros, splits
│   ├── models.py         # Definición de MLP y BiGRU+Attention
│   └── train.py          # Loop de entrenamiento y evaluación
├── tests/
│   └── bvbrc/            # Unit tests del paquete bvbrc
├── data/
│   ├── raw/              # Genomas FASTA descargados
│   └── processed/        # Etiquetas CSV, vectores k-meros (.npy)
├── results/              # Métricas, gráficas, checkpoints
├── notebooks/            # Exploración y visualización
├── docs/
│   ├── proposal/         # Propuesta del proyecto
│   ├── reference/        # Referencia técnica (API, etc.)
│   └── 1–5_*.md          # Documentos de diseño e implementación
├── PROGRESS.md           # Estado de implementación por fases
└── CHANGELOG.md          # Registro de cambios
```

## Instalación

**Requisitos:** Python 3.10+, [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
git clone <url-del-repo>
cd proyecto_redes_neuronales
uv sync
```

## CLI

El punto de entrada es `main.py`, un CLI basado en Typer. Cada fase del pipeline tiene su propio subcomando:

```bash
uv run python main.py --help
```

| Comando | Descripción | Documentación |
|---|---|---|
| `download-amr` | Descarga etiquetas AMR de BV-BRC para organismos ESKAPE | `docs/3_data_pipeline.md` |
| `download-genomes` | Descarga archivos FASTA de los genomas del CSV de etiquetas | `docs/3_data_pipeline.md` |
| `eda` | Análisis exploratorio: distribución, balance, outliers, baseline benchmark | `docs/2_eda.md` |

## Tests

```bash
# Correr todos los tests
uv run pytest

# Con output detallado
uv run pytest -v

# Un módulo específico
uv run pytest tests/bvbrc/test_amr.py -v
```
