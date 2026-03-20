# Entorno y uso

## Stack
- **Lenguaje:** Python 3.13
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
├── src/                  # Código fuente — ver src/README.md para detalle de cada módulo
│   ├── bvbrc/            # Cliente HTTP para la API REST de BV-BRC
│   ├── data_pipeline/    # Preprocesamiento: etiquetas, k-meros, splits
│   ├── eda.py            # Análisis exploratorio (EDA)
│   ├── models.py         # Definición de MLP y BiGRU+Attention
│   └── train.py          # Loop de entrenamiento y evaluación
├── tests/
│   ├── bvbrc/            # Unit tests del paquete bvbrc
│   └── data_pipeline/    # Unit tests del data pipeline
├── data/
│   ├── raw/              # Genomas FASTA descargados
│   └── processed/        # Etiquetas CSV, vectores k-meros (.npy)
├── results/              # Métricas, gráficas, checkpoints
├── notebooks/            # Exploración y visualización
├── docs/
│   ├── proposal/         # Propuesta del proyecto
│   ├── reference/        # Referencia técnica (API, etc.)
│   └── 1–5_*.md          # Documentos de diseño e implementación
└── main.py               # CLI (Typer) — punto de entrada
```

> Índice detallado de `src/` con descripción de cada archivo y flujo de datos: [`src/README.md`](../src/README.md)

## Instalación

**Requisitos:** Python 3.13 (gestionado automáticamente por uv), [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
git clone <url-del-repo>
cd proyecto_redes_neuronales

# 1. Crear el entorno virtual (opcional si usas 'uv sync' directamente)
uv venv

# 2. Sincronizar dependencias (crea el .venv si no existe)
uv sync
```

### Gestión del entorno virtual con `uv`

- **Creación:** Al ejecutar `uv venv`, `uv` detecta automáticamente el archivo `.python-version` (que actualmente especifica la versión **3.13**) y crea un entorno virtual en la carpeta `.venv`.
- **Opciones comunes:**
  - Especificar versión: `uv venv --python 3.12`
  - Nombre personalizado: `uv venv mi_entorno`
- **Activación:**
  - **Linux/macOS:** `source .venv/bin/activate`
  - **Windows:** `.venv\Scripts\activate`
- **Sincronización:** `uv sync` asegura que las librerías instaladas coincidan exactamente con lo definido en `uv.lock`.
- **Ejecución sin activación:** Puedes correr comandos directamente sin activar el entorno usando `uv run`:
  ```bash
  uv run python main.py --help
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
| `export-contradictions-cmd` | Exporta pares (genome_id, antibiotic) con etiquetas contradictorias a CSV | `docs/2_eda.md` |

## Tests

```bash
# Correr todos los tests
uv run pytest

# Con output detallado
uv run pytest -v

# Un módulo específico
uv run pytest tests/bvbrc/test_amr.py -v
```
