# Predicción de resistencia antimicrobiana con redes neuronales

Proyecto académico: clasificación binaria de resistencia/susceptibilidad antibiótica en bacterias del grupo ESKAPE a partir de secuencias genómicas completas (WGS), usando un MLP baseline y un modelo BiGRU + Attention.

---

## Requisitos

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (gestor de paquetes)

## Instalación

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd proyecto_redes_neuronales

# Instalar dependencias
uv sync
```

---

## Uso

> **Nota:** El pipeline completo se irá documentando conforme se implementen las fases.
> Ver `PROGRESS.md` para el estado actual.

```bash
uv run python main.py --help
```

### 1. Descargar etiquetas AMR

Descarga las etiquetas de resistencia/susceptibilidad para todos los organismos ESKAPE desde BV-BRC y las guarda como CSV.

```bash
uv run python main.py download-amr
# Ruta personalizada:
uv run python main.py download-amr --output data/processed/amr_labels.csv
```

### 2. Descargar genomas FASTA

Descarga genomas FASTA para todos los genome_id del CSV de etiquetas. Ejecutar después del paso 1.

```bash
uv run python main.py download-genomes
# Rutas personalizadas:
uv run python main.py download-genomes --labels data/processed/amr_labels.csv --output-dir data/raw/fasta
# Muestra de N genomas por especie (estratificada por fenotipo):
uv run python main.py download-genomes --sample-per-species 20 --output-dir data/raw/fasta_sample
```

### 3. Análisis exploratorio (EDA)

Muestra distribución por especie, balance de clases, ranking de antibióticos, calidad de datos, outliers, baseline benchmark y análisis de secuencias genómicas.

```bash
uv run python main.py eda --genomes-dir data/raw/fasta_sample
# Ruta personalizada al CSV:
uv run python main.py eda --genomes-dir data/raw/fasta_sample --labels data/processed/amr_labels.csv
# Mostrar más antibióticos:
uv run python main.py eda --genomes-dir data/raw/fasta_sample --top-n 30
```

---

## Tests

```bash
# Correr todos los tests
uv run pytest

# Con output detallado
uv run pytest -v

# Un módulo específico
uv run pytest tests/bvbrc/test_amr.py -v
```

---

## Estructura del proyecto

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
│   └── implementation/   # Documentación de diseño e implementación
├── PROGRESS.md           # Estado de implementación por fases
└── CHANGELOG.md          # Registro de cambios
```

---

## Documentación

- `PROGRESS.md` — plan de implementación y estado actual
- `docs/implementation/` — decisiones de diseño, arquitectura y experimentos
- `docs/implementation/bvbrc_api.md` — referencia de la API de BV-BRC
