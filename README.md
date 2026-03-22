# Predicción de resistencia antimicrobiana con redes neuronales

Proyecto académico: clasificación binaria de resistencia/susceptibilidad antibiótica en bacterias del grupo ESKAPE a partir de secuencias genómicas completas (WGS), usando un MLP baseline y un modelo BiGRU + Attention.

---

## Uso e instalación

Ver [`docs/1_environment.md`](docs/1_environment.md) para instrucciones de instalación, comandos CLI y cómo correr los tests.

---

## Estructura del proyecto

```
proyecto_redes_neuronales/
├── src/
│   ├── bvbrc/            # Descarga de datos desde BV-BRC
│   │   ├── _http.py      #   Utilidades HTTP compartidas
│   │   ├── amr.py        #   Etiquetas AMR (genome_amr endpoint)
│   │   └── genomes.py    #   Genomas FASTA (genome_sequence endpoint)
│   ├── data_pipeline/    # Preprocesamiento: etiquetas, k-meros, splits
│   ├── train/            # Paquete de entrenamiento: loop, evaluate
│   ├── dataset.py        # AMRDataset — Dataset de PyTorch
│   ├── mlp_model.py      # AMRMLP — Perceptrón multicapa
│   ├── eda.py            # Análisis exploratorio
│   └── main.py           # CLI (Typer) — punto de entrada
├── tests/
│   ├── bvbrc/            # Tests del cliente BV-BRC
│   └── data_pipeline/    # Tests del pipeline de datos
├── data/
│   ├── raw/              # Genomas FASTA descargados
│   └── processed/        # Etiquetas CSV, vectores k-meros (.npy)
├── results/              # Métricas, gráficas, checkpoints
├── notebooks/            # Exploración y visualización
├── docs/
│   ├── proposal/         # Propuesta del proyecto
│   ├── reference/        # Referencia técnica (API, etc.)
│   ├── PROGRESS.md       # Plan de implementación y estado por fases
│   ├── CHANGELOG.md      # Registro histórico de cambios
│   └── 1–5_*.md          # Decisiones de diseño e implementación
└── main.py               # Punto de entrada CLI
```
