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
│   ├── usage.md          # Instalación, comandos CLI y tests
│   └── 1–5_*.md          # Decisiones de diseño e implementación
├── docs/PROGRESS.md      # Estado de implementación por fases
└── docs/PROGRESS.md      # Plan, estado y changelog
```
