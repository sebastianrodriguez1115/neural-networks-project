# Environment

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

## Estructura de carpetas (propuesta)
```
proyecto_redes_neuronales/
├── implementation/       # archivos de planeación
├── data/
│   ├── raw/              # FASTA descargados de PATRIC
│   └── processed/        # k-mers extraídos, etiquetas, splits
├── src/
│   ├── data_pipeline.py  # descarga, parseo, extracción de k-meros
│   ├── models.py         # definición de MLP y BiRNN
│   └── train.py          # loop de entrenamiento y evaluación
├── notebooks/            # exploración y visualización
├── results/              # métricas, gráficas, modelos guardados
└── pyproject.toml
```

