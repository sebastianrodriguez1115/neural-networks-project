# Data Pipeline

## Fuentes de datos
- **PATRIC/BV-BRC** — genomas bacterianos en formato FASTA + metadatos de susceptibilidad (etiquetas AST)
- **CARD** — referencia de verdad terreno para validar genes de resistencia
- **NCBI Assembly** — fuente suplementaria si se necesitan más genomas para balancear clases

## Pasos del pipeline

### 1. Descarga
- Descargar genomas en formato FASTA de BV-BRC para el grupo ESKAPE
- Descargar las etiquetas fenotípicas (AMR phenotype dataset) asociadas a cada genoma
- Conservar todos los antibióticos con evidencia de laboratorio (`Laboratory Method`); no filtrar por antibiótico específico

#### Comandos CLI

```bash
# Descargar etiquetas AMR (rutas por defecto):
uv run python main.py download-amr
# Ruta personalizada:
uv run python main.py download-amr --output data/processed/amr_labels.csv

# Descarga todos los genomas del CSV (rutas por defecto):
uv run python main.py download-genomes
# Rutas personalizadas:
uv run python main.py download-genomes --labels data/processed/amr_labels.csv --output-dir data/raw/fasta
# Muestra de N genomas por especie, estratificada por fenotipo (útil para EDA):
uv run python main.py download-genomes --sample-per-species 20 --output-dir data/raw/fasta_sample
```

### 2. Preprocesamiento de etiquetas
- Filtrar únicamente registros con `laboratory_typing_method == 'Broth dilution'` (estándar de oro para MIC; recomendación del equipo)
- Conservar solo etiquetas binarias: `Resistant` / `Susceptible`
- Excluir etiquetas intermedias (`Intermediate`)
- Resultado: un archivo con triples `(genome_id, antibiotic, label)`

#### Comandos CLI

```bash
# Exportar pares con etiquetas contradictorias a CSV para inspección (rutas por defecto):
uv run python main.py export-contradictions-cmd
# Rutas personalizadas:
uv run python main.py export-contradictions-cmd --labels data/processed/amr_labels.csv --output data/processed/contradictory_labels.csv
```

### 3. Extracción de k-meros
- Para el **MLP**: extraer histogramas de frecuencia para k=3, 4, 5 y concatenarlos en un vector
  - Tamaño del vector: 4³ + 4⁴ + 4⁵ = 64 + 256 + 1024 = 1344 dimensiones
  - Normalizar (media 0, varianza 1)
- Para la **BiRNN** (dos variantes, implementar en orden):
  - **Variante A — artículo de referencia (prioridad):** mismos histogramas k=3,4,5 que el MLP, cada histograma paddeado a 1024 y apilado → matriz 3×1024 como input a la BiRNN
  - **Variante B — secuencia ordenada (extensión):** tokenizar el genoma como secuencia de k-meros con k=5; vocabulario 4⁵=1024 tokens + padding; requiere decidir longitud máxima (truncado, sliding window o submuestreo)

### 4. Manejo del desbalance de clases
- Analizar proporción Resistant/Susceptible en el dataset
- Si hay desbalance significativo: aplicar class weights en la función de pérdida (`pos_weight` en `BCEWithLogitsLoss` de PyTorch)

### 5. División de datos
- Split estratificado: 70% entrenamiento / 15% validación / 15% prueba
- Split por `genome_id` (no por registro) para evitar data leakage — el mismo genoma no puede aparecer en más de un conjunto
- Estratificación por etiqueta dentro del split para preservar proporción de clases

### Ejecutar el pipeline completo

Los pasos 2–5 están orquestados en el comando `prepare-data`, que debe ejecutarse después de completar la descarga.

#### Comandos CLI

```bash
# Pipeline completo con rutas por defecto:
uv run python main.py prepare-data

# Rutas personalizadas:
uv run python main.py prepare-data \
    --labels data/processed/amr_labels.csv \
    --fasta-dir data/raw/fasta \
    --output-dir data/processed

# Extracción de k-meros en paralelo (todos los CPUs):
uv run python main.py prepare-data --n-jobs -1
# N procesos específicos:
uv run python main.py prepare-data --n-jobs 4
```

El comando genera en `output_dir`:
- `cleaned_labels.csv` — triples limpios `(genome_id, antibiotic, label)`
- `antibiotic_index.csv` — mapeo antibiótico → entero
- `splits.csv` — asignación de cada `genome_id` a train/val/test
- `discarded_genomes.csv` — genomas descartados con motivo (`missing_fasta`, `below_min_length`)
- `mlp/<genome_id>.npy` — vector 1344-dim por genoma
- `bigru/<genome_id>.npy` — matriz `[1024, 3]` por genoma

## Decisiones pendientes
- [x] Qué organismo(s) bacteriano(s) usar → Grupo ESKAPE (excluyendo *Enterobacter spp.* por complejidad extra de la API de BV-BRC), modelo único entrenado con las 5 especies restantes
- [x] Qué antibiótico(s) usar → todos los disponibles con evidencia de laboratorio; el antibiótico entra como feature al modelo (embedding o índice entero)
- [ ] Longitud máxima de secuencia para la BiRNN Variante B (solo si se implementa) — truncado, sliding window o submuestreo
- [x] Formato de almacenamiento → etiquetas en CSV, vectores de k-meros en `.npy` (numpy); migrar a HDF5 si el rendimiento lo requiere
