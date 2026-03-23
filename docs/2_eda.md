# EDA — Análisis Exploratorio de Datos

Este documento resume los hallazgos del análisis exploratorio sobre el dataset de etiquetas AMR (resistencia antimicrobiana) descargado de BV-BRC y una muestra de genomas FASTA. El objetivo es entender la estructura, calidad y distribuciones del dataset antes de implementar el pipeline de preprocesamiento.

---

## Cómo correr el EDA

El código está en `src/eda.py`. El comando `eda` en `main.py` es el punto de entrada CLI.

### 1. Preparar la muestra (opcional)

Si no tienes los genomas, descarga una muestra pequeña (ej. 30 por especie) para el análisis genómico:

```bash
# Descargar etiquetas (si no existen)
uv run python main.py download-amr
# Descargar muestra de genomas
uv run python main.py download-genomes --sample-per-species 30 --output-dir data/raw/fasta_sample
```

### 2. Ejecutar el análisis

```bash
# Análisis completo (CSV y genomas en rutas por defecto):
uv run python main.py eda --genomes-dir data/raw/fasta_sample
# Ruta personalizada al CSV de etiquetas:
uv run python main.py eda --genomes-dir data/raw/fasta_sample --labels data/processed/amr_labels.csv
# Mostrar más antibióticos en el ranking (por defecto: 20):
uv run python main.py eda --genomes-dir data/raw/fasta_sample --top-n 30
```

---

## Panorama del dataset

El dataset contiene **162,170 registros** de pruebas de laboratorio que asocian genomas bacterianos con su respuesta a antibióticos. Cada registro es un triple `(genome_id, antibiotic, resistant_phenotype)`.

| Dimensión | Valor |
|---|---|
| Registros totales | 162,170 |
| Genomas únicos | 16,204 |
| Antibióticos distintos | 96 |
| Especies ESKAPE presentes | 5 de 6 |

*Enterobacter spp.* (taxon_id=547) no aparece — posible problema con el nivel de agrupación por género en BV-BRC.

### Columnas del CSV

| Columna | Tipo | Descripción | Nulos |
|---|---|---|---|
| `genome_id` | string | Identificador del genoma en BV-BRC (ej. `1280.10000`) | 0 |
| `taxon_id` | int | ID taxonómico NCBI de la especie | 0 |
| `antibiotic` | string | Nombre del antibiótico testeado (lowercase) | 0 |
| `resistant_phenotype` | string | `"Resistant"` o `"Susceptible"` | 0 |
| `laboratory_typing_method` | string | Método de laboratorio (ej. MIC, disk diffusion) | 14.4% |
| `testing_standard` | string | Estándar clínico aplicado (ej. CLSI, EUCAST) | 16.8% |

Solo `genome_id` y `antibiotic` son inputs del modelo; `resistant_phenotype` es el target. Las demás columnas son metadata del experimento de laboratorio y no se usan como features.

---

## Distribución y balance

### Por especie

El desbalance entre especies es significativo y representa un confound potencial: un modelo podría aprender a predecir la especie en vez del genotipo de resistencia.

| Especie | Registros | Genomas | R% | S% |
|---|---|---|---|---|
| *Enterococcus faecium* | 22,318 | 3,214 | 47.8% | 52.2% |
| *Staphylococcus aureus* | 41,458 | 4,437 | 27.6% | 72.4% |
| *Klebsiella pneumoniae* | 66,140 | 5,750 | 63.6% | 36.4% |
| *Acinetobacter baumannii* | 24,193 | 1,426 | 79.5% | 20.5% |
| *Pseudomonas aeruginosa* | 8,061 | 1,377 | 50.9% | 49.1% |

*A. baumannii* tiene 80% Resistant mientras *S. aureus* tiene 72% Susceptible — extremos opuestos. Mitigación: no incluir `taxon_id` como feature; la especie queda implícita en los k-meros del genoma.

### Balance global

El balance global es razonable: **54% Resistant / 46% Susceptible**.

`pos_weight = Susceptible / Resistant = 74,615 / 87,555 = 0.8522`

Un valor menor a 1 indica que la clase positiva (Resistant) es mayoritaria, por lo que `BCEWithLogitsLoss` la penaliza levemente menos para evitar sesgar el modelo hacia predecir siempre Resistant.

### Por antibiótico

Los 5 antibióticos con más registros cubren una mezcla de perfiles:

| Antibiótico | Registros | R% |
|---|---|---|
| gentamicin | 11,197 | 37% |
| ciprofloxacin | 9,622 | 64% |
| ampicillin | 7,818 | 92% |
| tetracycline | 7,086 | 45% |
| trimethoprim/sulfamethoxazole | 6,474 | 59% |

**27 antibióticos** tienen desbalance extremo (R% ≥ 90 o R% ≤ 10). Algunos con muy pocos registros (ej. amoxicillin con 2, dicloxacillin con 1) no tendrán suficiente variabilidad para que el modelo aprenda.

→ Dimensión de embedding del antibiótico: **49** `[min(50, (96 // 2) + 1)]`

---

## Métodos de laboratorio

El dataset incluye diversos métodos de fenotipado. La consistencia técnica entre estos métodos es crucial para la calidad de las etiquetas.

| Método de laboratorio | Registros | % |
|---|---|---|
| **Broth dilution** | **86,705** | **53.5%** |
| Nulo (No especificado) | 23,321 | 14.4% |
| Disk diffusion | 20,444 | 12.6% |
| MIC | 14,886 | 9.2% |
| Biofosun Gram-positive panels | 8,406 | 5.2% |
| Otros (Vitek, Agar dilution, etc.) | 8,408 | 5.2% |

> **Recomendación del equipo:** Filtrar el dataset para concentrar el entrenamiento únicamente en registros con el método **'Broth dilution'**. Es considerado el *estándar de oro* para determinar la Concentración Mínima Inhibitoria (MIC) y asegura la mayor consistencia biológica en las etiquetas.

---

## Calidad de datos

### Valores nulos

Los nulos están concentrados en `laboratory_typing_method` (14.4%) y `testing_standard` (16.8%). No afectan el pipeline porque estas columnas no se usan como features.

### Duplicados

**10,383 registros duplicados** (mismo `genome_id` + `antibiotic`). De estos, **488 pares tienen etiquetas contradictorias** (un registro dice Resistant y otro dice Susceptible para el mismo genoma y antibiótico). Esto puede indicar variación experimental o errores de medición.

Estrategia propuesta: conservar el primer registro al eliminar duplicados en `src/data_pipeline/cleaning.py`.

### Genomas con cobertura extrema

200 genomas (1.2% del total) tienen más de 32 registros (media: 10, umbral: mean+3σ). No son errores — son genomas testeados contra muchos antibióticos. No requieren tratamiento especial.

---

## Análisis genómico

Se analizó una muestra de 138 genomas (30 por especie, estratificados por fenotipo). El objetivo es verificar que la materia prima para extraer k-meros sea de calidad suficiente.

### Estadísticas por especie

| Especie | N | Long. media (Mb) | Contigs med. | GC% med. |
|---|---|---|---|---|
| *Enterococcus faecium* | 29 | 2.86 | 563 | 38.2% |
| *Staphylococcus aureus* | 27 | 2.78 | 53 | 32.7% |
| *Klebsiella pneumoniae* | 26 | 5.55 | 118 | 57.2% |
| *Acinetobacter baumannii* | 29 | 4.20 | 288 | 39.3% |
| *Pseudomonas aeruginosa* | 27 | 6.69 | 160 | 66.0% |

La variabilidad de GC entre especies (32.7%–66.0%) es una señal biológica real que los k-meros capturarán naturalmente.

### Alertas

- **2 genomas muy cortos** (<0.5 Mb): `1352.11605` (3.6 kb) y `1352.11302` (16.9 kb) — ensambles casi vacíos de *E. faecium*. Un genoma típico de esta especie tiene ~3 Mb; estos tienen menos del 1%. Producirían histogramas de k-meros casi vacíos y deben filtrarse en el pipeline.
- **6 genomas muy fragmentados** (>500 contigs): principalmente *E. faecium*, consistente con la calidad típica de draft assemblies de esta especie. No requieren filtrado.
- **Bases N**: prácticamente 0% en toda la muestra — buena calidad de secuenciación.

### Implicaciones para el pipeline

- Los ~16,000 genomas completos (~4.4 Mb promedio) requieren ~64 GB en disco.
- Se necesita un filtro de longitud mínima (ej. 0.5 Mb) para descartar ensambles incompletos.
- La fragmentación de *E. faecium* no requiere tratamiento especial — los k-meros operan sobre cada contig independientemente.

---

## Baseline benchmark

El baseline establece el **piso mínimo** que los modelos deben superar sin usar información genómica.

**Predicción naive** (siempre "Resistant"): accuracy 54.0%.

**Predicción por antibiótico** (clase mayoritaria de cada antibiótico):

| Métrica | Valor |
|---|---|
| Accuracy | 71.2% |
| Precision (Resistant) | 0.7281 |
| Recall (Resistant) | 0.7453 |
| **F1 (Resistant)** | **0.7366** |

Este baseline captura la señal epidemiológica del antibiótico (ej. ampicillin → casi siempre Resistant) sin usar ninguna información genómica. Los modelos deben superar **F1 ≥ 0.85** para demostrar que los k-meros aportan valor predictivo.

---

## Leakage y confounds

### Leakage

No se identificó leakage. Las features (k-meros del FASTA) y el target (`resistant_phenotype` del CSV) provienen de fuentes independientes. Las columnas `laboratory_typing_method` y `testing_standard` describen el mismo experimento que produjo el target — por eso se descartan.

### Confounds identificados

| Confound | Riesgo | Mitigación |
|---|---|---|
| Desbalance por especie | El modelo aprende la especie en vez del genotipo | No incluir `taxon_id` como feature; split por `genome_id` |
| Desbalance por antibiótico | 27 antibióticos con R% ≥ 90 o ≤ 10 | Considerar en el análisis de resultados |
| Etiquetas contradictorias | 488 pares con fenotipos distintos | Conservar primer registro al deduplicar |

---

## Decisiones derivadas

- [x] **Dimensión de embedding del antibiótico: 49** → `min(50, (96 // 2) + 1)`
- [x] **Filtrado técnico**: Limitar el dataset únicamente a registros con `laboratory_typing_method == 'Broth dilution'`.
- [x] **Duplicados**: Se conserva el primer registro usando `drop_duplicates(keep='first')` en `src/data_pipeline/cleaning.py`.
- [x] **Enterobacter ausente**: Se investigó el `taxon_id=547`. El endpoint de AMR de BV-BRC no soporta filtrar por `taxon_lineage_ids`, lo que requeriría una doble consulta (primero genomas, luego AMR por IDs) añadiendo complejidad no justificada. Se excluyó *Enterobacter spp.* del alcance del proyecto.
- [x] **Filtro de longitud mínima**: Se descartan genomas < 0.5 Mb en el pipeline (`MIN_GENOME_LENGTH = 500_000`).

