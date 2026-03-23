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
uv run python main.py eda --genomes-dir data/raw/fasta_sample --top-n-antibiotics 30
```

---

## Panorama del dataset

El dataset contiene **162,170 registros** de pruebas de laboratorio que asocian genomas bacterianos con su respuesta a antibióticos. Cada registro es un triple `(genome_id, antibiotic, resistant_phenotype)`.

| Dimensión | Valor |
|---|---|
| Registros totales | 162,170 |
| Genomas únicos | 16,281 |
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
| *Enterococcus faecium* | 22,318 | 3,234 | 47.8% | 52.2% |
| *Staphylococcus aureus* | 41,458 | 4,437 | 27.6% | 72.4% |
| *Klebsiella pneumoniae* | 66,140 | 5,751 | 63.6% | 36.4% |
| *Acinetobacter baumannii* | 24,193 | 1,469 | 79.5% | 20.5% |
| *Pseudomonas aeruginosa* | 8,061 | 1,390 | 50.9% | 49.1% |

*A. baumannii* tiene 80% Resistant mientras *S. aureus* tiene 72% Susceptible — extremos opuestos. Mitigación: no incluir `taxon_id` como feature; la especie queda implícita en los k-meros del genoma.

### Balance global

El balance global es razonable: **54% Resistant / 46% Susceptible**.

`pos_weight = Susceptible / Resistant = 74,615 / 87,555 = 0.8522`

Un valor menor a 1 indica que la clase positiva (Resistant) es mayoritaria, por lo que `BCEWithLogitsLoss` la penaliza levemente menos para evitar sesgar el modelo hacia predecir siempre Resistant.

### Por antibiótico (Lista completa)

A continuación se detallan los 96 antibióticos presentes en el dataset original:

| Antibiótico | Registros | R% | S% |
|---|---|---|---|
| gentamicin | 11,197 | 37.0% | 63.0% |
| ciprofloxacin | 9,622 | 63.5% | 36.5% |
| ampicillin | 7,818 | 91.5% | 8.5% |
| tetracycline | 7,086 | 44.5% | 55.5% |
| trimethoprim/sulfamethoxazole | 6,474 | 59.3% | 40.7% |
| vancomycin | 6,469 | 32.1% | 67.9% |
| ceftazidime | 5,949 | 73.3% | 26.7% |
| meropenem | 5,658 | 43.6% | 56.4% |
| cefoxitin | 5,309 | 64.4% | 35.6% |
| amikacin | 5,121 | 28.8% | 71.2% |
| erythromycin | 4,913 | 58.8% | 41.2% |
| tobramycin | 4,340 | 56.4% | 43.6% |
| imipenem | 4,129 | 47.3% | 52.7% |
| ceftriaxone | 4,092 | 88.0% | 12.0% |
| levofloxacin | 4,067 | 71.2% | 28.8% |
| aztreonam | 3,764 | 84.0% | 16.0% |
| linezolid | 3,600 | 4.3% | 95.7% |
| piperacillin/tazobactam | 3,234 | 69.4% | 30.6% |
| cefazolin | 3,202 | 86.3% | 13.7% |
| ampicillin/sulbactam | 3,058 | 74.9% | 25.1% |
| clindamycin | 3,009 | 53.3% | 46.7% |
| tigecycline | 2,962 | 9.6% | 90.4% |
| cefepime | 2,884 | 65.0% | 35.0% |
| penicillin | 2,647 | 88.4% | 11.6% |
| rifampin | 2,607 | 11.5% | 88.5% |
| cefuroxime | 2,472 | 95.7% | 4.3% |
| fusidic acid | 2,430 | 14.4% | 85.6% |
| cefotaxime | 2,398 | 81.2% | 18.8% |
| chloramphenicol | 2,169 | 29.2% | 70.8% |
| colistin | 2,155 | 16.5% | 83.5% |
| oxacillin | 1,941 | 45.7% | 54.3% |
| daptomycin | 1,891 | 21.9% | 78.1% |
| teicoplanin | 1,870 | 37.7% | 62.3% |
| nitrofurantoin | 1,869 | 88.3% | 11.7% |
| streptomycin | 1,743 | 46.4% | 53.6% |
| mupirocin | 1,680 | 3.3% | 96.7% |
| ertapenem | 1,567 | 74.5% | 25.5% |
| methicillin | 1,440 | 46.5% | 53.5% |
| kanamycin | 1,366 | 67.6% | 32.4% |
| quinupristin/dalfopristin | 1,317 | 9.0% | 91.0% |
| amoxicillin/clavulanic acid | 1,257 | 83.2% | 16.8% |
| trimethoprim | 928 | 18.1% | 81.9% |
| doxycycline | 749 | 48.5% | 51.5% |
| fosfomycin | 658 | 13.4% | 86.6% |
| ceftazidime/avibactam | 556 | 37.1% | 62.9% |
| doripenem | 484 | 59.1% | 40.9% |
| extended spectrum beta lactamase | 451 | 42.4% | 57.6% |
| polymyxin B | 396 | 16.2% | 83.8% |
| moxifloxacin | 356 | 66.3% | 33.7% |
| ticarcillin/clavulanic acid | 348 | 54.6% | 45.4% |
| ceftolozane/tazobactam | 311 | 59.2% | 40.8% |
| cefiderocol | 270 | 2.2% | 97.8% |
| trimethoprim/sulfobactam | 248 | 35.1% | 64.9% |
| phosphomycin | 247 | 1.2% | 98.8% |
| carbenicillin | 244 | 98.4% | 1.6% |
| nalidixic acid | 232 | 96.1% | 3.9% |
| cefpodoxime | 227 | 79.7% | 20.3% |
| cefpodoxime_clavulanic_acid | 212 | 85.8% | 14.2% |
| norfloxacin | 210 | 14.8% | 85.2% |
| minocycline | 192 | 20.3% | 79.7% |
| cefotetan | 180 | 78.9% | 21.1% |
| neomycin | 168 | 73.2% | 26.8% |
| beta-lactam | 148 | 66.9% | 33.1% |
| aminogycosides | 145 | 60.0% | 40.0% |
| fluoroquinolones | 145 | 57.2% | 42.8% |
| trimotheprim | 145 | 71.0% | 29.0% |
| cefalotin | 138 | 99.3% | 0.7% |
| ceftarolin | 87 | 50.6% | 49.4% |
| spectinomycin | 83 | 100.0% | 0.0% |
| piperacillin | 75 | 90.7% | 9.3% |
| ofloxacin | 74 | 63.5% | 36.5% |
| ceftriaxone/cefpodoxime | 60 | 51.7% | 48.3% |
| ceftaroline | 55 | 47.3% | 52.7% |
| azidothymidine | 50 | 98.0% | 2.0% |
| tedizolid | 49 | 75.5% | 24.5% |
| dalbavancin | 49 | 77.6% | 22.4% |
| ticarcillin | 46 | 100.0% | 0.0% |
| synercid | 44 | 22.7% | 77.3% |
| sulfamethazine | 42 | 92.9% | 7.1% |
| tylosin | 42 | 45.2% | 54.8% |
| tgecycline | 42 | 0.0% | 100.0% |
| lincomycin | 41 | 82.9% | 17.1% |
| virginiamycin | 41 | 78.0% | 22.0% |
| florfenicol | 36 | 66.7% | 33.3% |
| cefotaxime/clavulanic acid | 24 | 100.0% | 0.0% |
| sulfonamides | 23 | 100.0% | 0.0% |
| tiamulin | 20 | 60.0% | 40.0% |
| azithromycin | 4 | 100.0% | 0.0% |
| clarithromycin | 4 | 100.0% | 0.0% |
| sulbactam | 3 | 100.0% | 0.0% |
| cefalexin | 3 | 100.0% | 0.0% |
| cephalothin | 3 | 66.7% | 33.3% |
| amoxicillin | 2 | 100.0% | 0.0% |
| sulfamethoxazole | 2 | 100.0% | 0.0% |
| netilmicin | 1 | 100.0% | 0.0% |
| dicloxacillin | 1 | 100.0% | 0.0% |

**Decisión técnica:** Se aplicará un filtro de **mínimo 20 registros por antibiótico**. Aquellos con una frecuencia menor no tienen suficiente variabilidad para que el modelo aprenda un embedding significativo y complican la partición train/val/test.

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
| Biofosun Gram-positive panels broth dilution | 8,406 | 5.2% |
| Vitek_2-P607_card | 3,118 | 1.9% |
| 2014,2015 | 2,708 | 1.7% |
| Agar dilution | 1,982 | 1.2% |
| Otros | 600 | 0.4% |

> **Recomendación del equipo:** Filtrar el dataset para concentrar el entrenamiento únicamente en registros con el método **'Broth dilution'**. Es considerado el *estándar de oro* para determinar la Concentración Mínima Inhibitoria (MIC) y asegura la mayor consistencia biológica en las etiquetas.

---

## Calidad de datos

### Valores nulos

Los nulos están concentrados en `laboratory_typing_method` (14.4%) y `testing_standard` (16.8%). No afectan el pipeline porque estas columnas no se usan como features.

### Duplicados

**10,031 registros duplicados** (mismo `genome_id` + `antibiotic`). De estos, **382 pares tienen etiquetas contradictorias** (un registro dice Resistant y otro dice Susceptible para el mismo genoma y antibiótico). Esto puede indicar variación experimental o errores de medición.

Estrategia propuesta: conservar el primer registro al eliminar duplicados en `src/data_pipeline/cleaning.py`.

### Genomas con cobertura extrema

191 genomas (1.2% del total) tienen más de 32 registros (media: 10, umbral: mean+3σ). No son errores — son genomas testeados contra muchos antibióticos. No requieren tratamiento especial.

---

## Análisis genómico

Se analizó la calidad de los **16,571 archivos FASTA** presentes en el repositorio. El objetivo es verificar que la materia prima para extraer k-meros sea de calidad suficiente y definir los umbrales de filtrado técnico.

### Estadísticas globales

| Métrica | Media | Std | Min | Max |
|---|---|---|---|---|
| Longitud total (Mb) | 4.23 | 1.53 | 0.00 | 12.46 |
| Número de contigs | 151.09 | 284.74 | 1.00 | 8507.00 |
| Contenido GC (%) | 45.87 | 12.05 | 31.19 | 66.67 |
| Bases N (%) | 0.01 | 0.04 | 0.00 | 2.96 |

### Estadísticas por especie

| Especie | N | Long. media (Mb) | Contigs med. | GC% med. |
|---|---|---|---|---|
| *Acinetobacter baumannii* | 1,469 | 3.97 | 147.1 | 39.0% |
| *Enterococcus faecium* | 3,234 | 2.83 | 346.4 | 38.1% |
| *Klebsiella pneumoniae* | 5,751 | 5.56 | 126.6 | 57.2% |
| *Pseudomonas aeruginosa* | 1,390 | 6.73 | 122.7 | 66.1% |
| *Staphylococcus aureus* | 4,437 | 2.79 | 54.1 | 32.7% |
| desconocida | 290 | 4.54 | 98.3 | 47.9% |

La variabilidad de GC entre especies es una señal biológica real que los k-meros capturarán naturalmente para diferenciar especies antes de predecir la resistencia.

### Alertas de calidad

- **219 genomas muy cortos** (<0.5 Mb): Corresponden a ensambles incompletos o fallidos. Estos deben ser filtrados en el pipeline para evitar ruidos en los histogramas de k-meros.
- **567 genomas muy fragmentados** (>500 contigs): Mayoritariamente concentrados en *E. faecium*. La fragmentación no impide la extracción de k-meros, pero indica menor calidad de ensamble.
- **Bases N**: El promedio es despreciable (0.01%), indicando secuencias de alta confianza.

### Implicaciones para el pipeline

- **Filtro de longitud mínima**: Es indispensable aplicar un filtro de 0.5 Mb (`MIN_GENOME_LENGTH = 500_000`) para descartar los 219 genomas anómalos.
- **Espacio en disco**: El dataset completo ocupa ~64 GB.
- **Robustez**: Los k-meros operan sobre contigs independientes, por lo que la fragmentación (incluso en los 567 casos extremos) no requiere un tratamiento especial más allá de la normalización por longitud.

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

