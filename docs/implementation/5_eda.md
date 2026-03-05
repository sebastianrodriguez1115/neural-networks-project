# EDA — Análisis Exploratorio de Datos

## Data dictionary

El CSV de etiquetas (`data/processed/amr_labels.csv`) contiene las siguientes columnas:

| Columna | Tipo | Descripción | Valores / Rango | Nulos |
|---|---|---|---|---|
| `genome_id` | string | Identificador único del genoma en BV-BRC (formato: `taxon_id.número`) | ej. `1280.10000` | 0 |
| `taxon_id` | int | ID taxonómico NCBI de la especie | Uno de los 6 ESKAPE (ver bvbrc_api.md) | 0 |
| `antibiotic` | string | Nombre del antibiótico testeado (lowercase) | 96 valores únicos | 0 |
| `resistant_phenotype` | string | Fenotipo AMR binario | `"Resistant"` o `"Susceptible"` | 0 |
| `laboratory_typing_method` | string | Método de laboratorio usado | ej. `"MIC"`, `"disk diffusion"` | 14.4% |
| `testing_standard` | string | Estándar de interpretación clínica aplicado | ej. `"CLSI"`, `"EUCAST"` | 16.8% |

**Nota:** `laboratory_typing_method` y `testing_standard` son metadata del experimento de laboratorio. No se usan como features del modelo — solo `genome_id` y `antibiotic` son inputs; `resistant_phenotype` es el target.

---

## Qué es el EDA

El **Exploratory Data Analysis (EDA)** es una etapa previa a la implementación del pipeline en la que se examina el dataset crudo para entender su estructura, calidad y distribuciones. En este proyecto, el EDA opera sobre el CSV de etiquetas AMR descargado de BV-BRC.

El EDA responde preguntas clave antes de escribir código de preprocesamiento:
- ¿Cuántos datos hay por especie y antibiótico?
- ¿Qué tan balanceadas están las clases (Resistant / Susceptible)?
- ¿Hay duplicados o valores nulos que deba manejar el pipeline?
- ¿Cuántos antibióticos distintos hay? → determina la dimensión del embedding

## Cómo correrlo

```bash
uv run python main.py eda --genomes-dir data/raw/fasta_sample
uv run python main.py eda --genomes-dir data/raw/fasta_sample --labels data/processed/amr_labels.csv --top-n 30
```

El código está en `src/eda.py`. El comando `eda` en `main.py` es solo el punto de entrada CLI.

## Secciones del reporte

| Sección | Qué muestra | Por qué es importante |
|---|---|---|
| Resumen general | Total registros, genome IDs únicos, antibióticos, especies, dimensión (dim) de embedding sugerida | Da el scope del dataset y determina hiperparámetros (dimensión de embedding) |
| Registros por especie | Registros, genomas únicos, % Resistant / Susceptible por especie | Revela desbalance por especie que puede confundir al modelo |
| Balance de clases global | Conteo R/S y `pos_weight` (positive weight) sugerido para `BCEWithLogitsLoss` | Determina el `pos_weight` necesario para la función de pérdida |
| Top N antibióticos | Ranking por número de registros con balance R/S de cada uno | Identifica antibióticos con poca evidencia o desbalance severo |
| Calidad de datos | Valores nulos por columna y registros duplicados (genome_id + antibiotic) | Define qué limpiar en el pipeline antes de entrenar |
| Outliers | Genomas con registros extremos, antibióticos muy desbalanceados, etiquetas contradictorias | Detecta anomalías que pueden distorsionar el entrenamiento |
| Baseline benchmark | Majority class global y por antibiótico (accuracy, precision, recall, F1-score) | Establece el piso mínimo que los modelos deben superar para ser útiles |
| Análisis genómico | Longitud, contigs, GC content y alertas de calidad | Verifica que las secuencias de entrada sean de calidad suficiente para extraer k-meros |

## Hallazgos del EDA inicial (2026-03-03)

Dataset: ESKAPE completo, evidencia de laboratorio, fenotipos binarios.

### Resumen general
- **162,170 registros** · 16,204 genomas únicos · **96 antibióticos** · 5 especies
- *Enterobacter spp.* (taxon_id=547) no aparece en el dataset — posible problema con el nivel de agrupación por género

### Registros por especie
- Desbalance importante por especie:

| Especie | taxon_id | R% | S% |
|---|---|---|---|
| *Enterococcus faecium* | 1352 | 47.8% | 52.2% |
| *Staphylococcus aureus* | 1280 | 27.6% | 72.4% |
| *Klebsiella pneumoniae* | 573 | 63.6% | 36.4% |
| *Acinetobacter baumannii* | 470 | 79.5% | 20.5% |
| *Pseudomonas aeruginosa* | 287 | 50.9% | 49.1% |
| *Enterobacter spp.* | 547 | — | — |

### Balance de clases global
- Global: **54% Resistant / 46% Susceptible** (relativamente balanceado)
- **`pos_weight` (positive weight) = 0.8522**

`pos_weight` es un factor que ajusta cuánto penaliza la función de pérdida los errores en la clase positiva (Resistant). Se calcula como `negativos / positivos` (convención de PyTorch para `BCEWithLogitsLoss`):

`pos_weight = Susceptible / Resistant = 74,615 / 87,555 = 0.8522`

Un valor menor a 1 indica que la clase positiva es mayoritaria, por lo que se penaliza levemente menos para evitar que el modelo se sesgue hacia predecir siempre Resistant.

### Top N antibióticos
| Antibiótico | Registros | R% |
|---|---|---|
| gentamicin | 11,197 | 37% |
| ciprofloxacin | 9,622 | 64% |
| ampicillin | 7,818 | 92% |
| tetracycline | 7,086 | 45% |
| trimethoprim/sulfamethoxazole | 6,474 | 59% |

### Calidad de datos
- `laboratory_typing_method`: 23,321 nulos (14.4%) — esperado, filtro es por `evidence=Laboratory`
- `testing_standard`: 27,211 nulos (16.8%)
- **10,383 duplicados** (mismo genome_id + antibiotic) → a resolver en `data_pipeline.py`

## Análisis genómico (muestra de 89 genomas)

Muestra de 89 genomas descargados (20 por especie estratificados por fenotipo, 11 fallaron en BV-BRC).

### Estadísticas globales

| Métrica | Media | Std (desviación estándar) | Min | Max |
|---|---|---|---|---|
| Longitud total (Mb, megabases) | 4.44 | 1.61 | 2.67 | 8.60 |
| Número de contigs (fragmentos ensamblados continuos del genoma) | 296 | 880 | 9 | 5,776 |
| Contenido GC (guanina-citosina, %) | 46.17 | 12.69 | 32.64 | 66.43 |
| Bases N (%) | 0.00 | 0.01 | 0.00 | 0.04 |

### Por especie

| Especie | N | Long. media (Mb) | Contigs med. | GC% med. |
|---|---|---|---|---|
| *Acinetobacter baumannii* | 19 | 4.32 | 392 | 39.4% |
| *Enterococcus faecium* | 19 | 3.14 | 739 | 38.5% |
| *Klebsiella pneumoniae* | 16 | 5.56 | 138 | 57.2% |
| *Pseudomonas aeruginosa* | 17 | 6.73 | 100 | 66.1% |
| *Staphylococcus aureus* | 18 | 2.77 | 55 | 32.7% |

### Alertas y observaciones

- **Genomas cortos (<0.5 Mb):** 0 — no hay secuencias incompletas preocupantes
- **Genomas muy fragmentados (>500 contigs):** 4 — principalmente *Enterococcus faecium* (media 739 contigs); es un draft assembly típico de esta especie
- **Bases N (%):** prácticamente 0 en toda la muestra — buena calidad de secuenciación
- **GC content:** alta variabilidad entre especies (32.7% en *S. aureus* vs 66.1% en *P. aeruginosa*); los k-meros capturarán esta variación naturalmente

### Implicaciones para el pipeline

- Los 16,204 genomas completos (~4.4 Mb en promedio) requieren espacio significativo en disco (~64 GB total estimado)
- La alta fragmentación de *E. faecium* puede afectar la distribución de k-meros; es una variación esperada y no requiere tratamiento especial
- No se identifican problemas de calidad que requieran filtrado adicional

---

## Análisis de leakage y confounds

**Leakage** (fuga de información) ocurre cuando el modelo tiene acceso durante el entrenamiento a información que no debería tener, haciendo que parezca mejor de lo que realmente es.

**Confounds** (variables de confusión) son variables correlacionadas tanto con los features como con el target, que pueden hacer que el modelo aprenda una correlación espuria en lugar de la relación real.

### Features del modelo
El modelo recibe exclusivamente:
- **Input genómico**: frecuencias de k-meros extraídas del FASTA del genoma (no del CSV)
- **Input antibiótico**: embedding aprendido del nombre del antibiótico

El CSV de etiquetas **no se usa como fuente de features**. Solo provee el target (`resistant_phenotype`) y las claves para cruzar con los FASTA (`genome_id`, `antibiotic`).

### Leakage
No se identificó leakage. Las features (k-meros extraídos de los FASTA) y el target (`resistant_phenotype` del CSV) provienen de fuentes completamente separadas. Las columnas `laboratory_typing_method` y `testing_standard` podrían constituir leakage si se usaran como features, ya que describen el mismo experimento que produjo el target — por eso se descartan.

### Columnas descartadas
| Columna | Razón |
|---|---|
| `laboratory_typing_method` | Metadata del experimento de laboratorio, no del genoma; 14.4% nulos |
| `testing_standard` | Mismo razonamiento; 16.8% nulos |
| `taxon_id` | Redundante con `genome_id`; la especie ya está implícita en el genoma |

### Confounds identificados
- **Desbalance por especie**: *Acinetobacter baumannii* tiene 79.5% Resistant. Un modelo podría "hacer trampa" aprendiendo la especie en lugar del genotipo. Mitigación: no incluir taxon_id como feature; estratificar el split por `genome_id`.
- **Desbalance por antibiótico**: 27 antibióticos con R%≥90 o R%≤10 (ver sección Outliers). Para antibióticos con muy pocos registros (<10), el modelo tendrá poca evidencia.
- **Etiquetas contradictorias**: 488 pares (genome_id, antibiotic) con fenotipos distintos en registros diferentes. Pueden indicar errores de medición o variación experimental. Se resuelven conservando el primer registro (ver Decisiones derivadas).

---

## Outliers

### Genomas con registros extremos
- Media: **10.0 registros/genoma**, std: 7.3, umbral (mean+3σ): **31.9**
- **200 genomas** superan el umbral (de 16,204 total — 1.2%)
- Ejemplo: genoma `1280.1593` tiene 49 registros
- Interpretación: estos genomas fueron testeados contra muchos más antibióticos que el promedio. No son errores — son genomas con cobertura experimental mayor.

### Antibióticos con desbalance extremo (R%≥90 o R%≤10)
**27 antibióticos** con desbalance extremo. Los más notables:

| Antibiótico | Registros | R% | Nota |
|---|---|---|---|
| linezolid | 3,600 | 4.3% | Casi todo Susceptible |
| ampicillin | 7,818 | 91.5% | Casi todo Resistant |
| ceftriaxone | 4,092 | 88.0% | Alto R% |
| aztreonam | 3,764 | 84.0% | Alto R% |
| amoxicillin | 2 | 100.0% | Solo 2 registros |

Los antibióticos con muy pocos registros (n<10) y 100% de una clase probablemente no tendrán suficiente variabilidad para aprender. Se considerarán en el análisis de resultados.

### Etiquetas contradictorias
**488 pares** (genome_id, antibiotic) con fenotipos distintos en registros duplicados (de 10,383 duplicados totales — 4.7%). Estrategia: conservar el primer registro al eliminar duplicados.

---

## Baseline benchmark

Establece el **piso mínimo** que deben superar los modelos sin usar información genómica.

### Majority class global
Predecir siempre "Resistant" (clase mayoritaria):
- **Accuracy: 54.0%**

### Majority class por antibiótico
Para cada registro, predecir la clase mayoritaria de su antibiótico:
- **Accuracy: 71.2%**
- **Precision (Resistant): 0.7281**
- **Recall (Resistant): 0.7453**
- **F1 (Resistant): 0.7366**

Este baseline captura la señal epidemiológica del antibiótico (ej. ampicillin → casi siempre Resistant) sin usar ninguna información genómica. Los modelos deben superar **F1 ≥ 0.85** sobre este piso.

---

## Decisiones derivadas del EDA

- [x] **Dimensión (dim) de embedding del antibiótico: 49** → `min(50, (96 // 2) + 1)`
- [ ] **Duplicados**: decidir estrategia en `data_pipeline.py` (conservar el primero, promediar, o eliminar)
- [ ] **Enterobacter ausente**: investigar si taxon_id=547 captura los datos o si hay que usar IDs de especie
