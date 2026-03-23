# Entrega 2 — Red Neuronal Superficial (MLP) para Prediccion de Resistencia Antimicrobiana

## 1. Project Goal and Scope

La resistencia antimicrobiana (RAM) constituye una amenaza critica para la salud publica global. La deteccion rapida de los determinantes de resistencia es fundamental para orientar un tratamiento eficaz [1][4][5]. La secuenciacion del genoma completo (WGS) permite la identificacion basada en genotipo de estos determinantes, pero se requieren metodos de prediccion escalables y precisos [2][6][7].

Este proyecto desarrolla un sistema basado en redes neuronales que predice fenotipos de resistencia antimicrobiana (Resistente vs. Susceptible) a partir de datos de secuencias del genoma completo bacteriano. Se implementan y comparan dos modelos complementarios:

1. **Red neuronal superficial (MLP):** perceptron multicapa entrenado sobre vectores de frecuencias de k-meros — esta entrega.
2. **Red neuronal profunda (BiGRU + Attention):** red recurrente bidireccional con mecanismo de atencion — entrega final.

El alcance del proyecto abarca la adquisicion de datos desde repositorios publicos, el preprocesamiento, el entrenamiento y la evaluacion mediante metricas estandar de clasificacion binaria, excluyendo la validacion clinica y el despliegue.

---

## 2. Literature Review

### 2.1 Resistencia antimicrobiana y WGS

La resistencia a los antibioticos es impulsada por mutaciones geneticas y la transferencia horizontal de genes de resistencia [1][4]. Los enfoques tradicionales basados en alineamiento (BLAST, ResFinder, CARD) son altamente precisos para identificar genes documentados, pero presentan dificultades frente a mecanismos novedosos o patrones genomicos complejos [2][5][6].

### 2.2 Aprendizaje automatico para prediccion de RAM

Los modelos de aprendizaje automatico y aprendizaje profundo han emergido como alternativas que aprenden representaciones ocultas de los datos genomicos y pueden generalizar patrones sin depender de catalogos preexistentes [3][7][8]:

- **Ren et al. (2022)** [3] evaluaron CNN y Random Forest para la prediccion de resistencia a ciprofloxacina y cefotaxima usando vectores de k-meros y codificacion One-Hot, alcanzando AUC de hasta 0.96.
- **Jia et al. (2024)** demostraron que redes neuronales profundas integradas con datos de expresion genica alcanzan 98.64% de precision en la prediccion de susceptibilidad en *A. baumannii* multirresistente.
- **Nguyen et al. (2019)** [9] demostraron prediccion a gran escala de MICs a partir de caracteristicas genomicas en *Salmonella*.

### 2.3 Representaciones basadas en k-meros

Los k-meros (subsecuencias de longitud k) son una representacion ampliamente validada para codificar secuencias de ADN en vectores numericos. Los histogramas de frecuencias de k-meros capturan estadisticas composicionales locales sin necesidad de alineamiento y han sido empleados exitosamente en clasificacion taxonomica [11][12][13] y prediccion de RAM [3]. Sin embargo, al tratarse de un resumen composicional, no preservan el contexto posicional dentro del genoma.

### 2.4 Definiciones clave

- **K-mero:** subsecuencia contigua de *k* nucleotidos (ej. para k=3: ACG, CGT, GTA, ...). El vocabulario tiene 4^k posibles k-meros.
- **Histograma de k-meros:** vector de frecuencias normalizadas de cada k-mero posible en un genoma.
- **MLP (Perceptron Multicapa):** red neuronal feedforward con capas densas completamente conectadas, funciones de activacion no lineales, y entrenamiento por retropropagacion del error.
- **BCEWithLogitsLoss:** funcion de perdida de entropia cruzada binaria con transformacion sigmoide integrada, adecuada para clasificacion binaria.
- **Early stopping:** tecnica de regularizacion que detiene el entrenamiento cuando la metrica de validacion deja de mejorar, previniendo sobreajuste [Haykin, 2009].

---

## 3. Case Study

Este proyecto esta inspirado en el articulo de **Lugo y Barreto-Hernandez (2021)** [11]: *"A Recurrent Neural Network approach for whole genome bacteria identification"*, publicado en Applied Artificial Intelligence.

El articulo introduce:
- Una representacion distribuida de genomas bacterianos basada en k-meros (k=3,4,5).
- Una arquitectura de RNN bidireccional (BiGRU) con mecanismo de atencion para procesar secuencias genomicas completas.
- Resultados competitivos para identificacion taxonomica a partir de WGS.

**Relevancia para el proyecto:** La arquitectura y la representacion del caso de estudio se adaptan directamente a la prediccion de RAM. La tarea es mas dificil dado que la senal relevante corresponde a variaciones sutiles asociadas a la resistencia dentro de una misma especie, en lugar de amplias diferencias interespecificas. Para esta entrega, usamos los mismos histogramas de k-meros como entrada al MLP baseline; en la entrega final, se implementara la BiGRU+Attention del articulo.

---

## 4. Training Data Set

### 4.1 Fuentes de datos

- **PATRIC/BV-BRC:** genomas bacterianos completos en formato FASTA y metadatos de susceptibilidad antimicrobiana (etiquetas fenotipicas AST).
- **CARD (Comprehensive Antibiotic Resistance Database):** referencia de verdad terreno para validar genes de resistencia.

### 4.2 Composicion del dataset

El dataset original contiene **162,170 registros** de pruebas de laboratorio que asocian genomas bacterianos con su respuesta a antibioticos. Cada registro es un triple `(genome_id, antibiotic, resistant_phenotype)`.

| Dimension | Valor |
|---|---|
| Registros totales | 162,170 |
| Genomas unicos | 16,204 |
| Antibioticos distintos | 96 |
| Especies ESKAPE presentes | 5 de 6 |

Las especies incluidas son: *Enterococcus faecium*, *Staphylococcus aureus*, *Klebsiella pneumoniae*, *Acinetobacter baumannii* y *Pseudomonas aeruginosa*. *Enterobacter spp.* esta ausente del dataset de BV-BRC.

### 4.3 Filtrado y dataset final para entrenamiento

Tras el preprocesamiento (seccion 5), el dataset se reduce a registros con metodo de laboratorio *Broth dilution* (estandar de oro), etiquetas binarias (R/S), y genomas validos. El dataset final de entrenamiento contiene:

| Particion | Muestras |
|---|---|
| Entrenamiento (70%) | 57,088 |
| Validacion (15%) | 12,613 |
| Prueba (15%) | 12,519 |
| **Total** | **82,220** |

### 4.4 Entradas y salidas

- **Entrada (features):** vector de histograma de frecuencias de k-meros concatenados (k=3,4,5), resultando en un vector de **1,344 dimensiones** por genoma, normalizado (media 0, varianza 1). El antibiotico se codifica como un indice entero que alimenta una tabla de embeddings aprendidos (dimension 49).
- **Salida (label):** clasificacion binaria — `1` = Resistant, `0` = Susceptible.

---

## 5. Data Pre-processing and Exploratory Data Analysis

### 5.1 Pipeline de preprocesamiento

El pipeline de datos se ejecuta con el comando `uv run python main.py prepare-data` y comprende los siguientes pasos:

1. **Filtrado por metodo de laboratorio:** solo se conservan registros con `laboratory_typing_method == 'Broth dilution'` (53.5% del dataset original), considerado el estandar de oro para la determinacion de la Concentracion Minima Inhibitoria (MIC).
2. **Eliminacion de etiquetas contradictorias:** se descartan 488 pares genoma-antibiotico donde existian etiquetas simultaneas de Resistant y Susceptible.
3. **Deduplicacion:** se conserva el primer registro para duplicados consistentes (mismo `genome_id` + `antibiotic`).
4. **Filtrado de genomas incompletos:** se descartan genomas < 0.5 Mb (2 ensambles casi vacios de *E. faecium*).
5. **Extraccion de k-meros:** para cada genoma se calculan histogramas de frecuencia para k=3, 4 y 5 (4^3 + 4^4 + 4^5 = 1,344 dimensiones) y se concatenan en un unico vector.
6. **Normalizacion:** media 0 y varianza 1, calculadas exclusivamente sobre el conjunto de entrenamiento para evitar data leakage.
7. **Split estratificado:** 70/15/15 por `genome_id` (no por registro), con estratificacion por etiqueta.

### 5.2 Analisis exploratorio de datos

#### Balance de clases

El balance global es razonable: **54% Resistant / 46% Susceptible**.

`pos_weight = n_susceptible / n_resistant = 74,615 / 87,555 = 0.8522`

#### Distribucion por especie

| Especie | Registros | Genomas | R% | S% |
|---|---|---|---|---|
| *Enterococcus faecium* | 22,318 | 3,214 | 47.8% | 52.2% |
| *Staphylococcus aureus* | 41,458 | 4,437 | 27.6% | 72.4% |
| *Klebsiella pneumoniae* | 66,140 | 5,750 | 63.6% | 36.4% |
| *Acinetobacter baumannii* | 24,193 | 1,426 | 79.5% | 20.5% |
| *Pseudomonas aeruginosa* | 8,061 | 1,377 | 50.9% | 49.1% |

El desbalance entre especies es significativo: *A. baumannii* tiene 80% Resistant mientras *S. aureus* tiene 72% Susceptible. Mitigacion: no se incluye `taxon_id` como feature; la especie queda implicitamente codificada en los k-meros del genoma.

#### Antibioticos con mas registros

| Antibiotico | Registros | R% |
|---|---|---|
| gentamicin | 11,197 | 37% |
| ciprofloxacin | 9,622 | 64% |
| ampicillin | 7,818 | 92% |
| tetracycline | 7,086 | 45% |
| trimethoprim/sulfamethoxazole | 6,474 | 59% |

27 antibioticos presentan desbalance extremo (R% >= 90 o R% <= 10).

#### Analisis genomico

Se analizo una muestra de 138 genomas (30 por especie, estratificados por fenotipo):

| Especie | N | Long. media (Mb) | GC% med. |
|---|---|---|---|
| *E. faecium* | 29 | 2.86 | 38.2% |
| *S. aureus* | 27 | 2.78 | 32.7% |
| *K. pneumoniae* | 26 | 5.55 | 57.2% |
| *A. baumannii* | 29 | 4.20 | 39.3% |
| *P. aeruginosa* | 27 | 6.69 | 66.0% |

La variabilidad de GC entre especies (32.7%-66.0%) es una senal biologica real que los k-meros capturan naturalmente.

#### Calidad de datos

- **Valores nulos:** concentrados en `laboratory_typing_method` (14.4%) y `testing_standard` (16.8%). No afectan el pipeline ya que no se usan como features.
- **Leakage:** no identificado. Features (k-meros del FASTA) y target (`resistant_phenotype` del CSV) provienen de fuentes independientes.
- **Confounds identificados:** desbalance por especie y por antibiotico (mitigados con la estrategia descrita).

#### Baseline benchmark

Se establecio un baseline sin informacion genomica para definir el piso minimo:

| Metrica | Prediccion naive (siempre R) | Prediccion por antibiotico |
|---|---|---|
| Accuracy | 54.0% | 71.2% |
| Precision | — | 0.7281 |
| Recall | — | 0.7453 |
| **F1** | — | **0.7366** |

Los modelos deben superar **F1 >= 0.85** para demostrar que los k-meros aportan valor predictivo mas alla de la senal epidemiologica del antibiotico.

---

## 6. Proposed Shallow Neural Network Based Approach

### 6.1 Tipo y justificacion de la arquitectura

Se implemento un **Perceptron Multicapa (MLP)** como red neuronal superficial de referencia. La arquitectura usa 2 capas ocultas (512 → 128), lo cual tecnicamente supera la definicion minima de "deep" (>1 capa oculta). Sin embargo, se considera superficial en el contexto de este proyecto por dos razones:

1. La propuesta define "superficial" en contraste con la BiGRU+Attention, que posee capas recurrentes, mecanismo de atencion y mayor capacidad de modelar dependencias secuenciales.
2. En la literatura de deep learning, redes de 2-3 capas densas se consideran *shallow* frente a arquitecturas con decenas o cientos de capas [12][13]. La segunda capa oculta (128) cumple un rol de compresion progresiva.

El MLP proporciona un punto de referencia solido para determinar si la composicion genomica (histogramas de k-meros) es suficiente para discriminar aislamientos resistentes de susceptibles, sin necesidad de modelar dependencias posicionales.

### 6.2 Feature engineering

**Extraccion de features (k-meros):**
- Para cada genoma bacteriano, se extraen histogramas de frecuencia de k-meros para k=3, 4 y 5.
- Los tres histogramas se concatenan en un unico vector de 4^3 + 4^4 + 4^5 = **1,344 dimensiones**.
- El vector se normaliza (media 0, varianza 1) usando estadisticas calculadas exclusivamente sobre el conjunto de entrenamiento.

**Codificacion del antibiotico:**
- Cada antibiotico se codifica como un indice entero.
- El modelo contiene una tabla de embeddings aprendidos de dimension 49 (`min(50, (96 // 2) + 1)`), que transforma el indice en un vector denso.
- El embedding del antibiotico se concatena con el vector genomico antes de las capas densas.

**Entrada total al clasificador:** vector de 1,344 + 49 = **1,393 dimensiones**.

### 6.3 Arquitectura de la red

```
Entrada genomica (1344) ─────────────┐
                                      ├─ Concatenar ─→ Dense(1393, 512) + ReLU + Dropout(0.3)
Indice antibiotico → Embedding (49) ─┘                        │
                                                    Dense(512, 128) + ReLU + Dropout(0.3)
                                                               │
                                                        Dense(128, 1) → logit
```

| Componente | Detalle |
|---|---|
| Parametros totales | 782,804 |
| Capa oculta 1 | Linear(1393, 512) + ReLU + Dropout(0.3) |
| Capa oculta 2 | Linear(512, 128) + ReLU + Dropout(0.3) |
| Capa de salida | Linear(128, 1) — logit sin sigmoid |
| Embedding antibiotico | Embedding(96, 49) |

### 6.4 Proceso de entrenamiento

| Hiperparametro | Valor |
|---|---|
| Optimizador | Adam (lr=0.001) |
| Funcion de perdida | BCEWithLogitsLoss con pos_weight=0.6302 |
| Batch size | 32 |
| Epocas maximas | 200 |
| Early stopping | Patience=20 sobre val_loss |
| Checkpoint | Mejor F1 en validacion |
| Semilla aleatoria | 42 |
| Dispositivo | CUDA (GPU) |

El `pos_weight` (0.6302) se recalcula sobre el conjunto de entrenamiento (no sobre el dataset completo) para reflejar la proporcion real de clases en el split de entrenamiento.

**Criterio de early stopping:** se monitorea la perdida de validacion. Si no mejora durante 20 epocas consecutivas, se detiene el entrenamiento. El checkpoint se guarda basado en el mejor F1 de validacion (metrica de interes clinico), no la loss.

### 6.5 Proceso de evaluacion

1. Al finalizar el entrenamiento, se carga el mejor checkpoint (mejor val F1).
2. Se evalua sobre el conjunto de validacion para encontrar el **umbral optimo de decision** que maximiza F1 (en vez de usar 0.5 por defecto, ya que con clases desbalanceadas puede no ser optimo).
3. Se evalua sobre el conjunto de prueba con el umbral calibrado en validacion.

**Metricas de evaluacion:**
- **Accuracy:** proporcion de predicciones correctas.
- **Precision:** de los que se predijeron como Resistant, cuantos lo son realmente.
- **Recall (sensibilidad):** de los realmente Resistant, cuantos detecto el modelo. Metrica critica en AMR: un falso negativo (resistente clasificado como susceptible) puede llevar a un tratamiento ineficaz.
- **F1-score:** media armonica de precision y recall.
- **AUC-ROC:** area bajo la curva ROC, mide la calidad del ranking de probabilidades independientemente del umbral.

---

## 7. Preliminary Results and Performance Evaluation

### 7.1 Convergencia del entrenamiento

El modelo se entreno durante **75 epocas** antes de que el early stopping detuviera el entrenamiento (sin mejora en val_loss por 20 epocas). El mejor checkpoint se guardo en la epoca 71 con val_F1 = 0.8392.

**Curvas de entrenamiento:**

| Metrica | Epoca 1 | Epoca 75 (final) | Mejor checkpoint (epoca 71) |
|---|---|---|---|
| Train loss | 0.4270 | 0.2842 | 0.2844 |
| Val loss | 0.3923 | 0.3072 | 0.3144 |
| Val F1 | 0.7828 | 0.8227 | 0.8392 |

La brecha entre train_loss (0.28) y val_loss (0.31) al final del entrenamiento es moderada (~0.03), lo que indica un **ligero sobreajuste controlado**. Las curvas de loss muestran convergencia gradual sin divergencia abrupta, confirmando que el early stopping detuvo el entrenamiento en un punto adecuado.

### 7.2 Metricas en el conjunto de prueba

| Metrica | Valor |
|---|---|
| Accuracy | 0.8215 |
| Precision | 0.8238 |
| Recall | 0.9031 |
| **F1** | **0.8616** |
| **AUC-ROC** | **0.9098** |
| Umbral optimo | 0.3973 |
| Loss | 0.3062 |

### 7.3 Evaluacion contra criterios de exito

| Criterio | Objetivo | Resultado | Estado |
|---|---|---|---|
| F1 >= 0.85 | Superar baseline | 0.8616 | Cumplido |
| Recall >= 0.90 | Minimizar falsos negativos | 0.9031 | Cumplido |
| Superar baseline F1 | > 0.7366 | 0.8616 (+17.0%) | Cumplido |
| Sin sobreajuste evidente | Gap train-val < 0.05 | 0.03 | Cumplido |

### 7.4 Analisis de resultados

**Fortalezas:**
- El recall de 0.9031 indica que el modelo detecta correctamente el 90.3% de los casos resistentes, lo cual es critico en un contexto clinico donde un falso negativo (resistente clasificado como susceptible) puede derivar en tratamiento ineficaz.
- El AUC-ROC de 0.9098 demuestra buena capacidad discriminativa del modelo a traves de diferentes umbrales.
- El F1 de 0.8616 supera el baseline por antibiotico (0.7366) en un 17%, confirmando que los k-meros aportan informacion genomica predictiva mas alla de la senal epidemiologica.

**Diagnostico de sobreajuste/subajuste:**
- **Sobreajuste:** la brecha train-val loss es ~0.03, indicando un ligero sobreajuste controlado por el dropout (0.3) y el early stopping.
- **Subajuste:** no se observa — tanto train loss como val loss descienden consistentemente durante las primeras 50 epocas.
- El umbral optimo (0.3973) es menor que 0.5, lo cual es esperado dado el desbalance de clases y refleja la calibracion adecuada del modelo con pos_weight.

**Limitaciones:**
- El modelo no captura dependencias posicionales en el genoma, ya que los histogramas de k-meros son un resumen composicional sin orden.
- Los 27 antibioticos con desbalance extremo pueden tener predicciones sesgadas hacia la clase mayoritaria de cada antibiotico.
- El modelo trata cada par (genoma, antibiotico) de forma independiente; no modela interacciones de resistencia cruzada entre antibioticos.

---

## 8. Ethical Considerations

### 8.1 Datos y privacidad

Los datos utilizados provienen exclusivamente de repositorios publicos (BV-BRC/PATRIC) con datos bacterianos, no humanos. No se maneja informacion confidencial de pacientes ni datos personales sensibles. Los genomas son de aislados bacterianos y las etiquetas fenotipicas son resultados estandarizados de pruebas de susceptibilidad antimicrobiana de laboratorio.

### 8.2 Impacto del modelo

Un modelo de prediccion de RAM puede tener impacto clinico indirecto si se usara en un entorno real. Los falsos negativos (clasificar un aislado resistente como susceptible) representan el mayor riesgo etico, ya que podrian derivar en la seleccion de antibioticos ineficaces. Por esta razon, el recall es la metrica prioritaria y se exige >= 0.90.

Este proyecto es exclusivamente academico y de investigacion. No se pretende su uso directo en decisiones clinicas sin validacion experimental adicional.

### 8.3 Reproducibilidad

Se emplea semilla aleatoria fija (42) para garantizar reproducibilidad de resultados. Todo el codigo fuente, datos procesados y resultados se documentan en el repositorio del proyecto.

---

## 9. Web Links to Source Code and Explanatory Video

### Source Code

El codigo fuente, documentacion y resultados del proyecto estan organizados en el siguiente repositorio de GitHub:

**Repositorio:** https://github.com/sebastianrodriguez1115/neural-networks-project

#### Estructura del repositorio

| Directorio/Archivo | Descripcion |
|---|---|
| `src/bvbrc/` | Paquete de descarga de datos desde BV-BRC |
| `src/data_pipeline/` | Preprocesamiento: etiquetas, k-meros, splits |
| `src/dataset.py` | `AMRDataset` — Dataset de PyTorch |
| `src/mlp_model.py` | `AMRMLP` — Perceptron multicapa |
| `src/train/` | Entrenamiento: evaluacion, loop, early stopping |
| `main.py` | Punto de entrada CLI (Typer) |
| `results/mlp/` | Resultados: modelo, metricas, graficas |
| `docs/` | Documentacion del proyecto |

#### Comandos principales

```bash
# Instalar dependencias
uv sync

# Ejecutar pipeline de datos
uv run python main.py prepare-data

# Entrenar MLP
uv run python main.py train-mlp

# Entrenar con hiperparametros personalizados
uv run python main.py train-mlp --epochs 50 --batch-size 64 --lr 0.0005
```

### Explanatory Video

*[Pendiente de agregar enlace al video explicativo]*

---

## 10. Conclusions

### Logros de esta entrega

1. Se implemento un pipeline de datos completo y reproducible que descarga, limpia, transforma y divide datos genomicos de 5 especies ESKAPE con 96 antibioticos.
2. Se entreno un MLP superficial que alcanza **F1=0.8616** y **Recall=0.9031** en el conjunto de prueba, cumpliendo ambos criterios de exito (F1 >= 0.85, Recall >= 0.90).
3. Los resultados confirman que los histogramas de frecuencias de k-meros capturan informacion genomica suficiente para predecir resistencia antimicrobiana con un rendimiento significativamente superior al baseline epidemiologico (+17% en F1).

### Limitaciones

- **Sin contexto posicional:** los histogramas de k-meros son un resumen composicional que no preserva el orden de las secuencias genomicas. Mutaciones puntuales en regiones criticas podrian no estar bien representadas.
- **Antibioticos con datos escasos:** 27 antibioticos con desbalance extremo y algunos con muy pocos registros limitan la capacidad del modelo para generalizarlos.
- **Especie ausente:** *Enterobacter spp.* no esta en el dataset, limitando la cobertura del grupo ESKAPE completo.
- **Sin validacion externa:** el modelo se evaluo solo en datos de BV-BRC; no se probo con datos de otras fuentes [10].

### Trabajo futuro (Entrega final)

- **Implementar BiGRU + Attention:** red recurrente bidireccional con mecanismo de atencion de Bahdanau que procese los histogramas como secuencia, potencialmente capturando relaciones entre los distintos ordenes de k-meros.
- **Comparacion sistematica:** evaluar si la complejidad adicional del modelo profundo se traduce en mejoras significativas en F1 y recall.
- **Analisis de pesos de atencion:** examinar que ordenes de k-meros (k=3, 4 o 5) son mas informativos para la prediccion.

---

## 11. Use of Artificial Intelligence Tools

Claude Code (Anthropic) fue utilizado como asistente de programacion a lo largo del desarrollo de este proyecto. Especificamente:

- **Planificacion:** Se utilizo para discutir y refinar la arquitectura del proyecto, decisiones de diseño del pipeline de datos, y la estructura de la documentacion.
- **Implementacion:** Se uso para asistir en la escritura de codigo en Python/PyTorch, incluyendo el pipeline de datos, el modelo MLP, el ciclo de entrenamiento y las funciones de evaluacion. Todo el codigo generado fue revisado, editado y validado por el autor.
- **Documentacion:** Se utilizo para asistir en la redaccion y estructuracion de la documentacion del proyecto y este reporte. El contenido fue revisado y editado para asegurar precision y estilo.
- **Debugging:** Se utilizo para asistir en la identificacion y resolucion de errores durante el desarrollo.

El autor es responsable de la informacion presentada, incluyendo la verificacion de los resultados experimentales y la precision de las afirmaciones tecnicas.

---

## References

[1] C. A. Munita and C. A. Arias, "Mechanisms of Antibiotic Resistance," *Microbiology Spectrum*, 2016. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC4888801/

[2] D. Boolchandani, E. D'Souza, and G. Dantas, "Sequencing-based methods and resources to study antimicrobial resistance," *Nature Reviews Genetics*, 2019. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC6525649/

[3] J. Ren et al., "Prediction of antimicrobial resistance based on whole-genome sequencing and machine learning," *Bioinformatics*, 2021. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC8722762/

[4] J. Davies and D. Davies, "Origins and evolution of antibiotic resistance," *Microbiology and Molecular Biology Reviews*, 2010. Available: https://journals.asm.org/doi/10.1128/mmbr.00016-10

[5] J. M. A. Blair et al., "Molecular mechanisms of antibiotic resistance," *Nature Reviews Microbiology*, 2015. Available: https://www.nature.com/articles/nrmicro3380

[6] C. U. Koser et al., "Whole-genome sequencing to control antimicrobial resistance," *Trends in Genetics*, 2014. Available: https://www.cell.com/trends/genetics/fulltext/S0168-9525(14)00114-0

[7] G. Werner et al., "Antimicrobial susceptibility prediction from genomes," *Trends in Microbiology*, 2024. Available: https://www.cell.com/trends/microbiology/fulltext/S0966-842X(24)00052-0

[8] Y. Hu et al., "Assessing computational predictions of antimicrobial resistance," *Briefings in Bioinformatics*, 2024. Available: https://academic.oup.com/bib/article/25/3/bbae206/7665136

[9] M. Nguyen et al., "Using machine learning to predict antimicrobial MICs and associated genomic features for nontyphoidal Salmonella," *Journal of Clinical Microbiology*, 2019. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC6355527/

[10] J. Nsubuga et al., "Generalizability of machine learning in predicting antimicrobial resistance in E. coli," *BMC Genomics*, 2024. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC10946178/

[11] L. Lugo and E. Barreto-Hernandez, "A Recurrent Neural Network approach for whole genome bacteria identification," *Applied Artificial Intelligence*, 2021. Available: https://doi.org/10.1080/08839514.2021.1922842

[12] R. Rizzo et al., "A Deep Learning Approach to DNA Sequence Classification," in *Computational Intelligence Methods for Bioinformatics and Biostatistics*, Springer, 2016. Available: https://link.springer.com/chapter/10.1007/978-3-319-44332-4_10

[13] A. Fiannaca et al., "Deep learning models for bacteria taxonomic classification of metagenomic data," *BMC Bioinformatics*, 2018. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC6069770/
