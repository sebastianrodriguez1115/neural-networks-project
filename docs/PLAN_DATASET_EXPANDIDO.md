# PLAN_DATASET_EXPANDIDO.md - Dataset AMR expandido

## Objetivo

Expandir el dataset más allá de ESKAPE usando registros AMR públicos de BV-BRC que cumplan las mismas condiciones de calidad definidas en el EDA. El objetivo experimental es medir si más genomas bacterianos con etiquetas limpias mejoran la generalización del modelo, manteniendo el modelo actual como baseline congelado.

## Cambio de narrativa

La narrativa pasaría de:

> Predicción de resistencia antimicrobiana en organismos ESKAPE.

a:

> Predicción de resistencia antimicrobiana bacteriana a partir de WGS, usando ESKAPE como benchmark inicial y un dataset expandido como prueba de generalización multi-organismo.

Este cambio solo debe hacerse si el experimento expandido demuestra valor. Hasta entonces, ESKAPE sigue siendo el benchmark primario.

## Regla de congelamiento

No modificar ni sobrescribir los artefactos actuales:

- `data/processed/`
- `data/raw/fasta/`
- `results/hier_set/`
- `results/hier_set/best_model.pt`
- `results/hier_set/metrics.json`

Todo el trabajo expandido debe vivir en rutas nuevas:

- `data/expanded/raw/`
- `data/expanded/fasta/`
- `data/expanded/processed/`
- `results/hier_set_expanded/`

El modelo actual `HierSet` queda congelado como baseline. El experimento expandido entrena un nuevo checkpoint desde cero, con la misma arquitectura e hiperparámetros salvo que se documente explícitamente lo contrario.

## Filtros obligatorios

El dataset expandido debe conservar las condiciones del EDA:

- `evidence == "Laboratory"`
- `resistant_phenotype in {"Resistant", "Susceptible"}`
- `laboratory_typing_method == "Broth dilution"`
- FASTA disponible en BV-BRC
- longitud total del genoma `>= 500_000` bp
- eliminar pares contradictorios `(genome_id, antibiotic)` con R y S simultáneos
- eliminar duplicados consistentes, conservando el primer registro
- conservar solo antibióticos con al menos `20` registros después de limpieza

## Riesgo central

Los k-meros codifican taxonomía. Al incluir no-ESKAPE, el modelo puede aprender priors por especie o género en vez de mecanismos de resistencia.

Mitigación obligatoria:

- mantener un test ESKAPE congelado;
- reportar métricas por taxon o especie cuando sea posible;
- reportar métricas por antibiótico;
- comparar contra el baseline `HierSet` actual, no solo contra el nuevo split expandido.

## Fase 1 - Descarga de etiquetas AMR expandidas

Agregar soporte para descargar etiquetas AMR sin limitarse a `ESKAPE_TAXON_IDS`.

Cambios propuestos:

- `src/bvbrc/amr.py`: permitir construir queries sin filtro `taxon_id`.
- `main.py`: agregar opciones al comando `download-amr`:
  - `--all-taxa`, descarga todos los taxones con AMR válido.
  - `--exclude-eskape`, opcional, excluye taxones ESKAPE para estudiar solo aporte externo. Si la API no soporta bien la negación por taxon, aplicar este filtro localmente después de descargar.
  - `--typing-method "Broth dilution"`, opcional, filtra en descarga si la API lo soporta de forma estable.

Comando esperado:

```bash
uv run python main.py download-amr \
  --all-taxa \
  --output data/expanded/raw/amr_labels_all_taxa.csv
```

Si la API no filtra bien `laboratory_typing_method`, descargar con `evidence == Laboratory` y aplicar el filtro de método en la limpieza local.

## Fase 2 - Limpieza previa antes de descargar FASTA

No conviene descargar FASTA de genomas que luego serán descartados por labels.

Agregar un comando liviano:

```bash
uv run python main.py prepare-labels-for-download \
  --labels data/expanded/raw/amr_labels_all_taxa.csv \
  --output data/expanded/processed/labels_for_download.csv
```

Debe aplicar, en este orden:

- filtro `Broth dilution`;
- eliminación de pares contradictorios;
- deduplicación por `(genome_id, antibiotic)`;
- filtro final de antibióticos con mínimo `20` registros después de la limpieza.

Salida mínima:

- `data/expanded/processed/labels_for_download.csv`
- resumen en consola con registros iniciales, registros finales, genomas únicos y antibióticos conservados.

## Fase 3 - Descarga de FASTA expandido

Descargar FASTA solo para los `genome_id` que sobreviven la limpieza previa.

Comando esperado:

```bash
uv run python main.py download-genomes \
  --labels data/expanded/processed/labels_for_download.csv \
  --output-dir data/expanded/fasta \
  --n-jobs 8
```

Mejora recomendada:

- usar `--n-jobs` con un valor conservador; en la ejecución inicial se usaron `8` workers sin fallos reportados;
- mantener reintentos y omitir FASTA ya existentes para poder reanudar descargas largas.

## Fase 4 - Pipeline expandido con splits bloqueados

Ejecutar el pipeline en un output separado.

Comando esperado:

```bash
uv run python main.py prepare-data \
  --labels data/expanded/raw/amr_labels_all_taxa.csv \
  --fasta-dir data/expanded/fasta \
  --output-dir data/expanded/processed \
  --locked-splits data/processed/splits.csv \
  --n-jobs -1
```

Política de split:

- todo genoma presente en `data/processed/splits.csv` conserva su split original;
- el test ESKAPE actual nunca puede entrar a train ni val;
- los genomas nuevos se dividen 70/15/15 con `random_seed = 42`, estratificados por fenotipo mayoritario;
- el archivo resultante `data/expanded/processed/splits.csv` debe incluir una columna `split_source` con valores como `locked` o `new`.

Razón: esta política permite evaluar si los no-ESKAPE ayudan al benchmark ESKAPE sin contaminar la comparación.

Ejecución completada:

- `cleaned_labels.csv`: 282716 registros después de filtro genómico;
- `splits.csv`: 37678 genomas válidos (`train=26373`, `val=5651`, `test=5654`);
- `split_source`: 9060 genomas `locked`, 28618 genomas `new`;
- 0 cambios de split para los 9060 genomas del split ESKAPE original;
- 214 genomas descartados por `below_min_length`;
- features MLP/BiGRU generadas para 37678 genomas.

## Fase 5 - Extracción de features HierSet

Como el mejor modelo actual es `HierSet`, el primer experimento expandido debe usar la misma representación `hier_bigru/`.

Comando:

```bash
uv run python main.py prepare-hier \
  --data-dir data/expanded/processed \
  --fasta-dir data/expanded/fasta \
  --n-jobs -1
```

No entrenar `HierSet v2` en la primera pasada. Primero hay que probar si más datos ayudan a la arquitectura ganadora actual.

Ejecución completada:

- `data/expanded/processed/hier_bigru/`: 37678 archivos `.npy`;
- shape validada por muestra: `(256, 256)`, `float32`.

## Fase 6 - Entrenamiento del modelo expandido

Entrenar un nuevo checkpoint desde cero con la misma arquitectura `HierSet`.

Comando:

```bash
uv run python main.py train-hier-set \
  --data-dir data/expanded/processed \
  --output-dir results/hier_set_expanded
```

Hiperparámetros iniciales:

- `epochs = 100`
- `batch_size = 32`
- `lr = 0.001`
- `patience = 15`
- `pos_weight_scale = 2.5`
- `weight_decay = 1e-3`

Estos valores deben mantenerse iguales a `HierSet` v1 para que el primer resultado sea comparable.

## Fase 7 - Evaluación comparativa

Agregar un comando o script de evaluación de checkpoint:

```bash
uv run python main.py evaluate-hier-set-checkpoint \
  --checkpoint results/hier_set_expanded/best_model.pt \
  --data-dir data/expanded/processed \
  --split test \
  --subset locked
```

El subconjunto `locked` debe seleccionar, dentro de `data/expanded/processed/splits.csv`, los genomas con `split_source == "locked"`. Para reproducir el test ESKAPE congelado, usar `split == "test"` y `split_source == "locked"`.

Ejecución completada:

- comando implementado: `evaluate-hier-set-checkpoint`;
- test expandido completo: F1=0.8129, Recall=0.8075, AUC=0.9355, n=43506;
- test ESKAPE congelado (`split_source=locked`): F1=0.8874, Recall=0.9039, AUC=0.9384, n=12532;
- test nuevo (`split_source=new`): F1=0.7361, Recall=0.7130, AUC=0.9184, n=30974.
- métricas por grupo generadas en `results/hier_set_expanded/`:
  - `metrics_test_all_by_antibiotic.csv`, `metrics_test_all_by_taxon.csv`;
  - `metrics_test_locked_by_antibiotic.csv`, `metrics_test_locked_by_taxon.csv`;
  - `metrics_test_new_by_antibiotic.csv`, `metrics_test_new_by_taxon.csv`.

Lectura provisional: el modelo expandido no mejora de forma clara el benchmark ESKAPE congelado y generaliza mal, en F1/Recall, a los genomas nuevos con el umbral global calibrado.

Control adicional con BiGRU expandido:

- configuración: `train-bigru --data-dir data/expanded/processed --output-dir results/bigru_expanded --batch-size 128 --pos-weight-scale 2.5 --patience 15`;
- test expandido completo: F1=0.7683, Recall=0.7830, AUC=0.9009, n=43506;
- test ESKAPE congelado (`split_source=locked`): F1=0.8481, Recall=0.8771, AUC=0.8888, n=12532;
- test nuevo (`split_source=new`): F1=0.6878, Recall=0.6908, AUC=0.8847, n=30974.

Lectura: la BiGRU sufre más que HierSet al expandir el dataset; no rescata el problema de generalización y confirma que el sesgo secuencial/pseudo-secuencial no es adecuado para esta expansión.

Ajuste posterior de `pos_weight_scale` en HierSet expandido:

- `scale=1.0`: test completo F1=0.8273, locked F1=0.8973, new F1=0.7539, AUC locked=0.9463;
- `scale=1.5`: test completo F1=0.8152, locked F1=0.8883, new F1=0.7408;
- `scale=2.5`: test completo F1=0.8129, locked F1=0.8874, new F1=0.7361.

Decisión provisional: `results/hier_set_expanded_pw1_0/` es el mejor checkpoint expandido actual. El ajuste mejora el test ESKAPE congelado respecto al baseline histórico, pero `new` sigue claramente más difícil.

Evaluaciones obligatorias:

1. `HierSet` congelado sobre test ESKAPE actual.
2. `HierSet_expanded` sobre el mismo test ESKAPE congelado.
3. `HierSet_expanded` sobre test expandido completo.
4. Métricas por taxon o especie.
5. Métricas por antibiótico.

Métricas mínimas:

- F1
- Recall
- Precision
- AUC-ROC
- accuracy
- umbral usado

## Criterios de éxito

El experimento expandido se considera exitoso si cumple al menos una de estas condiciones sin degradaciones graves:

- mejora F1 o AUC sobre el test ESKAPE congelado;
- mantiene F1 similar pero mejora recall clínico con caída pequeña de precision;
- mejora métricas en antibióticos o taxones con baja cobertura sin dañar especies ESKAPE principales.

Se considera resultado negativo si:

- solo mejora el test expandido, pero baja el test ESKAPE congelado;
- mejora el promedio global, pero empeora fuertemente una especie ESKAPE;
- sube AUC pero baja recall por debajo del criterio clínico;
- el modelo parece depender de priors taxonómicos y no de señales AMR generalizables.

## Tests requeridos

Agregar tests unitarios y de integración para:

- query AMR sin filtro `taxon_id`;
- query AMR con filtro opcional de `laboratory_typing_method`;
- exclusión opcional de taxones ESKAPE;
- comando de limpieza previa para descarga;
- split con `locked_splits`;
- garantía de que genomas del test ESKAPE original no pasan a train;
- pipeline expandido escribiendo en directorios separados;
- evaluación de checkpoint sin reentrenar.

## Documentación a actualizar

Al completar la implementación o el experimento, revisar:

- `docs/2_eda.md`: EDA expandido y comparación contra ESKAPE.
- `docs/3_data_pipeline.md`: nuevas rutas y comandos.
- `docs/4_models.md`: aclarar que la arquitectura no cambia.
- `docs/5_experiments.md`: nuevo experimento de dataset expandido.
- `docs/PROGRESS.md`: nueva fase de implementación.
- `docs/CHANGELOG.md`: resumen de cambios y resultados.
- `README.md`: actualizar alcance solo si el experimento justifica el cambio de narrativa.
- `AGENTS.md`: actualizar si el dataset expandido pasa a ser parte estable del proyecto.

## Secuencia recomendada de implementación

1. Implementar descarga AMR all-taxa sin tocar el flujo ESKAPE existente.
2. Implementar limpieza previa para obtener `labels_for_download.csv`.
3. Implementar splits bloqueados en el pipeline.
4. Agregar tests para los tres puntos anteriores.
5. Descargar FASTA expandido.
6. Ejecutar `prepare-data` y `prepare-hier` en rutas `data/expanded/`.
7. Entrenar `HierSet` expandido.
8. Evaluar contra test ESKAPE congelado y test expandido.
9. Documentar resultado, aunque sea negativo.

## Decisión pendiente

Antes de ejecutar descargas masivas, decidir si se quiere:

- `all-taxa`: ESKAPE más no-ESKAPE en un solo dataset expandido;
- `non-eskape-only`: solo no-ESKAPE para medir transferencia;
- ambos, si hay tiempo y espacio en disco.

La opción recomendada para el primer experimento es `all-taxa`, manteniendo el test ESKAPE congelado como referencia principal.
