# Borrador del Informe Final: Predicción de RAM desde Genomas Completos

> Estado del borrador: versión de trabajo. Pendiente: nombres de autores, URL del repositorio, URL del video explicativo y figuras finales. La versión IEEE LaTeX está en `main.tex`.

## Título

Predicción de resistencia antimicrobiana a partir de secuencias genómicas completas usando redes neuronales

## Resumen

La resistencia antimicrobiana (RAM) es un problema clínico crítico porque clasificar una bacteria resistente como susceptible puede llevar a un tratamiento inefectivo. Este proyecto evalúa modelos de redes neuronales para predecir RAM binaria a partir de secuenciación del genoma completo (WGS, por *Whole-Genome Sequencing*), es decir, la lectura de la secuencia de ADN completa o casi completa de una bacteria. Usamos archivos FASTA, un formato de texto para almacenar secuencias de ADN, y etiquetas fenotípicas públicas de BV-BRC. Cada muestra es un par genoma-antibiótico, y el antibiótico entra al modelo como un embedding aprendido. Primero se construyó un benchmark ESKAPE y se compararon un perceptrón multicapa (MLP), modelos BiGRU y modelos jerárquicos basados en histogramas locales de k-meros. El mejor modelo ESKAPE fue HierSet, un encoder de conjunto invariante a permutaciones sobre 256 segmentos de histogramas k=4 con atención condicionada por antibiótico. En el benchmark ESKAPE obtuvo F1=0.8900, Recall=0.9088 y AUC=0.9368; con un umbral clínico de 0.40 alcanzó Recall=0.9289 con una caída pequeña de F1. Luego se expandió el dataset más allá de ESKAPE, manteniendo congelado el test ESKAPE original. El dataset expandido mejoró el test ESKAPE congelado cuando se recalibró `pos_weight_scale` a 1.0: F1=0.8973, Recall=0.9261 y AUC=0.9463. Sin embargo, el rendimiento en nuevos taxones siguió siendo menor (F1=0.7539). La conclusión principal es que HierSet es la mejor arquitectura probada y que más datos pueden mejorar el benchmark ESKAPE si se ajusta correctamente el manejo del desbalance. La generalización multitaxón sigue limitada por cambios de distribución, confounding taxonómico y calibración.

## 1. Objetivo y alcance

El objetivo del proyecto es predecir si un genoma bacteriano es resistente o susceptible a un antibiótico dado usando features derivadas de WGS y redes neuronales. En este informe, WGS significa secuenciación del genoma completo: no se analiza solo un gen aislado, sino la composición de la secuencia de ADN disponible para todo el genoma bacteriano. Cada registro corresponde a un par `(genome_id, antibiotic)` con una etiqueta binaria: `Resistant` o `Susceptible`.

El alcance inicial fue un benchmark sobre organismos ESKAPE usando datos de BV-BRC. Después se agregó un experimento expandido all-taxa para evaluar si más datos públicos de RAM mejoran la generalización sin contaminar el test ESKAPE original.

El proyecto compara tres familias de modelos:

- Baseline superficial: MLP sobre histogramas globales de k-meros.
- Modelos recurrentes profundos: BiGRU + Attention y variantes.
- Modelos jerárquicos tipo conjunto: HierSet sobre histogramas segmentados.

La arquitectura seleccionada es HierSet porque obtiene el mejor F1/AUC en ESKAPE y también es la más fuerte en los experimentos expandidos.

## 2. Contexto y fundamentos

La predicción de RAM desde WGS es un problema de clasificación supervisada. El modelo debe aprender relaciones entre contenido genómico y susceptibilidad antimicrobiana. El genoma puede contener genes de resistencia, mutaciones puntuales, elementos móviles y señales taxonómicas. Como los genomas son largos y variables, la representación de entrada es decisiva.

Los histogramas de k-meros son una representación práctica porque resumen composición de ADN sin requerir alineamiento ni anotación genética. Un k-mero es una subsecuencia de ADN de longitud `k`; por ejemplo, si `k=4`, `ATGC` es un k-mero. Un histograma de k-meros cuenta con qué frecuencia aparece cada posible subsecuencia. En este proyecto se usaron histogramas globales k=3,4,5 para MLP y BiGRU, e histogramas segmentados k=4 para HierSet.

Se probaron redes recurrentes y mecanismos de atención porque algunos trabajos procesan representaciones genómicas como secuencias. Sin embargo, nuestros resultados muestran que tratar bins de histogramas o segmentos genómicos como secuencias puede introducir supuestos de orden artificiales. HierSet elimina ese sesgo al tratar los segmentos como un conjunto y usar atención condicionada por antibiótico.

Conceptos clave usados: `BCEWithLogitsLoss` con `pos_weight`, early stopping, selección de umbral en validación, dropout, AdamW, gradient clipping para modelos recurrentes y AUC-ROC como métrica independiente del umbral.

## 3. Datos

Los datos provienen de BV-BRC:

- Etiquetas fenotípicas AMR desde `genome_amr`.
- Secuencias genómicas FASTA desde endpoints de genomas de BV-BRC.

El benchmark original incluyó organismos ESKAPE, excepto *Enterobacter spp.*, que no apareció de forma limpia en el endpoint AMR usado. El dataset raw ESKAPE inicial tenía 162,170 registros, 16,281 genomas únicos, 96 antibióticos y 5 taxones ESKAPE.

El dataset expandido all-taxa tuvo:

- Raw all-taxa combinado: 383,068 registros, 44,905 genomas, 581 taxones, 125 antibióticos.
- Raw no-ESKAPE: 220,898 registros, 28,624 genomas, 576 taxones, 104 antibióticos.
- Labels limpios para descarga FASTA: 282,997 registros, 37,892 genomas, 581 taxones, 100 antibióticos.
- Dataset efectivo después del filtro genómico: 282,716 registros, 37,678 genomas.

## 4. Limpieza y EDA

### 4.1 Reglas de limpieza

Las reglas finales se derivaron del EDA:

- Conservar evidencia de laboratorio.
- Conservar solo fenotipos binarios: `Resistant` y `Susceptible`.
- Conservar solo `laboratory_typing_method == "Broth dilution"`.
- Eliminar pares contradictorios `(genome_id, antibiotic)` con etiquetas R y S simultáneas.
- Deduplicar pares consistentes, conservando el primer registro.
- Conservar antibióticos con al menos 20 registros útiles finales.
- Exigir FASTA disponible.
- Filtrar genomas menores a 0.5 Mb.

En el dataset expandido, la limpieza eliminó:

- 75,465 registros con método distinto a `Broth dilution`.
- 1,833 pares contradictorios, equivalentes a 3,809 filas.
- 20,749 duplicados consistentes.
- 8 antibióticos de baja frecuencia, equivalentes a 48 filas.
- 214 genomas por longitud menor a 0.5 Mb.

### 4.2 Hallazgos EDA en ESKAPE

El benchmark ESKAPE original tenía un balance global cercano a 54% Resistant y 46% Susceptible. Sin embargo, el balance variaba mucho por especie y antibiótico. Por ejemplo, *A. baumannii* era mayoritariamente Resistant, mientras *S. aureus* era mayoritariamente Susceptible.

Esto crea un riesgo de confounding: los k-meros codifican identidad taxonómica, por lo que el modelo podría aprender priors por especie en vez de mecanismos de resistencia.

El baseline sin información genómica fue la clase mayoritaria por antibiótico:

- Accuracy=71.2%.
- Precision=0.7281.
- Recall=0.7453.
- F1=0.7366.

El criterio de éxito del proyecto fue superar F1=0.85 con recall clínicamente útil.

### 4.3 Hallazgos EDA en el dataset expandido

El dataset expandido cambió fuertemente la distribución:

- Dataset efectivo: 101,350 Resistant y 181,366 Susceptible.
- Balance global: 35.8% Resistant y 64.2% Susceptible.
- `pos_weight` en train: 1.7890.
- Subset `locked` ESKAPE: 61.3% Resistant.
- Subset `new`: 25.4% Resistant.

Este cambio de distribución es fuerte. Los nuevos taxones son mucho más susceptibles que el benchmark ESKAPE congelado. Por eso, la configuración de desbalance que funcionaba en ESKAPE no transfiere directamente.

El dataset expandido también tiene una cola taxonómica larga:

- 581 taxones.
- Mediana de registros por taxón: 9.
- Percentil 75: 16.
- Mayor taxón: 52,154 registros.

Los antibióticos más frecuentes también cambiaron. Aparecen `isoniazid`, `ethambutol` y `pyrazinamide`, lo que confirma que el dataset expandido no es simplemente "más ESKAPE", sino un problema multiorganismo más heterogéneo.

## 5. Ingeniería de features

### 5.1 Histogramas globales de k-meros

El MLP usó histogramas concatenados k=3,4,5:

- k=3: 64 bins.
- k=4: 256 bins.
- k=5: 1024 bins.
- Total: 1344 dimensiones.

Las features se normalizaron usando media y desviación estándar del conjunto de entrenamiento para evitar leakage.

### 5.2 Representación BiGRU

La BiGRU reutilizó los histogramas k=3,4,5, rellenó cada histograma hasta 1024 bins y apiló los tres k como columnas. Esto produce una matriz `[1024, 3]`. La representación permite usar una BiGRU, pero impone un orden pseudosecuencial sobre bins que no son biológicamente secuenciales.

### 5.3 Representación jerárquica segmentada

La representación jerárquica divide cada genoma en 256 segmentos contiguos y calcula un histograma k=4 por segmento. El resultado es una matriz `[256, 256]` por genoma. Esto preserva localidad genómica gruesa y cobertura completa.

La motivación principal para explorar esta arquitectura fue que la señal asociada a resistencia puede ocupar una fracción muy pequeña del genoma completo. Si el modelo recibe solo un histograma global, esa señal puede quedar diluida dentro de millones de bases. Al partir el genoma en secciones, el modelo evalúa regiones más pequeñas y puede aprender a asignar mayor peso a segmentos donde aparezcan patrones relevantes para un antibiótico.

HierSet trata estos 256 segmentos como un conjunto. Esta decisión es importante porque los genes de resistencia y elementos móviles no aparecen en posiciones fijas del genoma.

## 6. Modelos

### 6.1 MLP

El MLP recibe el vector normalizado de 1344 dimensiones, lo concatena con el embedding del antibiótico y usa dos capas ocultas con dropout.

Resultado ESKAPE:

- F1=0.8600.
- Recall=0.9165.
- AUC=0.9035.

Esto confirmó que la composición global de k-meros contiene señal fuerte para RAM.

### 6.2 BiGRU + Attention

La BiGRU procesa la matriz `[1024, 3]` con una GRU bidireccional y atención de Bahdanau. Luego concatena el contexto con el embedding del antibiótico.

Mejor resultado ESKAPE:

- F1=0.8566.
- Recall=0.9032.
- AUC=0.8998.

Cumple el criterio de recall, pero no supera al MLP. La causa probable es que el modelo trata el orden de bins como una secuencia, aunque no tenga significado biológico directo.

### 6.3 MultiBiGRU / encoder multi-stream sin orden

Este modelo procesa k=3,4,5 en streams separados usando un encoder sin dependencias secuenciales entre bins. Luego fusiona los streams con gates condicionados por antibiótico.

Resultado:

- F1=0.8514.
- Recall=0.8925.
- AUC=0.8944.

Fue interpretable, pero no competitivo frente a HierSet.

### 6.4 HierBiGRU

HierBiGRU procesa la representación segmentada `[256, 256]` como una secuencia de segmentos.

Resultado:

- F1=0.8307.
- Recall=0.8788.
- AUC=0.8539.

Fue un resultado negativo. Confirmó que segmentar el genoma no basta si el modelo impone un sesgo secuencial incorrecto.

### 6.5 HierSet

HierSet es el mejor modelo. Trata los 256 segmentos como un conjunto y usa cross-attention condicionada por antibiótico:

`score(segmento, antibiótico) = h_segmento · q_antibiótico / sqrt(D)`

Esto permite atender cualquier segmento sin asumir adyacencia o posición fija.

Mejor resultado ESKAPE:

- F1=0.8900.
- Recall=0.9088.
- Precision=0.8720.
- AUC=0.9368.

Resultado con umbral clínico θ=0.40:

- F1=0.8876.
- Recall=0.9289.
- Precision=0.8498.
- AUC=0.9368.

Este fue el modelo seleccionado.

### 6.6 HierSet v2

HierSet v2 agregó multi-head attention e histogramas segmentados multiescala k=3,4,5. Esto aumentó la dimensión por segmento de 256 a 1344.

Resultado:

- F1=0.8895.
- Recall=0.8971.
- AUC=0.9366.

Fue un resultado negativo: no mejoró a HierSet v1 y redujo el recall.

## 7. Protocolo de entrenamiento y evaluación

Protocolo compartido:

- Split por `genome_id`, no por registro, para evitar leakage.
- Split 70/15/15 train/val/test.
- Semilla: 42.
- Pérdida: `BCEWithLogitsLoss` con `pos_weight` calculado sobre train.
- Umbral seleccionado en validación para maximizar F1.
- Early stopping y checkpoint sobre F1 de validación.
- Métricas: F1, Recall, Precision, AUC-ROC y Accuracy.

Para experimentos expandidos:

- Los splits ESKAPE originales se congelaron con `locked_splits`.
- Los genomas nuevos se dividieron por separado.
- `splits.csv` incluye `split_source` con valores `locked` y `new`.
- La evaluación se reportó para `all`, `locked` y `new`.

## 8. Resultados principales

### 8.1 Benchmark ESKAPE

| Modelo | F1 | Recall | Precision | AUC | Nota |
|---|---:|---:|---:|---:|---|
| MLP | 0.8600 | 0.9165 | 0.8100 | 0.9035 | Baseline fuerte |
| BiGRU + Attention | 0.8566 | 0.9032 | -- | 0.8998 | Cumple recall |
| MultiBiGRU | 0.8514 | 0.8925 | -- | 0.8944 | Recall borderline |
| HierBiGRU | 0.8307 | 0.8788 | -- | 0.8539 | Resultado negativo |
| HierSet | 0.8900 | 0.9088 | 0.8720 | 0.9368 | Mejor ESKAPE |
| HierSet θ=0.40 | 0.8876 | 0.9289 | 0.8498 | 0.9368 | Umbral clínico |
| HierSet v2 | 0.8895 | 0.8971 | -- | 0.9366 | No mejora v1 |

### 8.2 Dataset expandido: resultado inicial

El primer entrenamiento expandido reutilizó `pos_weight_scale=2.5` de la configuración ESKAPE.

| Evaluación | F1 | Recall | Precision | AUC | Umbral |
|---|---:|---:|---:|---:|---:|
| all | 0.8129 | 0.8075 | 0.8184 | 0.9355 | 0.7302 |
| locked | 0.8874 | 0.9039 | 0.8716 | 0.9384 | 0.7302 |
| new | 0.7361 | 0.7130 | 0.7608 | 0.9184 | 0.7302 |

Esto no mejoró claramente el test ESKAPE congelado y tuvo bajo F1/Recall en nuevos taxones, aunque el AUC siguió siendo razonable.

### 8.3 Dataset expandido: ajuste de `pos_weight_scale`

El EDA expandido mostró que `pos_weight` de train subió a 1.7890. Multiplicarlo por 2.5 daba un peso efectivo de 4.4725, demasiado agresivo para un dataset donde `new` es mayoritariamente susceptible. Por eso se probaron `pos_weight_scale=1.0` y `1.5`.

| Scale | Subset | F1 | Recall | Precision | AUC | Umbral |
|---:|---|---:|---:|---:|---:|---:|
| 1.0 | all | 0.8273 | 0.8255 | 0.8290 | 0.9421 | 0.5274 |
| 1.0 | locked | 0.8973 | 0.9261 | 0.8702 | 0.9463 | 0.5274 |
| 1.0 | new | 0.7539 | 0.7271 | 0.7828 | 0.9260 | 0.5274 |
| 1.5 | all | 0.8152 | 0.8299 | 0.8009 | 0.9363 | 0.5675 |
| 1.5 | locked | 0.8883 | 0.9221 | 0.8568 | 0.9377 | 0.5675 |
| 1.5 | new | 0.7408 | 0.7397 | 0.7419 | 0.9197 | 0.5675 |
| 2.5 | all | 0.8129 | 0.8075 | 0.8184 | 0.9355 | 0.7302 |
| 2.5 | locked | 0.8874 | 0.9039 | 0.8716 | 0.9384 | 0.7302 |
| 2.5 | new | 0.7361 | 0.7130 | 0.7608 | 0.9184 | 0.7302 |

La mejor configuración expandida fue HierSet entrenado sobre el dataset all-taxa con `pos_weight_scale=1.0`. Esta configuración fue la más equilibrada: mejoró el test ESKAPE congelado y también obtuvo el mejor F1 en los nuevos taxones frente a las variantes con `pos_weight_scale=1.5` y `2.5`.

Este es el primer resultado claramente positivo del dataset expandido sobre el test ESKAPE congelado:

- F1 sube de 0.8900 a 0.8973.
- Recall sube de 0.9088 a 0.9261.
- AUC sube de 0.9368 a 0.9463.

Sin embargo, `new` sigue por debajo de `locked`, por lo que el modelo no está resuelto para generalización multitaxón.

### 8.4 Control BiGRU en dataset expandido

BiGRU expandido se entrenó con `batch_size=128`, `pos_weight_scale=2.5` y `patience=15`.

| Evaluación | F1 | Recall | AUC |
|---|---:|---:|---:|
| all | 0.7683 | 0.7830 | 0.9009 |
| locked | 0.8481 | 0.8771 | 0.8888 |
| new | 0.6878 | 0.6908 | 0.8847 |

BiGRU no se benefició de la expansión y rindió peor que HierSet en todos los cortes. Esto refuerza que el sesgo pseudosecuencial no ayuda en este régimen multitaxón.

## 9. Discusión

El hallazgo arquitectónico principal es que los supuestos de orden importan, pero también importa la escala a la que se busca la señal. La resistencia puede depender de genes, mutaciones o elementos móviles que representan una proporción pequeña del genoma. Los histogramas globales de k-meros ya son fuertes, pero pueden diluir señales locales. Procesarlos con modelos recurrentes puede agregar sesgos de orden artificiales. HierSet funciona mejor porque combina información local por segmentos con una atención que no fuerza adyacencia ni posición fija, lo que ayuda al modelo a buscar patrones relevantes en regiones más pequeñas del genoma.

El hallazgo de datos principal es que más datos no garantizan mejor rendimiento. La expansión introdujo cambios fuertes en prior de clase, composición taxonómica y distribución de antibióticos. Cuando se recalibró `pos_weight_scale`, el entrenamiento expandido sí mejoró el test ESKAPE congelado. Pero `new` siguió teniendo F1/Recall bajos.

La diferencia entre AUC y F1 en `new` sugiere un problema de calibración y umbral. El AUC razonable indica que el modelo ordena muchos resistentes por encima de susceptibles. El F1 bajo indica que un único umbral global no transfiere bien entre `locked` y `new`, que tienen tasas base muy distintas.

## 10. Consideraciones éticas

La predicción de RAM tiene riesgo clínico. Un falso negativo, es decir, predecir susceptible cuando la bacteria es resistente, puede llevar a tratamiento inefectivo. Por eso el recall es una métrica prioritaria. Un falso positivo también puede causar daños al empujar el uso innecesario de antibióticos de reserva.

El modelo no debe interpretarse como herramienta diagnóstica clínica sin validación externa. Las etiquetas provienen de bases públicas y pueden contener sesgo de muestreo, desbalance taxonómico, estándares de laboratorio heterogéneos y sobrerrepresentación regional. Los experimentos expandidos muestran que el rendimiento varía por taxón y antibiótico.

Uso responsable requiere:

- Reportar métricas por subgrupo.
- Mantener splits de test congelados.
- Evitar afirmar generalización amplia sin validación externa.
- Documentar calibración y umbrales.
- Usar predicciones como apoyo, no como reemplazo de pruebas de laboratorio.

## 11. Limitaciones

- El modelo usa composición de k-meros, no anotaciones explícitas de genes de resistencia, plásmidos, mutaciones o elementos móviles.
- El dataset expandido tiene cola taxonómica larga, con muchos taxones de bajo soporte.
- El conjunto de antibióticos cambia con la expansión, dificultando comparaciones directas.
- No se midió calibración con ECE o reliability diagrams.
- No hubo validación externa fuera de BV-BRC.
- No se aplicaron pruebas estadísticas formales sobre diferencias de métricas.

## 12. Trabajo futuro

Siguientes pasos recomendados:

1. Agregar features biológicas explícitas desde CARD, ResFinder o AMRFinderPlus.
2. Medir calibración con ECE, Brier score y reliability diagrams.
3. Probar calibración de umbral por `split_source`, antibiótico y taxones de alto soporte.
4. Evaluar en un dataset externo fuera de BV-BRC.
5. Comparar contra baselines simples taxón-antibiótico para medir shortcuts taxonómicos.
6. Explorar modelos híbridos que combinen k-meros y anotaciones de genes.

## 13. Conclusiones

El mejor modelo del proyecto es HierSet, un encoder invariante a permutaciones sobre histogramas segmentados de k-meros, condicionado por el antibiótico. Su diseño responde a una hipótesis concreta: como la señal de resistencia puede ser local y ocupar una fracción pequeña del genoma, segmentar el genoma ayuda a que el modelo busque patrones en regiones más pequeñas en vez de depender solo de una composición global diluida. En el benchmark ESKAPE obtuvo el mejor rendimiento general y, al usar un umbral conservador de 0.40 orientado a aumentar recall, alcanzó Recall=0.9289 con pérdida mínima de F1. Este umbral no fue validado clínicamente; se usó como simulación de una preferencia por reducir falsos negativos.

El experimento expandido cambió la conclusión de forma matizada. Reutilizar `pos_weight_scale=2.5` produjo resultados débiles. Al recalibrar `pos_weight_scale=1.0`, el dataset expandido mejoró las métricas sobre el test ESKAPE congelado. Esto sugiere que más datos pueden ayudar, pero solo si el desbalance se reestima para la nueva distribución.

Al mismo tiempo, el rendimiento en nuevos taxones sigue siendo sustancialmente menor. Por lo tanto, la afirmación correcta no es que el proyecto resolvió RAM all-taxa. La afirmación fuerte es que HierSet fue la mejor arquitectura probada, que los datos expandidos mejoran ESKAPE bajo ponderación adecuada y que la generalización multitaxón robusta sigue siendo un problema abierto.

## 14. Repositorio y reproducibilidad

Pendiente: URL final del repositorio.

Comandos clave:

```bash
uv run python main.py eda-expanded
uv run python main.py train-hier-set --data-dir data/processed --output-dir results/hier_set
uv run python main.py train-hier-set --data-dir data/expanded/processed --output-dir results/hier_set_expanded_pw1_0 --pos-weight-scale 1.0
uv run python main.py evaluate-hier-set-checkpoint --checkpoint results/hier_set_expanded_pw1_0/best_model.pt --data-dir data/expanded/processed --split test --subset locked
```

## 15. Contribuciones del equipo



## 16. Uso de herramientas de IA

Se usaron herramientas de IA para apoyar generación de código, depuración, refactorización, seguimiento experimental y redacción del informe. La verificación humana incluyó revisar diffs, ejecutar tests unitarios, correr entrenamientos y evaluaciones, y contrastar métricas contra artefactos guardados. Las conclusiones científicas se basan en los experimentos ejecutados y documentados en el repositorio.
