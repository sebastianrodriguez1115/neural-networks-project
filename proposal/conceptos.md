# Glosario de Conceptos del Proyecto

---

## Biología y Resistencia Antimicrobiana

### Resistencia antimicrobiana (RAM / AMR)
Capacidad de un microorganismo de sobrevivir o crecer en presencia de un antibiótico que normalmente lo inhibiría o eliminaría. Es una amenaza creciente para la salud pública global. Sus dos motores principales son las **mutaciones genéticas** y la **transferencia horizontal de genes**.

### Transferencia horizontal de genes (THG)
Proceso por el cual un organismo transfiere material genético a otro que **no es su descendiente directo**, es decir, fuera de la reproducción vertical (de progenitor a hijo). Es uno de los principales mecanismos de diseminación de resistencia antimicrobiana porque permite que genes de resistencia pasen entre bacterias de distintas especies en cuestión de minutos.

**Mecanismos principales en bacterias:**
| Mecanismo | Descripción |
|---|---|
| Transformación | La bacteria capta ADN libre del ambiente |
| Transducción | Un bacteriófago transporta ADN de una bacteria a otra |
| Conjugación | Dos bacterias se conectan físicamente y transfieren plásmidos |

### Determinantes de resistencia
Genes o mutaciones específicas que confieren resistencia a un antibiótico. Ejemplos: el gen *mecA* (resistencia a meticilina en *S. aureus*) o el gen *blaNDM* (resistencia a carbapenémicos).

### Genotipo vs. fenotipo
- **Genotipo:** información contenida en la secuencia de ADN del organismo.
- **Fenotipo:** característica observable resultante, por ejemplo, si la bacteria es resistente o susceptible a un antibiótico.

### Identificación basada en genotipo
Detectar determinantes de resistencia **leyendo la secuencia de ADN** de la bacteria, en lugar de realizar una prueba de laboratorio fenotípica (exponer la bacteria al antibiótico y observar si sobrevive). Nuestro modelo hace exactamente esto: infiere el **fenotipo** (resistente/susceptible) a partir del **genotipo** (secuencia del genoma).

| Enfoque | Cómo funciona |
|---|---|
| Fenotípico | Cultivar la bacteria + agregarle el antibiótico → ¿sobrevive? |
| Genotípico | Secuenciar el ADN → ¿tiene el gen de resistencia? |

### WGS (Whole Genome Sequencing / Secuenciación del Genoma Completo)
Técnica que determina la secuencia completa de ADN de un organismo. Permite la identificación genotípica de determinantes de resistencia a escala, sin necesidad de pruebas de laboratorio para cada antibiótico.

### Susceptibilidad antimicrobiana
Grado en que un microorganismo es afectado por un antibiótico. Las pruebas de susceptibilidad (AST, *Antimicrobial Susceptibility Testing*) clasifican típicamente a las bacterias en tres categorías: susceptible, intermedia y resistente. En este proyecto se excluyen las etiquetas intermedias para formular el problema como clasificación binaria.

### Variabilidad intraespecífica
Diferencias genéticas entre individuos de una **misma especie**. En el contexto del proyecto, es la razón por la que la predicción de resistencia es más difícil que la identificación de especies: la señal relevante son variaciones sutiles dentro de una especie, no diferencias amplias entre especies distintas.

---

## Bases de Datos

### PATRIC (Pathosystems Resource Integration Center)
Base de datos pública de genomas bacterianos con anotaciones de susceptibilidad antimicrobiana. Es el candidato principal del proyecto para obtener los datos de entrenamiento.

### NDARO (National Database of Antibiotic Resistant Organisms)
Base de datos pública del NCBI con genomas bacterianos y datos de resistencia. Se considera como alternativa a PATRIC en el proyecto.

---

## Representación de Secuencias

### K-meros (k-mers)
Subcadenas de longitud fija *k* extraídas de una secuencia de ADN. Por ejemplo, para la secuencia `ATCGA` con k=3, los k-meros son `ATC`, `TCG`, `CGA`. Son la unidad básica de representación en ambos modelos del proyecto.

### Histograma de frecuencia de k-meros
Representación composicional de un genoma que cuenta cuántas veces aparece cada k-mero posible, ignorando su posición. Para k=5 hay 4⁵ = 1024 k-meros posibles. En el proyecto se concatenan los histogramas para k=3, 4 y 5 como entrada al modelo superficial (MLP). No preserva el orden ni el contexto posicional.

### Tokenización y embeddings de k-meros
Para el modelo profundo (RNN), el genoma se representa como una **secuencia ordenada** de k-meros, preservando su posición. Cada k-mero único se mapea a un vector numérico denso (**embedding**) aprendido durante el entrenamiento, de forma similar a como se representan palabras en modelos de lenguaje natural.

---

## Arquitecturas de Redes Neuronales

### MLP (Perceptrón Multicapa / Multilayer Perceptron)
Red neuronal con capas completamente conectadas. En el proyecto actúa como **línea de base superficial**: recibe el vector de histograma de k-meros concatenado y predice si el aislamiento es resistente o susceptible. Permite establecer si la composición del genoma por sí sola es suficiente para la tarea.

### RNN (Red Neuronal Recurrente / Recurrent Neural Network)
Arquitectura de red neuronal diseñada para procesar **secuencias ordenadas**. Mantiene un estado oculto que se actualiza en cada paso temporal, permitiendo capturar dependencias entre elementos de la secuencia. En el proyecto procesa la secuencia de k-meros del genoma.

### GRU (Gated Recurrent Unit) y LSTM (Long Short-Term Memory)
Variantes de RNN que incorporan mecanismos de "puertas" para controlar qué información se retiene o descarta a lo largo de la secuencia. Mitigan el problema del desvanecimiento del gradiente en secuencias largas. El proyecto usa una de estas dos variantes para el modelo profundo.

### RNN Bidireccional
Extensión de la RNN que procesa la secuencia en **ambas direcciones** (de inicio a fin y de fin a inicio) y combina los estados ocultos resultantes. Permite que cada posición tenga contexto tanto de lo que precede como de lo que sigue en la secuencia.

### Mecanismo de atención (Attention)
Componente que asigna un **peso de importancia** a cada posición de la secuencia, permitiendo que el modelo se enfoque en las regiones más informativas. En el proyecto se aplica sobre los estados ocultos de la RNN bidireccional. Potencialmente mejora la exactitud y la interpretabilidad del modelo, ya que las regiones con mayor peso podrían corresponder a zonas genómicas asociadas a la resistencia.

---

## Entrenamiento y Evaluación

### Particiones estratificadas (Stratified splits)
División del conjunto de datos en subconjuntos de entrenamiento, validación y prueba, garantizando que la **proporción de clases** (susceptible/resistente) se mantenga similar en cada partición. Importante en conjuntos desequilibrados.

### Función de pérdida de entropía cruzada (Cross-entropy loss)
Función de pérdida estándar para tareas de clasificación. Mide la discrepancia entre la distribución de probabilidad predicha por el modelo y la distribución real de las etiquetas.

### AdaGrad (2011)
Algoritmo de optimización que adapta la tasa de aprendizaje **individualmente por parámetro**: los parámetros que reciben gradientes grandes se actualizan menos, y los que reciben gradientes pequeños se actualizan más. Su limitación es que acumula todos los gradientes históricos sin límite, por lo que la tasa de aprendizaje se vuelve cada vez más pequeña con el tiempo hasta casi detenerse.

### RMSProp (2012)
Corrige el problema de AdaGrad usando una **media móvil** de los gradientes recientes en lugar de acumular todos los históricos. Así la tasa de aprendizaje se adapta por parámetro sin congelarse con el tiempo.

### Optimizador Adam
Algoritmo de optimización que combina las ventajas de AdaGrad y RMSProp, añadiendo además **momento** (inercia en la dirección del gradiente). Es el optimizador más utilizado en la práctica para entrenar redes neuronales profundas por su robustez y velocidad de convergencia.

| Algoritmo | Adaptación por parámetro | Memoria histórica limitada | Momento |
|---|---|---|---|
| AdaGrad | Sí | No | No |
| RMSProp | Sí | Sí | No |
| **Adam** | **Sí** | **Sí** | **Sí** |

### Desequilibrio de clases
Situación en la que una clase (por ejemplo, "resistente") está representada en mucho menor o mayor número que la otra ("susceptible") en el conjunto de datos. Es frecuente en datos de RAM y representa un riesgo porque el modelo puede aprender a predecir siempre la clase mayoritaria y aun así obtener una alta exactitud, sin haber aprendido nada útil. Se aborda mediante **pesos de clase** o **sobremuestreo**.

### Pesos de clase (*class weights*)
Estrategia para manejar el desequilibrio de clases que modifica la función de pérdida asignando un costo mayor a los errores cometidos en la clase minoritaria. Por ejemplo, si hay 9 veces más susceptibles que resistentes, los errores sobre la clase resistente tienen un peso 9x mayor, forzando al modelo a prestarle más atención. No modifica los datos, solo el entrenamiento.

### Sobremuestreo (*oversampling*)
Estrategia para manejar el desequilibrio de clases que genera o duplica ejemplos de la clase minoritaria hasta equilibrar las proporciones. La técnica más común es **SMOTE** (*Synthetic Minority Oversampling Technique*), que crea ejemplos sintéticos interpolando entre ejemplos reales existentes de la clase minoritaria.

| Estrategia | Cómo funciona | Ventaja | Desventaja |
|---|---|---|---|
| Pesos de clase | Modifica la función de pérdida | Simple, no altera los datos | Solo ajusta el entrenamiento |
| Sobremuestreo | Genera ejemplos sintéticos | El modelo ve más ejemplos diversos | Puede introducir ruido |

En el proyecto se aplica cualquiera de las dos estrategias **solo si** el análisis del dataset muestra un desequilibrio significativo, ya que la proporción de susceptibles y resistentes varía según el antibiótico analizado.

### Métricas de evaluación

| Métrica | Definición |
|---|---|
| **Exactitud** (Accuracy) | Proporción de predicciones correctas sobre el total |
| **Precisión** (Precision) | De los predichos como resistentes, ¿cuántos realmente lo son? |
| **Recall** (Exhaustividad) | De los resistentes reales, ¿cuántos detectó el modelo? |
| **Puntuación F** (F-score) | Media armónica de precisión y recall; útil con clases desequilibradas |

### Validación experimental (laboratorio húmedo / *wet lab*)
Verificación de resultados mediante experimentos físicos con muestras biológicas reales: cultivo de bacterias, pruebas de susceptibilidad antimicrobiana (AST), extracción de ADN, etc. En el proyecto se usa el término **validación experimental** para referirse a este tipo de confirmación, que está fuera del alcance del trabajo por ser un proyecto puramente computacional.

### Hiperparámetros
Parámetros de configuración del modelo que no se aprenden durante el entrenamiento, sino que deben fijarse antes de entrenar. Ejemplos: tasa de aprendizaje, número de capas, número de neuronas por capa, tamaño de los embeddings, tamaño del batch, número de epochs.

### Búsqueda exhaustiva de hiperparámetros (*hyperparameter search*)
Proceso de encontrar los mejores valores de hiperparámetros probando sistemáticamente combinaciones posibles. La técnica más común es **Grid Search**, que evalúa todas las combinaciones de un conjunto predefinido de valores. El número de combinaciones crece exponencialmente con la cantidad de hiperparámetros, y cada combinación requiere entrenar el modelo completo, lo que lo hace costoso en tiempo y cómputo.

```
tasa de aprendizaje: [0.001, 0.01, 0.1]
capas:              [1, 2, 3]
neuronas:           [64, 128, 256]
→ 3 × 3 × 3 = 27 entrenamientos completos
```

En el proyecto está fuera del alcance porque consumiría una parte desproporcionada del tiempo disponible (2–3 meses). En su lugar se usan valores tomados de la literatura, que ya han demostrado funcionar en tareas similares.
