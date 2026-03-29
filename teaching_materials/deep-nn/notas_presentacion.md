# Plan de Presentacion De Redes Neuronales de Grafos a Arquitecturas Fractales

**Audiencia:** Estudiantes de pregrado (Ingeniería de Sistemas, Ciencias de la Computación o áreas afines).  
**Duración:** 60 minutos.  
**Objetivo Principal:** Comprender el mecanismo de paso de mensajes en las Redes Neuronales de Grafos (GNN), identificar el problema de la profundidad (*over-smoothing*) y analizar cómo las FractalNets ofrecen una solución estructural matemática sin depender de conexiones residuales.  

---

## ⏱️ Módulo 1: Fundamentos de Redes Neuronales de Grafos (15 minutos)
**Objetivo:** Establecer la intuición detrás del procesamiento de datos no euclidianos y su formalización matemática.

### 1. Introducción al dominio de grafos
* **Contraste de estructuras:** Diferencia entre datos en cuadrículas euclidianas (imágenes procesadas por CNNs) y datos relacionales.
* **Definición formal:** Un grafo se define como $G = (V, E)$, donde $V$ representa el conjunto de nodos y $E$ el conjunto de aristas. En este dominio, el número de vecinos varía por nodo y no existe un orden topológico estricto (como "arriba" o "abajo").

### 2. El paradigma del Paso de Mensajes (Message Passing)
* **Intuición conceptual:** Analogía de una red social donde cada individuo actualiza su estado interno basándose en la información agregada de sus conexiones directas.
* **Invarianza a las permutaciones:** Explicación de por qué la operación de agregación debe ser agnóstica al orden (concepto de *Neighborhood Aggregation*, cf. Aggarwal §10.3.1). Ejemplo intuitivo: "Si ordeno a tus amigos de Facebook alfabéticamente o por edad, el resumen de tu entorno social debería ser exactamente el mismo". Demostración de por qué una concatenación simple falla (depende del orden), y por qué se requieren operaciones como la suma ($\sum$), el promedio ($\mu$) o el máximo ($\max$).
* **Formalización de la capa:** La ecuación general de actualización para un nodo $v$ en la capa $k$:
  $$h_v^{(k)} = \sigma \left( W^{(k)} \cdot \text{COMBINE} \left( h_v^{(k-1)}, \text{AGG} \left( \{ h_u^{(k-1)} \}_{u \in \mathcal{N}(v)} \right) \right) \right)$$
  *Donde $W^{(k)}$ es la matriz de pesos aprendible y $\sigma$ representa una función de activación no lineal (ej. ReLU).*

---

## ⏱️ Módulo 2: El Desafío de la Profundidad (10 minutos)
**Objetivo:** Identificar los límites arquitectónicos de las GNNs convencionales al incrementar el número de capas.

### 1. El problema del Over-smoothing (Sobre-suavizado)
* **Expansión del campo receptivo:** Análisis de cómo cada capa adicional amplía el radio de vecindad consultado. Tras suficientes iteraciones (ej. 10-20 capas), el nodo integra información de prácticamente todo el grafo.
* **Suavizado Laplaciano:** Explicación matricial. Si la agregación se representa mediante la matriz de adyacencia normalizada $\tilde{A}$, la actualización tras $k$ capas es proporcional a $\tilde{A}^k H$. Matemáticamente, cuando $k \to \infty$, las características convergen al autovector principal de la matriz, provocando que todos los nodos posean representaciones casi idénticas y pierdan discriminabilidad.

### 2. Soluciones tradicionales
* **Conexiones residuales (*Skip Connections*):** Mecanismo de las ResNets para sumar la identidad del mapa de características ($x^{(l-1)}$) a la salida de la capa, anclando las representaciones originales. En GNNs se aplica el mismo principio sumando $h_v^{(l-1)}$ a la salida de la capa de paso de mensajes.
* **Transición:** Presentación del problema de investigación: ¿Es posible diseñar una red ultra-profunda que evite esta degradación sin utilizar sumas residuales de identidad? *Nota:* La arquitectura que se presentará a continuación (FractalNet) fue desarrollada originalmente para CNNs en visión por computadora, no para GNNs. Sin embargo, el problema de fondo —la dificultad de entrenar redes muy profundas— es compartido, y la solución estructural es transferible conceptualmente.

---

## ⏱️ Módulo 3: FractalNets y la Autosimilitud (15 minutos)
**Objetivo:** Introducir la topología de red basada en fractales propuesta por Larsson et al. (2017).

### 1. Ruptura con el paradigma residual
* Las FractalNets generan profundidad expansiva mediante una regla de diseño recursiva y autosimilar, en lugar de apilar capas secuencialmente.
* **Apoyo visual:** Se recomienda proyectar la Figura 1 del paper de FractalNet (diagrama de la regla de expansión y la red completa con $B=5, C=4$). La visualización de cómo la columna izquierda se profundiza mientras la derecha se mantiene superficial hace que la regla recursiva $C=1, C=2, \ldots$ sea inmediatamente comprensible.

### 2. La regla de expansión estructural
* **Caso base ($C=1$):** Una operación simple (una sola capa de procesamiento — convolucional en el paper original, pero puede ser cualquier tipo de capa, ej. agregación de vecinos en GNNs).
* **Primera expansión ($C=2$):** División del flujo de información.
  * *Ruta en serie:* Aplicación del caso base dos veces de forma consecutiva (aumenta la profundidad).
  * *Ruta en paralelo:* Aplicación de una convolución simple (ruta de atajo superficial).
* **La Capa de Unión (*Join Layer*):** Punto de convergencia de las rutas. El *Join Layer* calcula la media elemento a elemento de las rutas activas: $\frac{1}{N} \sum_{i=1}^N x_i$. La diferencia con las ResNets no es solo aritmética (media vs. suma), sino **estructural**: en ResNets existe una señal privilegiada (la identidad, *pass-through*) a la que se le suma un residuo; en FractalNet, todas las entradas al *Join* son salidas de convoluciones y ninguna tiene estatus de identidad. Esto estabiliza la magnitud de las activaciones independientemente de cuántos caminos se unan.

---

## ⏱️ Módulo 4: Formalización Matemática y Regularización (15 minutos)
**Objetivo:** Consolidar el conocimiento matemático de la arquitectura fractal y su método de entrenamiento.

### 1. La Ecuación de Expansión Fractal
* Formalización de un bloque fractal $f_C(z)$:
  * **Base:** $f_1(z) = \text{conv}(z)$
  * **Regla recursiva:**
    $$f_{C+1}(z) = \text{Join}( (f_C \circ f_C)(z), \text{conv}(z) )$$
* *Explicación analítica:* El término $(f_C \circ f_C)$ representa la composición matemática de la ruta profunda, garantizando un crecimiento exponencial de la profundidad máxima por bloque ($2^{C-1}$), mientras que $\text{conv}(z)$ mantiene un camino de longitud 1. Al apilar $B$ bloques fractales con capas de *pooling* entre ellos, la profundidad total de la red es $B \cdot 2^{C-1}$ (ej. $B=5, C=4 \Rightarrow 40$ capas de convolución).

### 2. Regularización mediante Drop-path
* **El problema de la co-adaptación:** Con múltiples caminos paralelos, existe el riesgo de que la red dependa de una sola ruta y las demás generen ruido.
* **Mecanismo Drop-path:** Extensión conceptual del *Dropout*, pero a escala diferente. El Dropout tradicional previene la co-adaptación de **activaciones individuales** (neuronas); el Drop-path previene la co-adaptación de **subrutas enteras** (caminos paralelos). Es la misma filosofía llevada de una escala "micro" a una escala "macro".
  * *Drop-path local:* En cada *Join Layer*, se descarta cada entrada $x_i$ con una probabilidad fija (15% en los experimentos del paper), asegurando que al menos una sobreviva.
  * *Drop-path global:* En cada mini-batch se utiliza un modelo mixto: con un 50% de probabilidad se aplica muestreo local y con el otro 50% se aplica muestreo global. Cuando se selecciona el modo global, se restringe toda la red a una única columna (ej. forzar el uso exclusivo de la ruta más profunda o la más superficial), promoviendo que cada columna sea un predictor competente de forma independiente.
* **Conclusión:** Esta técnica obliga a cada subred a ser competente por sí misma. En la fase de inferencia, permite extraer subredes superficiales de alta velocidad sin sacrificar drásticamente la precisión.

---

## ⏱️ Módulo 5: Cierre — Conexión con GNNs y Discusión (5 minutos)
**Objetivo:** Cerrar el arco narrativo conectando FractalNets de vuelta al dominio de grafos.

### 1. Recapitulación
* FractalNets fueron diseñadas para CNNs y demuestran que la profundidad efectiva, no las conexiones residuales, es la clave para entrenar redes ultra-profundas.
* El over-smoothing en GNNs y la degradación por profundidad en CNNs comparten la misma raíz: al apilar capas, la información se difumina y las representaciones colapsan.

### 2. Pregunta abierta para la audiencia
* ¿Cómo se podría adaptar la topología fractal (rutas paralelas de distinta profundidad + *Join Layers*) al paradigma de paso de mensajes en grafos?
* **Guía para el debate:** Relacionar la "profundidad" de la red con los "saltos" (*hops*) en el grafo:
  * En una CNN fractal, una ruta profunda aplica muchas convoluciones seguidas.
  * Extrapolando a GNNs, una "ruta profunda" significaría hacer agregaciones de vecinos de $K$-saltos (vecinos de vecinos de vecinos...), mientras que la ruta superficial paralela solo miraría a los vecinos directos (1 salto).
  * Un *Join Layer* fractal en grafos promediaría lo que dice el vecindario lejano con lo que dice el vecindario directo, sin darle estatus de "identidad" a ninguno — mitigando el over-smoothing al no depender exclusivamente de la ruta profunda.

### 3. Idea clave
* La contribución fundamental de FractalNets no es una arquitectura específica, sino un **principio de diseño**: la autosimilitud estructural como alternativa a las conexiones residuales para habilitar profundidad.

---

## 📚 Bibliografía y Referencias de Apoyo

1. **FractalNets:** Larsson, G., Maire, M., & Shakhnarovich, G. (2017). *FractalNet: Ultra-Deep Neural Networks without Residuals*. ICLR 2017.
2. **GNNs e Invarianza:** Hamilton, W., Ying, Z., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs (GraphSAGE)*. NIPS.
3. **Over-smoothing en GNNs:** Li, Q., Han, Z., & Wu, X. M. (2018). *Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning*. AAAI.
4. **Fundamentos Convolucionales:** Aggarwal, C. C. (2023). *Neural Networks and Deep Learning* (2nd ed.). Springer.