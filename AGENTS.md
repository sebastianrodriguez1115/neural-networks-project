# AGENTS.md — Instrucciones para agentes IA

## Qué es este proyecto
Proyecto académico: predecir resistencia antimicrobiana (AMR) en bacterias a partir de secuencias genómicas completas (WGS) usando redes neuronales. Clasificación binaria (Resistant / Susceptible) sobre organismos ESKAPE. El antibiótico entra como feature al modelo (embedding aprendido).

**Modelos implementados (en orden de rendimiento):**
- **HierSet** ← mejor modelo: encoder de conjunto sobre 256 segmentos de histograma k=4 con cross-attention condicionada por antibiótico. F1=0.8900, AUC=0.9368, Recall=0.9088.
- **MLP (baseline):** vector de frecuencias k-meros concatenados (k=3,4,5), 1344 dims. F1=0.8600, AUC=0.9035.
- **BiGRU + Attention:** histogramas paddeados → matriz `[batch, 1024, 3]`. F1=0.8566, AUC=0.8998.
- **MultiBiGRU:** 3 streams order-independent (k=3,4,5) con `bin_importance`. F1=0.8514, AUC=0.8944.
- **HierBiGRU:** mismos 256 segmentos que HierSet pero con BiGRU (sesgo secuencial). F1=0.8307, AUC=0.8539. No competitivo.

## Stack
- **Python 3.10+**, **PyTorch**, **uv** (gestor de paquetes — usar `uv run` y `uv add`)
- Librerías: `biopython`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

## Archivos clave
| Archivo | Rol |
|---|---|
| `src/bvbrc/` | Paquete de descarga de datos desde BV-BRC (genomas FASTA + etiquetas AMR) |
| `src/data_pipeline/` | Paquete de preprocesamiento: etiquetas, k-meros, splits |
| `src/models/base_dataset.py` | `BaseAMRDataset` — Dataset base de PyTorch para todos los modelos |
| `src/models/hier_set/` | `AMRHierSet` — **mejor modelo** (F1=0.89, AUC=0.94) |
| `src/models/mlp/` | `AMRMLP` — MLP baseline |
| `src/train/` | Paquete de entrenamiento: evaluación, loop, early stopping |
| `main.py` | Punto de entrada CLI (Typer) |

## Documentación de referencia
- `docs/1_environment.md` — stack, estructura, instalación y comandos CLI
- `docs/2_eda.md` — hallazgos del EDA (pos_weight, desbalance, baseline benchmark)
- `docs/3_data_pipeline.md` — pipeline de datos (decisivo para `src/data_pipeline/`)
- `docs/4_models.md` — arquitecturas y decisiones de diseño
- `docs/5_experiments.md` — experimentos y criterio de éxito
- `docs/PLAN_MLP.md` — plan de implementación detallado para MLP y entrenamiento

## Documentos a mantener al hacer cambios
Estos documentos pueden quedar desactualizados cuando se modifica código o se completan tareas. Revisarlos siempre:

| Documento | Cuándo actualizar |
|---|---|
| `docs/PROGRESS.md` | Al completar un ítem — marcar `[x]` y mover `[~]` según corresponda |
| `docs/CHANGELOG.md` | Al terminar una sesión con cambios significativos |
| `src/README.md` | Al agregar/renombrar/eliminar módulos en `src/`, o al agregar comandos CLI |
| `README.md` | Al cambiar la estructura del proyecto o agregar dependencias mayores |
| `AGENTS.md` | Al cambiar archivos clave, convenciones, o agregar docs de referencia |
| `docs/1_environment.md` | Al agregar nuevos comandos CLI o cambiar instrucciones de instalación |
| `docs/4_models.md` | Al modificar arquitecturas de modelos |
| `docs/PLAN_MLP.md` | Al completar pasos del plan o cambiar decisiones de diseño |

## Convenciones
- **Código en inglés** — nombres de funciones, variables, clases y tests
- **Docstrings y comentarios en español** — para que sean accesibles
- Prose y documentación (docs/, CHANGELOG.md, README.md) en español
- No implementar BiRNN Variante B (secuencia ordenada) a menos que se solicite explícitamente
- Semilla aleatoria fija: `random_seed = 42`
- Split: 70/15/15 estratificado por `genome_id` (no por registro)
- Pérdida: `BCEWithLogitsLoss` con `pos_weight` para manejar desbalance
- Early stopping: patience sobre val_F1 (tanto checkpoint como early stopping monitorizan val_F1); ReduceLROnPlateau también sobre val_F1

## Al terminar una tarea
1. Marcar como completada en `docs/PROGRESS.md`
2. Si el cambio es significativo, agregar entrada en `docs/CHANGELOG.md`
3. Revisar la tabla "Documentos a mantener" arriba y actualizar los que apliquen
