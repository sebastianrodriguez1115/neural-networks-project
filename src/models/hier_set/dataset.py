"""
HierSetDataset — Dataset para el modelo AMRHierSet.

Reutiliza exactamente los mismos archivos .npy generados por prepare-hier
(hier_bigru/). No requiere preprocesamiento adicional.
"""

from models.hier_bigru.dataset import HierBiGRUDataset


class HierSetDataset(HierBiGRUDataset):
    """
    Dataset para AMRHierSet.

    Subclase de HierBiGRUDataset — los datos son idénticos (HIER_N_SEGMENTS
    segmentos × 256 k-meros). Solo cambia el nombre de clase para claridad en logs.
    """
