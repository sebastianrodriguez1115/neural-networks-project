"""
HierBiGRUDataset — Dataset para el modelo Hierarchical BiGRU.

Implementa la carga de histogramas segmentados (tiled histograms) para 
alimentar la arquitectura recurrente jerárquica.
"""

import logging
from pathlib import Path

import numpy
import torch

from models.base_dataset import BaseAMRDataset
from data_pipeline.constants import HIER_KMER_DIM, HIER_N_SEGMENTS

logger = logging.getLogger(__name__)


class HierBiGRUDataset(BaseAMRDataset):
    """
    Dataset para el modelo Hierarchical BiGRU.

    Carga histogramas segmentados (HIER_N_SEGMENTS segmentos × 256 k-meros)
    desde archivos .npy generados durante la fase de preprocesamiento (prepare-hier).
    """

    def _load_genome_data(
        self, data_dir: Path, split_ids: set[str]
    ) -> dict[str, torch.Tensor]:
        """
        Carga los histogramas segmentados para los genomas del split.

        Parámetros:
            data_dir: directorio con los outputs del pipeline.
            split_ids: IDs de genomas asignados a este split.

        Retorna:
            Diccionario genome_id → tensor [HIER_N_SEGMENTS, 256] (float32).
        """
        hier_dir = data_dir / "hier_bigru"
        if not hier_dir.is_dir():
            raise FileNotFoundError(
                f"No se encontró el directorio de features jerárquicas: {hier_dir}. "
                "Asegúrate de ejecutar 'python main.py prepare-hier' primero."
            )

        logger.info(f"Cargando histogramas segmentados (split={self._split})...")
        genome_data: dict[str, torch.Tensor] = {}

        for genome_id in sorted(split_ids):
            npy_path = hier_dir / f"{genome_id}.npy"
            matrix = numpy.load(npy_path)
            expected = (HIER_N_SEGMENTS, HIER_KMER_DIM)
            if matrix.shape != expected:
                raise ValueError(
                    f"Shape inesperado para {genome_id}: {matrix.shape}. "
                    f"Se esperaba {expected}. Regenera las features con prepare-hier."
                )
            genome_data[genome_id] = torch.from_numpy(matrix).float()

        return genome_data
