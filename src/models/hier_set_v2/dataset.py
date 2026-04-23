"""
HierSetV2Dataset — Dataset para el modelo AMRHierSetV2.

Carga los histogramas multi-escala (k=3,4,5) segmentados generados por
prepare-hier-multi y almacenados en data/processed/hier_set_v2/.
"""

import logging
from pathlib import Path

import numpy
import torch

from data_pipeline.constants import HIER_KMER_DIM_MULTI, HIER_N_SEGMENTS
from models.base_dataset import BaseAMRDataset

logger = logging.getLogger(__name__)


class HierSetV2Dataset(BaseAMRDataset):
    """
    Dataset para AMRHierSetV2.

    Carga histogramas multi-escala segmentados (HIER_N_SEGMENTS × 1344)
    desde archivos .npy en hier_set_v2/.
    """

    def _load_genome_data(
        self, data_dir: Path, split_ids: set[str]
    ) -> dict[str, torch.Tensor]:
        hier_dir = data_dir / "hier_set_v2"
        if not hier_dir.is_dir():
            raise FileNotFoundError(
                f"No se encontró el directorio de features multi-escala: {hier_dir}. "
                "Ejecuta 'python main.py prepare-hier-multi' primero."
            )

        logger.info(f"Cargando histogramas multi-escala (split={self._split})...")
        genome_data: dict[str, torch.Tensor] = {}

        expected = (HIER_N_SEGMENTS, HIER_KMER_DIM_MULTI)
        for genome_id in sorted(split_ids):
            npy_path = hier_dir / f"{genome_id}.npy"
            matrix = numpy.load(npy_path)
            if matrix.shape != expected:
                raise ValueError(
                    f"Shape inesperado para {genome_id}: {matrix.shape}. "
                    f"Se esperaba {expected}. Regenera las features con prepare-hier-multi."
                )
            genome_data[genome_id] = torch.from_numpy(matrix).float()

        return genome_data
