"""
BiGRUDataset — Carga de matrices 2D (1024, 3) para BiGRU.
"""

from pathlib import Path

import numpy
import torch

from models.base_dataset import BaseAMRDataset


class BiGRUDataset(BaseAMRDataset):
    """
    Dataset para el modelo BiGRU. Carga matrices 2D de histogramas apilados.
    """

    def _load_genome_data(
        self, data_dir: Path, split_ids: set[str]
    ) -> dict[str, torch.Tensor]:
        """Carga matrices 2D (.npy) del directorio 'bigru/'."""
        vectors_dir = data_dir / "bigru"
        genome_data: dict[str, torch.Tensor] = {}

        for gid in split_ids:
            npy_path = vectors_dir / f"{gid}.npy"
            mat = numpy.load(npy_path)
            genome_data[gid] = torch.from_numpy(mat).float()

        return genome_data
