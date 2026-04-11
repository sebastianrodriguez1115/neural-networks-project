"""
MLPDataset — Carga de histogramas de k-meros concatenados (1344D).
"""

from pathlib import Path

import numpy
import torch

from models.base_dataset import BaseAMRDataset


class MLPDataset(BaseAMRDataset):
    """
    Dataset para el modelo MLP. Carga vectores 1D (histogramas concatenados).
    """

    def _load_genome_data(
        self, data_dir: Path, split_ids: set[str]
    ) -> dict[str, torch.Tensor]:
        """Carga vectores 1D (.npy) del directorio 'mlp/'."""
        vectors_dir = data_dir / "mlp"
        genome_data: dict[str, torch.Tensor] = {}

        for gid in split_ids:
            npy_path = vectors_dir / f"{gid}.npy"
            vec = numpy.load(npy_path)
            genome_data[gid] = torch.from_numpy(vec).float()

        return genome_data
