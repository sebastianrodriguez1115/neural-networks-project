"""
TokenBiGRUDataset — Carga de secuencias de tokens IDs para Token BiGRU.
"""

from pathlib import Path

import numpy
import torch

from models.base_dataset import BaseAMRDataset


class TokenBiGRUDataset(BaseAMRDataset):
    """
    Dataset para el modelo Token BiGRU. Carga secuencias de tokens (enteros).
    """

    def _load_genome_data(
        self, data_dir: Path, split_ids: set[str]
    ) -> dict[str, torch.Tensor]:
        """Carga secuencias de tokens (.npy) del directorio 'token_bigru/'."""
        vectors_dir = data_dir / "token_bigru"
        genome_data: dict[str, torch.Tensor] = {}

        for gid in split_ids:
            npy_path = vectors_dir / f"{gid}.npy"
            vec = numpy.load(npy_path)
            # Los tokens son IDs para nn.Embedding, cargados como long [Mikolov13]
            genome_data[gid] = torch.from_numpy(vec).long()

        return genome_data
