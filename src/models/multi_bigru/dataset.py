"""
MultiBiGRUDataset — Carga de histogramas segmentados (k=3, 4, 5) para Multi-Stream BiGRU.
"""

from pathlib import Path

import numpy
import torch

from data_pipeline.constants import KMER_OFFSETS
from models.base_dataset import BaseAMRDataset


class MultiBiGRUDataset(BaseAMRDataset):
    """
    Dataset para el modelo Multi-Stream BiGRU. Reutiliza los vectores MLP
    y los segmenta en __getitem__.
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

    def __getitem__(
        self, idx: int
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        """Devuelve ((k3, k4, k5), antibiotic_idx, label) como tensores CPU."""
        genome_tensor = self._vectors[idx]
        ab_idx = torch.tensor(self._antibiotic_idxs[idx], dtype=torch.long)
        label = torch.tensor(self._labels[idx], dtype=torch.float32)

        # Segmentar el vector MLP (1344,) en histogramas individuales [Lugo21, p. 647].
        # Se usa KMER_OFFSETS para asegurar consistencia.
        k3 = genome_tensor[KMER_OFFSETS[0] : KMER_OFFSETS[1]].unsqueeze(1)
        k4 = genome_tensor[KMER_OFFSETS[1] : KMER_OFFSETS[2]].unsqueeze(1)
        k5 = genome_tensor[KMER_OFFSETS[2] : KMER_OFFSETS[3]].unsqueeze(1)

        return (k3, k4, k5), ab_idx, label
