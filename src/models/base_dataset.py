"""
BaseAMRDataset — Clase base abstracta para los datasets de predicción de AMR.

Contiene la lógica común para cargar etiquetas, splits e índices de antibióticos.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy
import pandas
import torch
from torch.utils.data import Dataset


class BaseAMRDataset(Dataset, ABC):
    """
    Clase base abstracta para datasets de AMR.

    Maneja la carga de metadatos comunes (splits, labels, antibiotic_index)
    y delega la carga de datos genómicos específicos a las subclases.
    """

    def __init__(self, data_dir: str | Path, split: str) -> None:
        """
        Inicializa el dataset cargando metadatos comunes.

        Parámetros:
            data_dir: directorio con los outputs del pipeline.
            split: partición a cargar ("train", "val" o "test").
        """
        data_dir = Path(data_dir)
        self._split = split

        # Leer splits y filtrar por partición
        splits = pandas.read_csv(data_dir / "splits.csv", dtype={"genome_id": str})
        split_ids = set(splits[splits["split"] == split]["genome_id"])

        # Leer etiquetas limpias y filtrar por genomas del split
        labels = pandas.read_csv(
            data_dir / "cleaned_labels.csv", dtype={"genome_id": str}
        )
        labels = labels[labels["genome_id"].isin(split_ids)].reset_index(drop=True)

        # Mapeo antibiótico → índice entero
        antibiotic_index = pandas.read_csv(data_dir / "antibiotic_index.csv")
        self._antibiotic_to_idx = dict(
            zip(antibiotic_index["antibiotic"], antibiotic_index["index"])
        )
        self.n_antibiotics = len(self._antibiotic_to_idx)

        # Delegar la carga de datos genómicos a la subclase
        genome_data = self._load_genome_data(data_dir, split_ids)

        # Construir listas de muestras
        self._vectors: list[torch.Tensor] = []
        self._antibiotic_idxs: list[int] = []
        self._labels: list[float] = []

        for _, row in labels.iterrows():
            self._vectors.append(genome_data[row["genome_id"]])
            self._antibiotic_idxs.append(self._antibiotic_to_idx[row["antibiotic"]])
            self._labels.append(
                1.0 if row["resistant_phenotype"] == "Resistant" else 0.0
            )

    @abstractmethod
    def _load_genome_data(
        self, data_dir: Path, split_ids: set[str]
    ) -> dict[str, torch.Tensor]:
        """Carga los datos genómicos específicos del modelo."""
        pass

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Devuelve (genome_data, antibiotic_idx, label) como tensores CPU."""
        genome_tensor = self._vectors[idx]
        ab_idx = torch.tensor(self._antibiotic_idxs[idx], dtype=torch.long)
        label = torch.tensor(self._labels[idx], dtype=torch.float32)
        return genome_tensor, ab_idx, label

    @staticmethod
    def load_pos_weight(data_dir: str | Path) -> float:
        """Lee pos_weight desde train_stats.json."""
        stats_path = Path(data_dir) / "train_stats.json"
        stats = json.loads(stats_path.read_text())
        return stats["pos_weight"]
