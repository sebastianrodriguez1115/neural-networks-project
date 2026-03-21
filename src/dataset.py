"""
AMRDataset — Dataset de PyTorch para predicción de resistencia antimicrobiana.

Pre-carga los vectores genómicos (.npy) en RAM y devuelve triples
(genome_vector, antibiotic_idx, label) listos para el DataLoader.
"""

import json
from pathlib import Path

import numpy
import pandas
import torch
from torch.utils.data import Dataset


class AMRDataset(Dataset):
    """
    Dataset para clasificación binaria de resistencia antimicrobiana.

    Cada muestra es un triple:
        - genome_vector (float32, 1344 dims): histograma de k-meros normalizado
        - antibiotic_idx (long): índice entero del antibiótico
        - label (float32): 1.0 = Resistant, 0.0 = Susceptible

    Los vectores genómicos se pre-cargan en RAM durante __init__ para
    eliminar I/O de disco durante el entrenamiento.
    """

    def __init__(self, data_dir: str | Path, split: str) -> None:
        """
        Inicializa el dataset cargando etiquetas, splits y vectores genómicos.

        Parámetros:
            data_dir: directorio con los outputs del pipeline (splits.csv,
                      cleaned_labels.csv, antibiotic_index.csv, mlp/)
            split: partición a cargar ("train", "val" o "test")
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

        # Pre-cargar vectores genómicos en RAM (un tensor por genome_id)
        mlp_dir = data_dir / "mlp"
        genome_vectors: dict[str, torch.Tensor] = {}
        for gid in split_ids:
            npy_path = mlp_dir / f"{gid}.npy"
            vec = numpy.load(npy_path)
            genome_vectors[gid] = torch.from_numpy(vec).float()

        # Construir listas de muestras
        self._vectors: list[torch.Tensor] = []
        self._antibiotic_idxs: list[int] = []
        self._labels: list[float] = []

        for _, row in labels.iterrows():
            self._vectors.append(genome_vectors[row["genome_id"]])
            self._antibiotic_idxs.append(self._antibiotic_to_idx[row["antibiotic"]])
            self._labels.append(
                1.0 if row["resistant_phenotype"] == "Resistant" else 0.0
            )

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Devuelve (genome_vector, antibiotic_idx, label) como tensores CPU."""
        return (
            self._vectors[idx],
            torch.tensor(self._antibiotic_idxs[idx], dtype=torch.long),
            torch.tensor(self._labels[idx], dtype=torch.float32),
        )

    @staticmethod
    def load_pos_weight(data_dir: str | Path) -> float:
        """
        Lee pos_weight desde train_stats.json generado por el pipeline.

        Retorna n_susceptible / n_resistant para usar en BCEWithLogitsLoss.
        """
        stats_path = Path(data_dir) / "train_stats.json"
        stats = json.loads(stats_path.read_text())
        return stats["pos_weight"]
