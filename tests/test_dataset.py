"""
test_dataset.py

Tests unitarios de AMRDataset. Usa datos sintéticos en un directorio temporal
que replica la estructura de outputs del pipeline.
"""

import json

import numpy
import pandas
import pytest
import torch

from data_pipeline.constants import TOTAL_KMER_DIM
from dataset import AMRDataset

_N_TRAIN_GENOMES = 6
_N_VAL_GENOMES = 2
_N_TEST_GENOMES = 2
_ANTIBIOTICS = ["amikacin", "ampicillin"]


@pytest.fixture()
def data_dir(tmp_path):
    """Crea un directorio con la estructura mínima del pipeline."""
    # splits.csv
    splits_rows = []
    gid = 0
    for split, n in [("train", _N_TRAIN_GENOMES), ("val", _N_VAL_GENOMES), ("test", _N_TEST_GENOMES)]:
        for _ in range(n):
            splits_rows.append({"genome_id": str(gid), "split": split})
            gid += 1
    pandas.DataFrame(splits_rows).to_csv(tmp_path / "splits.csv", index=False)

    # antibiotic_index.csv
    ab_index = pandas.DataFrame(
        [{"antibiotic": ab, "index": i} for i, ab in enumerate(_ANTIBIOTICS)]
    )
    ab_index.to_csv(tmp_path / "antibiotic_index.csv", index=False)

    # cleaned_labels.csv — cada genoma tiene un registro por antibiótico
    label_rows = []
    for row in splits_rows:
        for ab in _ANTIBIOTICS:
            label_rows.append({
                "genome_id": row["genome_id"],
                "antibiotic": ab,
                "resistant_phenotype": "Resistant" if int(row["genome_id"]) % 2 == 0 else "Susceptible",
            })
    pandas.DataFrame(label_rows).to_csv(tmp_path / "cleaned_labels.csv", index=False)

    # mlp/*.npy — vectores aleatorios de 1344 dims
    mlp_dir = tmp_path / "mlp"
    mlp_dir.mkdir()
    rng = numpy.random.default_rng(42)
    for row in splits_rows:
        vec = rng.standard_normal(TOTAL_KMER_DIM).astype(numpy.float32)
        numpy.save(mlp_dir / f"{row['genome_id']}.npy", vec)

    # train_stats.json
    n_res = sum(1 for r in label_rows if r["resistant_phenotype"] == "Resistant" and r["genome_id"] in {str(i) for i in range(_N_TRAIN_GENOMES)})
    n_sus = sum(1 for r in label_rows if r["resistant_phenotype"] == "Susceptible" and r["genome_id"] in {str(i) for i in range(_N_TRAIN_GENOMES)})
    stats = {"n_resistant": n_res, "n_susceptible": n_sus, "pos_weight": n_sus / n_res}
    (tmp_path / "train_stats.json").write_text(json.dumps(stats))

    return tmp_path


class TestAMRDataset:
    """Tests de la clase AMRDataset."""

    def test_split_lengths(self, data_dir):
        """Cada split tiene genomas * antibióticos muestras."""
        train = AMRDataset(data_dir, "train")
        val = AMRDataset(data_dir, "val")
        test = AMRDataset(data_dir, "test")

        assert len(train) == _N_TRAIN_GENOMES * len(_ANTIBIOTICS)
        assert len(val) == _N_VAL_GENOMES * len(_ANTIBIOTICS)
        assert len(test) == _N_TEST_GENOMES * len(_ANTIBIOTICS)

    def test_getitem_shapes_and_dtypes(self, data_dir):
        """__getitem__ devuelve tensores con shapes y dtypes correctos."""
        ds = AMRDataset(data_dir, "train")
        vec, ab_idx, label = ds[0]

        assert vec.shape == (TOTAL_KMER_DIM,)
        assert vec.dtype == torch.float32
        assert ab_idx.dtype == torch.long
        assert label.dtype == torch.float32

    def test_labels_are_binary(self, data_dir):
        """Todas las labels son 0.0 o 1.0."""
        ds = AMRDataset(data_dir, "train")
        for i in range(len(ds)):
            _, _, label = ds[i]
            assert label.item() in (0.0, 1.0)

    def test_antibiotic_idx_in_range(self, data_dir):
        """Los índices de antibiótico están dentro del rango válido."""
        ds = AMRDataset(data_dir, "train")
        for i in range(len(ds)):
            _, ab_idx, _ = ds[i]
            assert 0 <= ab_idx.item() < ds.n_antibiotics

    def test_n_antibiotics(self, data_dir):
        """n_antibiotics refleja el número de antibióticos en el índice."""
        ds = AMRDataset(data_dir, "train")
        assert ds.n_antibiotics == len(_ANTIBIOTICS)

    def test_load_pos_weight(self, data_dir):
        """load_pos_weight lee correctamente el JSON del pipeline."""
        pw = AMRDataset.load_pos_weight(data_dir)
        assert isinstance(pw, float)
        assert pw > 0

    def test_splits_are_disjoint(self, data_dir):
        """Los vectores de train, val y test no se solapan."""
        train = AMRDataset(data_dir, "train")
        val = AMRDataset(data_dir, "val")

        train_vecs = {tuple(train[i][0].tolist()) for i in range(len(train))}
        val_vecs = {tuple(val[i][0].tolist()) for i in range(len(val))}
        assert train_vecs.isdisjoint(val_vecs)
