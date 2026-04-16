"""
test_datasets.py

Tests unitarios para las clases Dataset de cada modelo. Usa datos sintéticos en un directorio temporal
que replica la estructura de outputs del pipeline.
"""

import json

import numpy
import pandas
import pytest
import torch

from data_pipeline.constants import HIER_KMER_DIM, HIER_N_SEGMENTS, TOTAL_KMER_DIM, TOKEN_PAD_ID
from models.mlp.dataset import MLPDataset
from models.bigru.dataset import BiGRUDataset
from models.multi_bigru.dataset import MultiBiGRUDataset
from models.token_bigru.dataset import TokenBiGRUDataset
from models.hier_bigru.dataset import HierBiGRUDataset
from models.hier_set.dataset import HierSetDataset

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

    # bigru/*.npy — matrices aleatorias de (1024, 3)
    bigru_dir = tmp_path / "bigru"
    bigru_dir.mkdir()
    for row in splits_rows:
        mat = rng.standard_normal((1024, 3)).astype(numpy.float32)
        numpy.save(bigru_dir / f"{row['genome_id']}.npy", mat)

    # token_bigru/*.npy — IDs aleatorios de (4096,) en [0, 256]
    token_dir = tmp_path / "token_bigru"
    token_dir.mkdir()
    for row in splits_rows:
        tokens = rng.integers(0, 257, 4096).astype(numpy.int64)
        numpy.save(token_dir / f"{row['genome_id']}.npy", tokens)

    # hier_bigru/*.npy — matrices (64, 256) de frecuencias relativas sintéticas
    hier_dir = tmp_path / "hier_bigru"
    hier_dir.mkdir()
    for row in splits_rows:
        mat = rng.dirichlet(numpy.ones(HIER_KMER_DIM), size=HIER_N_SEGMENTS).astype(numpy.float32)
        numpy.save(hier_dir / f"{row['genome_id']}.npy", mat)

    # train_stats.json
    n_res = sum(1 for r in label_rows if r["resistant_phenotype"] == "Resistant" and r["genome_id"] in {str(i) for i in range(_N_TRAIN_GENOMES)})
    n_sus = sum(1 for r in label_rows if r["resistant_phenotype"] == "Susceptible" and r["genome_id"] in {str(i) for i in range(_N_TRAIN_GENOMES)})
    stats = {"n_resistant": n_res, "n_susceptible": n_sus, "pos_weight": n_sus / n_res}
    (tmp_path / "train_stats.json").write_text(json.dumps(stats))

    return tmp_path


class TestDatasets:
    """Tests para las clases Dataset modularizadas."""

    def test_mlp_dataset(self, data_dir):
        """Verifica que MLPDataset carga correctamente vectores 1D."""
        ds = MLPDataset(data_dir, "train")
        assert len(ds) == _N_TRAIN_GENOMES * len(_ANTIBIOTICS)
        vec, ab_idx, label = ds[0]
        assert vec.shape == (TOTAL_KMER_DIM,)
        assert vec.dtype == torch.float32
        assert ab_idx.dtype == torch.long
        assert label.dtype == torch.float32

    def test_bigru_dataset(self, data_dir):
        """Verifica que BiGRUDataset carga correctamente matrices 2D."""
        ds = BiGRUDataset(data_dir, "train")
        assert len(ds) == _N_TRAIN_GENOMES * len(_ANTIBIOTICS)
        mat, ab_idx, label = ds[0]
        assert mat.shape == (1024, 3)
        assert mat.dtype == torch.float32

    def test_multi_bigru_dataset(self, data_dir):
        """Verifica que MultiBiGRUDataset segmenta correctamente en 3 flujos."""
        ds = MultiBiGRUDataset(data_dir, "train")
        genome, ab_idx, label = ds[0]
        assert isinstance(genome, tuple)
        assert len(genome) == 3
        k3, k4, k5 = genome
        assert k3.shape == (64, 1)
        assert k4.shape == (256, 1)
        assert k5.shape == (1024, 1)
        assert k3.dtype == torch.float32

    def test_token_bigru_dataset(self, data_dir):
        """Verifica que TokenBiGRUDataset carga tokens como long en rango válido."""
        ds = TokenBiGRUDataset(data_dir, "train")
        tokens, ab_idx, label = ds[0]
        assert tokens.shape == (4096,)
        assert tokens.dtype == torch.long
        assert tokens.min().item() >= 0
        assert tokens.max().item() <= TOKEN_PAD_ID  # 256 (incluye padding)

    def test_hier_bigru_dataset(self, data_dir):
        """Verifica que HierBiGRUDataset carga matrices [64, 256] float32."""
        ds = HierBiGRUDataset(data_dir, "train")
        assert len(ds) == _N_TRAIN_GENOMES * len(_ANTIBIOTICS)
        genome, ab_idx, label = ds[0]
        assert genome.shape == (HIER_N_SEGMENTS, HIER_KMER_DIM)
        assert genome.dtype == torch.float32
        assert ab_idx.dtype == torch.long
        assert label.dtype == torch.float32

    def test_hier_set_dataset(self, data_dir):
        """Verifica que HierSetDataset carga los mismos datos que HierBiGRUDataset."""
        ds = HierSetDataset(data_dir, "train")
        assert len(ds) == _N_TRAIN_GENOMES * len(_ANTIBIOTICS)
        genome, ab_idx, label = ds[0]
        assert genome.shape == (HIER_N_SEGMENTS, HIER_KMER_DIM)
        assert genome.dtype == torch.float32
        assert ab_idx.dtype == torch.long
        assert label.dtype == torch.float32

    def test_hier_bigru_dataset_rejects_wrong_shape(self, data_dir):
        """Verifica que se lanza ValueError con shape incorrecto."""
        import numpy as np
        bad_npy = data_dir / "hier_bigru" / "0.npy"
        np.save(bad_npy, np.zeros((32, 256), dtype=numpy.float32))
        with pytest.raises(ValueError, match="Shape inesperado"):
            HierBiGRUDataset(data_dir, "train")

    def test_common_metadata(self, data_dir):
        """Verifica que la lógica común de BaseAMRDataset funciona en las subclases."""
        ds = MLPDataset(data_dir, "train")
        assert ds.n_antibiotics == len(_ANTIBIOTICS)
        assert ds.load_pos_weight(data_dir) > 0

        # Verificar etiquetas binarias
        for i in range(len(ds)):
            _, _, label = ds[i]
            assert label.item() in (0.0, 1.0)

    def test_splits_are_disjoint(self, data_dir):
        """Los vectores de train y val no se solapan."""
        train = MLPDataset(data_dir, "train")
        val = MLPDataset(data_dir, "val")

        train_vecs = {tuple(train[i][0].tolist()) for i in range(len(train))}
        val_vecs = {tuple(val[i][0].tolist()) for i in range(len(val))}
        assert train_vecs.isdisjoint(val_vecs)
