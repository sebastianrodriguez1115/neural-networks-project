"""
test_hier_set_v2.py

Tests unitarios para AMRHierSetV2. Verifican:
    - Forward pass con histogramas multi-escala [B, S, 1344]
    - Shape de _attention_weights [B, H, S] (multi-head)
    - Permutation-invariance sobre segmentos
    - Atención varía según el antibiótico
    - Dataset carga desde hier_set_v2/ con shape (S, 1344)
"""

import numpy
import pandas
import pytest
import torch
import torch.nn as nn

from data_pipeline.constants import (
    ANTIBIOTIC_EMBEDDING_DIM,
    HIER_KMER_DIM_MULTI,
    HIER_N_SEGMENTS,
)
from models.hier_set_v2.dataset import HierSetV2Dataset
from models.hier_set_v2.model import D_MODEL, N_HEADS, AMRHierSetV2

_N_ANTIBIOTICS = 10


@pytest.fixture()
def model():
    return AMRHierSetV2(n_antibiotics=_N_ANTIBIOTICS)


class TestAMRHierSetV2:

    def test_output_shape(self, model):
        for batch_size in [1, 4, 32]:
            genome = torch.randn(batch_size, HIER_N_SEGMENTS, HIER_KMER_DIM_MULTI)
            ab_idx = torch.randint(0, _N_ANTIBIOTICS, (batch_size,))
            assert model(genome, ab_idx).shape == (batch_size, 1)

    def test_attention_weights_shape(self, model):
        """Multi-head: los pesos tienen shape [batch, H, S]."""
        genome = torch.randn(4, HIER_N_SEGMENTS, HIER_KMER_DIM_MULTI)
        ab_idx = torch.randint(0, _N_ANTIBIOTICS, (4,))
        model(genome, ab_idx)
        assert model._attention_weights is not None
        assert model._attention_weights.shape == (4, N_HEADS, HIER_N_SEGMENTS)

    def test_attention_weights_sum_to_one_per_head(self, model):
        """Softmax aplicado por cabeza sobre el eje de segmentos."""
        genome = torch.randn(4, HIER_N_SEGMENTS, HIER_KMER_DIM_MULTI)
        ab_idx = torch.randint(0, _N_ANTIBIOTICS, (4,))
        model(genome, ab_idx)
        sums = model._attention_weights.sum(dim=2)  # [B, H]
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_dropout_inactive_in_eval(self, model):
        model.eval()
        genome = torch.randn(1, HIER_N_SEGMENTS, HIER_KMER_DIM_MULTI)
        ab_idx = torch.tensor([0])
        with torch.no_grad():
            assert torch.allclose(model(genome, ab_idx), model(genome, ab_idx))

    def test_dropout_active_in_train(self, model):
        model.train()
        genome = torch.randn(32, HIER_N_SEGMENTS, HIER_KMER_DIM_MULTI)
        ab_idx = torch.randint(0, _N_ANTIBIOTICS, (32,))
        assert not torch.allclose(model(genome, ab_idx), model(genome, ab_idx))

    def test_permutation_invariance(self, model):
        """El modelo sigue siendo permutation-invariant sobre segmentos."""
        model.eval()
        genome = torch.randn(2, HIER_N_SEGMENTS, HIER_KMER_DIM_MULTI)
        ab_idx = torch.zeros(2, dtype=torch.long)
        perm = torch.randperm(HIER_N_SEGMENTS)
        genome_shuffled = genome[:, perm, :]
        with torch.no_grad():
            out1 = model(genome, ab_idx)
            out2 = model(genome_shuffled, ab_idx)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_no_sequential_modules(self, model):
        for name, module in model.named_modules():
            assert not isinstance(module, (nn.GRU, nn.LSTM, nn.RNN)), (
                f"Módulo secuencial inesperado: {name} ({type(module).__name__})"
            )

    def test_embedding_dim(self, model):
        assert model.antibiotic_embedding.embedding_dim == ANTIBIOTIC_EMBEDDING_DIM
        assert model.antibiotic_embedding.num_embeddings == _N_ANTIBIOTICS

    def test_d_model_divisible_by_heads(self):
        assert D_MODEL % N_HEADS == 0

    def test_attention_varies_by_antibiotic(self, model):
        """El antibiótico debe alterar los pesos de atención (cross-attn condicionada)."""
        model.eval()
        with torch.no_grad():
            model.antibiotic_embedding.weight.data[0].fill_(0.0)
            model.antibiotic_embedding.weight.data[1].fill_(1.0)

        genome = torch.randn(1, HIER_N_SEGMENTS, HIER_KMER_DIM_MULTI)
        with torch.no_grad():
            model(genome, torch.tensor([0]))
            attn0 = model._attention_weights.clone()
            model(genome, torch.tensor([1]))
            attn1 = model._attention_weights.clone()
        assert not torch.allclose(attn0, attn1)

    def test_from_antibiotic_index(self, tmp_path):
        csv_path = tmp_path / "antibiotic_index.csv"
        pandas.DataFrame({
            "antibiotic": ["ab1", "ab2", "ab3"],
            "index": [0, 1, 2],
        }).to_csv(csv_path, index=False)
        loaded = AMRHierSetV2.from_antibiotic_index(str(csv_path))
        assert isinstance(loaded, AMRHierSetV2)
        assert loaded.antibiotic_embedding.num_embeddings == 3


# ── HierSetV2Dataset ──────────────────────────────────────────────────────────


def _write_dataset_fixture(tmp_path):
    """Escribe splits, labels, antibiotic_index y .npy mínimos en tmp_path."""
    genome_ids = ["g1", "g2"]
    pandas.DataFrame({
        "genome_id": genome_ids,
        "split": ["train", "val"],
    }).to_csv(tmp_path / "splits.csv", index=False)
    pandas.DataFrame({
        "genome_id": genome_ids,
        "antibiotic": ["amikacin", "amikacin"],
        "resistant_phenotype": ["Resistant", "Susceptible"],
    }).to_csv(tmp_path / "cleaned_labels.csv", index=False)
    pandas.DataFrame({
        "antibiotic": ["amikacin"],
        "index": [0],
    }).to_csv(tmp_path / "antibiotic_index.csv", index=False)
    (tmp_path / "train_stats.json").write_text('{"pos_weight": 1.0}')

    hier_dir = tmp_path / "hier_set_v2"
    hier_dir.mkdir()
    for gid in genome_ids:
        matrix = numpy.random.rand(HIER_N_SEGMENTS, HIER_KMER_DIM_MULTI).astype(numpy.float32)
        numpy.save(hier_dir / f"{gid}.npy", matrix)


def test_dataset_loads_train_split(tmp_path):
    _write_dataset_fixture(tmp_path)
    ds = HierSetV2Dataset(tmp_path, split="train")
    assert len(ds) == 1
    genome_tensor, ab_idx, label = ds[0]
    assert genome_tensor.shape == (HIER_N_SEGMENTS, HIER_KMER_DIM_MULTI)
    assert genome_tensor.dtype == torch.float32
    assert ab_idx.dtype == torch.long
    assert label.item() == 1.0


def test_dataset_raises_on_missing_dir(tmp_path):
    # Fixture sin el directorio hier_set_v2/
    pandas.DataFrame({
        "genome_id": ["g1"],
        "split": ["train"],
    }).to_csv(tmp_path / "splits.csv", index=False)
    pandas.DataFrame({
        "genome_id": ["g1"],
        "antibiotic": ["amikacin"],
        "resistant_phenotype": ["Resistant"],
    }).to_csv(tmp_path / "cleaned_labels.csv", index=False)
    pandas.DataFrame({"antibiotic": ["amikacin"], "index": [0]}).to_csv(
        tmp_path / "antibiotic_index.csv", index=False
    )

    with pytest.raises(FileNotFoundError):
        HierSetV2Dataset(tmp_path, split="train")


def test_dataset_raises_on_bad_shape(tmp_path):
    _write_dataset_fixture(tmp_path)
    # Sobrescribir con shape incorrecto
    bad = numpy.zeros((HIER_N_SEGMENTS, HIER_KMER_DIM_MULTI - 1), dtype=numpy.float32)
    numpy.save(tmp_path / "hier_set_v2" / "g1.npy", bad)

    with pytest.raises(ValueError, match="Shape inesperado"):
        HierSetV2Dataset(tmp_path, split="train")
