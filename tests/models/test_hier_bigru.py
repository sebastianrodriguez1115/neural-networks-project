"""
test_hier_bigru.py

Tests unitarios para el modelo AMRHierBiGRU y su mecanismo de atención.
"""

import pandas
import pytest
import torch

from models.hier_bigru.model import AMRHierBiGRU, ATTENTION_DIM, GRU_OUTPUT_DIM
from models.bigru.model import ANTIBIOTIC_EMBEDDING_DIM
from data_pipeline.constants import HIER_KMER_DIM, HIER_N_SEGMENTS

_N_ANTIBIOTICS = 10


@pytest.fixture()
def model():
    return AMRHierBiGRU(n_antibiotics=_N_ANTIBIOTICS)


class TestAMRHierBiGRU:
    """Tests para el modelo AMRHierBiGRU."""

    def test_output_shape(self, model):
        """El modelo debe devolver [batch, 1] para distintos tamaños de batch."""
        for batch_size in [1, 4, 32]:
            genome = torch.randn(batch_size, HIER_N_SEGMENTS, HIER_KMER_DIM)
            ab_idx = torch.randint(0, _N_ANTIBIOTICS, (batch_size,))

            logits = model(genome, ab_idx)
            assert logits.shape == (batch_size, 1)

    def test_output_is_unbounded_logits(self, model):
        """El modelo debe entregar logits sin acotar (sin sigmoid final)."""
        with torch.no_grad():
            model.classifier[-1].bias.fill_(5.0)

        genome = torch.randn(4, HIER_N_SEGMENTS, HIER_KMER_DIM)
        ab_idx = torch.randint(0, _N_ANTIBIOTICS, (4,))

        logits = model(genome, ab_idx)
        assert torch.all(logits > 1.0)

    def test_dropout_inactive_in_eval(self, model):
        """La salida debe ser determinista en modo eval()."""
        model.eval()
        genome = torch.randn(1, HIER_N_SEGMENTS, HIER_KMER_DIM)
        ab_idx = torch.tensor([0])

        with torch.no_grad():
            out1 = model(genome, ab_idx)
            out2 = model(genome, ab_idx)

        assert torch.allclose(out1, out2)

    def test_dropout_active_in_train(self, model):
        """La salida debe variar en modo train() debido al dropout."""
        model.train()
        batch_size = 32
        genome = torch.randn(batch_size, HIER_N_SEGMENTS, HIER_KMER_DIM)
        ab_idx = torch.randint(0, _N_ANTIBIOTICS, (batch_size,))

        out1 = model(genome, ab_idx)
        out2 = model(genome, ab_idx)
        assert not torch.allclose(out1, out2)

    def test_attention_weights_shape(self, model):
        """Los pesos de atención deben tener shape [batch, HIER_N_SEGMENTS]."""
        batch_size = 4
        genome = torch.randn(batch_size, HIER_N_SEGMENTS, HIER_KMER_DIM)
        ab_idx = torch.randint(0, _N_ANTIBIOTICS, (batch_size,))

        model(genome, ab_idx)
        assert model._attention_weights is not None
        assert model._attention_weights.shape == (batch_size, HIER_N_SEGMENTS)

    def test_attention_weights_sum_to_one(self, model):
        """Los pesos de atención deben sumar 1 por muestra (softmax)."""
        genome = torch.randn(4, HIER_N_SEGMENTS, HIER_KMER_DIM)
        ab_idx = torch.randint(0, _N_ANTIBIOTICS, (4,))

        model(genome, ab_idx)
        sums = model._attention_weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_embedding_dim(self, model):
        """La capa de embedding debe tener la dimensión configurada."""
        assert model.antibiotic_embedding.embedding_dim == ANTIBIOTIC_EMBEDDING_DIM
        assert model.antibiotic_embedding.num_embeddings == _N_ANTIBIOTICS

    def test_from_antibiotic_index(self, tmp_path):
        """El factory method debe instanciar el modelo correctamente desde un CSV."""
        csv_path = tmp_path / "antibiotic_index.csv"
        pandas.DataFrame({
            "antibiotic": ["ab1", "ab2", "ab3"],
            "index": [0, 1, 2],
        }).to_csv(csv_path, index=False)

        loaded = AMRHierBiGRU.from_antibiotic_index(str(csv_path))
        assert isinstance(loaded, AMRHierBiGRU)
        assert loaded.antibiotic_embedding.num_embeddings == 3
