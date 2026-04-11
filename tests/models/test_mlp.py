"""
test_mlp.py

Tests unitarios del modelo AMRMLP.
"""

import torch
import pandas
import pytest

from data_pipeline.constants import TOTAL_KMER_DIM
from models.mlp.model import AMRMLP, ANTIBIOTIC_EMBEDDING_DIM


_N_ANTIBIOTICS = 10


@pytest.fixture()
def model():
    """Modelo MLP con 10 antibióticos para tests."""
    return AMRMLP(n_antibiotics=_N_ANTIBIOTICS)


class TestAMRMLP:
    """Tests de la arquitectura AMRMLP."""

    def test_output_shape(self, model):
        """La salida tiene shape [batch, 1] para cualquier tamaño de batch."""
        for batch_size in [1, 4, 32]:
            genome = torch.randn(batch_size, TOTAL_KMER_DIM)
            ab_idx = torch.zeros(batch_size, dtype=torch.long)
            logits = model(genome, ab_idx)
            assert logits.shape == (batch_size, 1)

    def test_output_is_unbounded_logits(self, model):
        """La salida son logits sin sigmoid (pueden ser < 0 o > 1)."""
        # Usar input extremo para forzar valores fuera de [0, 1]
        torch.manual_seed(42)
        genome = torch.randn(100, TOTAL_KMER_DIM) * 10
        ab_idx = torch.zeros(100, dtype=torch.long)
        logits = model(genome, ab_idx)
        # Con 100 muestras e input amplificado, es casi seguro que algún
        # logit cae fuera de [0, 1]
        assert logits.min().item() < 0 or logits.max().item() > 1

    def test_dropout_inactive_in_eval(self, model):
        """En modo eval, el Dropout está inactivo y la salida es determinista."""
        model.eval()
        genome = torch.randn(4, TOTAL_KMER_DIM)
        ab_idx = torch.tensor([0, 1, 2, 3])
        out1 = model(genome, ab_idx)
        out2 = model(genome, ab_idx)
        assert torch.equal(out1, out2)

    def test_dropout_active_in_train(self, model):
        """En modo train, el Dropout introduce variación entre pasadas."""
        model.train()
        genome = torch.randn(32, TOTAL_KMER_DIM)
        ab_idx = torch.zeros(32, dtype=torch.long)
        out1 = model(genome, ab_idx)
        out2 = model(genome, ab_idx)
        # Con 32 muestras y dropout 0.3, es virtualmente imposible que
        # ambas pasadas produzcan el mismo resultado
        assert not torch.equal(out1, out2)

    def test_from_antibiotic_index(self, tmp_path):
        """Factory method crea el modelo con el conteo correcto de antibióticos."""
        df = pandas.DataFrame({"antibiotic": ["a", "b", "c"], "index": [0, 1, 2]})
        csv_path = tmp_path / "antibiotic_index.csv"
        df.to_csv(csv_path, index=False)

        model = AMRMLP.from_antibiotic_index(str(csv_path))
        assert model.antibiotic_embedding.num_embeddings == 3

    def test_embedding_dim(self, model):
        """El embedding de antibiótico tiene la dimensión esperada."""
        assert model.antibiotic_embedding.embedding_dim == ANTIBIOTIC_EMBEDDING_DIM
