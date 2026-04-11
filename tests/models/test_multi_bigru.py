"""
test_multi_bigru.py

Tests unitarios para el modelo AMRMultiBiGRU (arquitectura multi-stream).
"""

import pandas
import pytest
import torch

from models.multi_bigru.model import AMRMultiBiGRU
from data_pipeline.constants import ANTIBIOTIC_EMBEDDING_DIM

_N_ANTIBIOTICS = 10


@pytest.fixture()
def model():
    return AMRMultiBiGRU(n_antibiotics=_N_ANTIBIOTICS)


@pytest.fixture()
def sample_input():
    """Genera un input de ejemplo con las tres secuencias de k-meros."""
    batch_size = 4
    k3 = torch.randn(batch_size, 64, 1)
    k4 = torch.randn(batch_size, 256, 1)
    k5 = torch.randn(batch_size, 1024, 1)
    ab_idx = torch.randint(0, _N_ANTIBIOTICS, (batch_size,))
    return (k3, k4, k5), ab_idx


class TestAMRMultiBiGRU:
    """Tests para el modelo AMRMultiBiGRU."""

    def test_output_shape(self, model):
        """El modelo debe devolver [batch, 1] para distintos tamaños de batch."""
        for batch_size in [1, 4, 32]:
            k3 = torch.randn(batch_size, 64, 1)
            k4 = torch.randn(batch_size, 256, 1)
            k5 = torch.randn(batch_size, 1024, 1)
            ab_idx = torch.randint(0, _N_ANTIBIOTICS, (batch_size,))
            
            logits = model((k3, k4, k5), ab_idx)
            assert logits.shape == (batch_size, 1)

    def test_output_is_unbounded_logits(self, model, sample_input):
        """Verifica que el modelo entrega logits sin acotar (comportamiento)."""
        with torch.no_grad():
            model.classifier[-1].bias.fill_(5.0)

        genome, ab_idx = sample_input
        logits = model(genome, ab_idx)
        assert torch.all(logits > 1.0)

    def test_dropout_inactive_in_eval(self, model, sample_input):
        """La salida debe ser determinista en modo eval()."""
        model.eval()
        genome, ab_idx = sample_input
        
        with torch.no_grad():
            out1 = model(genome, ab_idx)
            out2 = model(genome, ab_idx)
            
        assert torch.allclose(out1, out2)

    def test_dropout_active_in_train(self, model):
        """La salida debe variar en modo train() debido al dropout."""
        model.train()
        batch_size = 32
        k3 = torch.randn(batch_size, 64, 1)
        k4 = torch.randn(batch_size, 256, 1)
        k5 = torch.randn(batch_size, 1024, 1)
        ab_idx = torch.randint(0, _N_ANTIBIOTICS, (batch_size,))
        
        out1 = model((k3, k4, k5), ab_idx)
        out2 = model((k3, k4, k5), ab_idx)
        assert not torch.allclose(out1, out2)

    def test_attention_weights_structure(self, model, sample_input):
        """_attention_weights debe ser un dict con las formas correctas."""
        genome, ab_idx = sample_input
        model(genome, ab_idx)
        
        weights = model._attention_weights
        assert isinstance(weights, dict)
        assert weights["k3"].shape == (4, 64)
        assert weights["k4"].shape == (4, 256)
        assert weights["k5"].shape == (4, 1024)

    def test_attention_weights_sum_to_one(self, model, sample_input):
        """Cada stream de atención debe sumar 1.0 independientemente."""
        genome, ab_idx = sample_input
        model(genome, ab_idx)
        
        for k in ["k3", "k4", "k5"]:
            sums = model._attention_weights[k].sum(dim=1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_streams_are_independent(self, model, sample_input):
        """Modificar k3 no debe afectar los pesos de atención de k4 y k5."""
        genome, ab_idx = sample_input
        
        # Pasada 1
        model(genome, ab_idx)
        w4_1 = model._attention_weights["k4"].clone()
        w5_1 = model._attention_weights["k5"].clone()
        
        # Pasada 2 con k3 modificado
        k3_new = torch.randn_like(genome[0])
        model((k3_new, genome[1], genome[2]), ab_idx)
        w4_2 = model._attention_weights["k4"]
        w5_2 = model._attention_weights["k5"]
        
        assert torch.allclose(w4_1, w4_2)
        assert torch.allclose(w5_1, w5_2)

    def test_embedding_dim(self, model):
        """La capa de embedding debe tener la dimensión configurada."""
        assert model.antibiotic_embedding.embedding_dim == ANTIBIOTIC_EMBEDDING_DIM
        assert model.antibiotic_embedding.num_embeddings == _N_ANTIBIOTICS

    def test_from_antibiotic_index(self, tmp_path):
        """El factory method debe instanciar el modelo correctamente."""
        csv_path = tmp_path / "antibiotic_index.csv"
        pandas.DataFrame({
            "antibiotic": ["ab1", "ab2"],
            "index": [0, 1]
        }).to_csv(csv_path, index=False)
        
        model = AMRMultiBiGRU.from_antibiotic_index(str(csv_path))
        assert isinstance(model, AMRMultiBiGRU)
        assert model.antibiotic_embedding.num_embeddings == 2
