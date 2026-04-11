"""
test_bigru.py

Tests unitarios para el modelo AMRBiGRU y el mecanismo de atención.
"""

import pandas
import pytest
import torch

from models.bigru.model import AMRBiGRU, ANTIBIOTIC_EMBEDDING_DIM

_N_ANTIBIOTICS = 10
_SEQ_LEN = 1024
_INPUT_DIM = 3


@pytest.fixture()
def model():
    return AMRBiGRU(n_antibiotics=_N_ANTIBIOTICS)


class TestAMRBiGRU:
    """Tests para el modelo AMRBiGRU."""

    def test_output_shape(self, model):
        """El modelo debe devolver [batch, 1] para distintos tamaños de batch."""
        for batch_size in [1, 4, 32]:
            genome = torch.randn(batch_size, _SEQ_LEN, _INPUT_DIM)
            ab_idx = torch.randint(0, _N_ANTIBIOTICS, (batch_size,))
            
            logits = model(genome, ab_idx)
            assert logits.shape == (batch_size, 1)

    def test_output_is_unbounded_logits(self, model):
        """Verifica que el modelo entrega logits sin acotar (comportamiento)."""
        # La GRU tiene tanh interna (~[-1, 1]), por lo que forzar logits fuera de [0, 1]
        # vía inputs grandes no es confiable. Forzamos el bias de la capa final [Haykin, Cap. 4.1].
        with torch.no_grad():
            model.classifier[-1].bias.fill_(5.0)

        genome = torch.randn(4, _SEQ_LEN, _INPUT_DIM)
        ab_idx = torch.randint(0, _N_ANTIBIOTICS, (4,))
        
        logits = model(genome, ab_idx)
        # Con bias=5, los logits deberían ser > 1.0
        assert torch.all(logits > 1.0)

    def test_dropout_inactive_in_eval(self, model):
        """La salida debe ser determinista en modo eval()."""
        model.eval()
        genome = torch.randn(1, _SEQ_LEN, _INPUT_DIM)
        ab_idx = torch.tensor([0])
        
        with torch.no_grad():
            out1 = model(genome, ab_idx)
            out2 = model(genome, ab_idx)
            
        assert torch.allclose(out1, out2)

    def test_dropout_active_in_train(self, model):
        """La salida debe variar en modo train() debido al dropout."""
        model.train()
        batch_size = 32
        genome = torch.randn(batch_size, _SEQ_LEN, _INPUT_DIM)
        ab_idx = torch.randint(0, _N_ANTIBIOTICS, (batch_size,))
        
        # Con batch_size=32 la probabilidad de coincidencia exacta es despreciable
        out1 = model(genome, ab_idx)
        out2 = model(genome, ab_idx)
        assert not torch.allclose(out1, out2)

    def test_attention_weights_shape(self, model):
        """Los pesos de atención deben guardarse con shape [batch, 1024]."""
        batch_size = 4
        genome = torch.randn(batch_size, _SEQ_LEN, _INPUT_DIM)
        ab_idx = torch.randint(0, _N_ANTIBIOTICS, (batch_size,))
        
        model(genome, ab_idx)
        assert model._attention_weights is not None
        assert model._attention_weights.shape == (batch_size, _SEQ_LEN)

    def test_attention_weights_sum_to_one(self, model):
        """Los pesos de atención deben sumar 1 por cada muestra (softmax)."""
        genome = torch.randn(4, _SEQ_LEN, _INPUT_DIM)
        ab_idx = torch.randint(0, _N_ANTIBIOTICS, (4,))
        
        model(genome, ab_idx)
        sums = model._attention_weights.sum(dim=1)
        # Mayor tolerancia (1e-5) para errores de redondeo en float32 con seq_len=1024
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
            "index": [0, 1, 2]
        }).to_csv(csv_path, index=False)
        
        model = AMRBiGRU.from_antibiotic_index(str(csv_path))
        assert isinstance(model, AMRBiGRU)
        assert model.antibiotic_embedding.num_embeddings == 3
