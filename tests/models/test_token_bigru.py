"""
test_token_bigru.py

Tests unitarios para el modelo AMRTokenBiGRU.
"""

import pandas
import pytest
import torch
import torch.nn as nn

from models.token_bigru.model import AMRTokenBiGRU
from data_pipeline.constants import TOKEN_VOCAB_SIZE, TOKEN_PAD_ID

_N_ANTIBIOTICS = 10
_SEQ_LEN = 4096


@pytest.fixture()
def model():
    return AMRTokenBiGRU(n_antibiotics=_N_ANTIBIOTICS)


@pytest.fixture()
def sample_input():
    batch_size = 4
    # Tokens en [0, 256], 256 es padding
    tokens = torch.randint(0, TOKEN_VOCAB_SIZE + 1, (batch_size, _SEQ_LEN))
    ab_idx = torch.randint(0, _N_ANTIBIOTICS, (batch_size,))
    return tokens, ab_idx


class TestAMRTokenBiGRU:
    """Tests para AMRTokenBiGRU."""

    def test_output_shape(self, model, sample_input):
        """El modelo produce un logit por muestra [batch, 1]."""
        tokens, ab_idx = sample_input
        output = model(tokens, ab_idx)
        assert output.shape == (tokens.shape[0], 1)
        assert output.dtype == torch.float32

    def test_output_is_unbounded_logits(self, model, sample_input):
        """La salida son logits (pueden ser > 1 o < 0), no probabilidades."""
        tokens, ab_idx = sample_input
        output = model(tokens, ab_idx)
        # Con pesos aleatorios, es altamente probable tener algún valor fuera de [0, 1]
        # o al menos no todos en [0, 1] si no hay sigmoid.
        # Pero para ser rigurosos, verificamos que no hay Sigmoid al final del classifer.
        assert not isinstance(model.classifier[-1], nn.Sigmoid)

    def test_dropout_inactive_in_eval(self, model, sample_input):
        """En modo eval(), la salida es determinista."""
        tokens, ab_idx = sample_input
        model.eval()
        with torch.no_grad():
            out1 = model(tokens, ab_idx)
            out2 = model(tokens, ab_idx)
        assert torch.allclose(out1, out2)

    def test_attention_weights_shape(self, model, sample_input):
        """Produce pesos de atención de forma [batch, seq_len]."""
        tokens, ab_idx = sample_input
        model(tokens, ab_idx)
        weights = model._attention_weights
        assert weights is not None
        assert weights.shape == (tokens.shape[0], _SEQ_LEN)

    def test_attention_weights_sum_to_one(self, model, sample_input):
        """Los pesos de atención en cada muestra suman 1.0 (softmax)."""
        tokens, ab_idx = sample_input
        model(tokens, ab_idx)
        weights = model._attention_weights
        sums = weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums))

    def test_embedding_padding(self, model):
        """El token de padding se mapea al vector cero."""
        pad_tensor = torch.tensor([[TOKEN_PAD_ID]])
        embedded = model.kmer_embedding(pad_tensor)
        assert torch.allclose(embedded, torch.zeros_like(embedded))

    def test_from_antibiotic_index(self, tmp_path):
        """Factory method lee correctamente el número de antibióticos."""
        csv_path = tmp_path / "antibiotics.csv"
        pandas.DataFrame({
            "antibiotic": ["a", "b", "c"],
            "index": [0, 1, 2]
        }).to_csv(csv_path, index=False)

        model = AMRTokenBiGRU.from_antibiotic_index(str(csv_path))
        assert model.antibiotic_embedding.num_embeddings == 3

    def test_dropout_active_in_train(self, model, sample_input):
        """En modo train(), el dropout introduce variabilidad (salidas distintas con batch grande)."""
        tokens, ab_idx = sample_input
        # Usar batch grande para reducir probabilidad de coincidencia exacta
        tokens_large = torch.randint(0, TOKEN_VOCAB_SIZE, (32, _SEQ_LEN))
        ab_idx_large = torch.randint(0, _N_ANTIBIOTICS, (32,))
        model.train()
        out1 = model(tokens_large, ab_idx_large)
        out2 = model(tokens_large, ab_idx_large)
        assert not torch.allclose(out1, out2)

    def test_input_dtype(self, model):
        """El modelo falla si se le pasan floats en lugar de longs para tokens."""
        tokens = torch.randn(4, _SEQ_LEN)
        ab_idx = torch.randint(0, _N_ANTIBIOTICS, (4,))
        with pytest.raises(RuntimeError):
            model(tokens, ab_idx)
