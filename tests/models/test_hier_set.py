"""
test_hier_set.py

Tests unitarios para AMRHierSet. La propiedad central que se verifica es la
invarianza a permutaciones de segmentos — propiedad que HierBiGRU no cumple.
"""

import pandas
import pytest
import torch
import torch.nn as nn

from models.hier_set.model import AMRHierSet
from data_pipeline.constants import ANTIBIOTIC_EMBEDDING_DIM, HIER_KMER_DIM, HIER_N_SEGMENTS

_N_ANTIBIOTICS = 10


@pytest.fixture()
def model():
    return AMRHierSet(n_antibiotics=_N_ANTIBIOTICS)


class TestAMRHierSet:

    def test_output_shape(self, model):
        """El modelo debe devolver [batch, 1] para distintos tamaños de batch."""
        for batch_size in [1, 4, 32]:
            genome = torch.randn(batch_size, HIER_N_SEGMENTS, HIER_KMER_DIM)
            ab_idx = torch.randint(0, _N_ANTIBIOTICS, (batch_size,))
            assert model(genome, ab_idx).shape == (batch_size, 1)

    def test_output_is_unbounded_logits(self, model):
        """El modelo debe entregar logits sin sigmoid final."""
        with torch.no_grad():
            model.classifier[-1].bias.fill_(5.0)
        genome = torch.randn(4, HIER_N_SEGMENTS, HIER_KMER_DIM)
        ab_idx = torch.randint(0, _N_ANTIBIOTICS, (4,))
        assert torch.all(model(genome, ab_idx) > 1.0)

    def test_dropout_inactive_in_eval(self, model):
        """La salida debe ser determinista en modo eval()."""
        model.eval()
        genome = torch.randn(1, HIER_N_SEGMENTS, HIER_KMER_DIM)
        ab_idx = torch.tensor([0])
        with torch.no_grad():
            assert torch.allclose(model(genome, ab_idx), model(genome, ab_idx))

    def test_dropout_active_in_train(self, model):
        """La salida debe variar en modo train() debido al dropout."""
        model.train()
        genome = torch.randn(32, HIER_N_SEGMENTS, HIER_KMER_DIM)
        ab_idx = torch.randint(0, _N_ANTIBIOTICS, (32,))
        assert not torch.allclose(model(genome, ab_idx), model(genome, ab_idx))

    def test_attention_weights_shape(self, model):
        """Los pesos de atención deben tener shape [batch, HIER_N_SEGMENTS]."""
        genome = torch.randn(4, HIER_N_SEGMENTS, HIER_KMER_DIM)
        ab_idx = torch.randint(0, _N_ANTIBIOTICS, (4,))
        model(genome, ab_idx)
        assert model._attention_weights is not None
        assert model._attention_weights.shape == (4, HIER_N_SEGMENTS)

    def test_attention_weights_sum_to_one(self, model):
        """Los pesos de atención deben sumar 1 por muestra (softmax)."""
        genome = torch.randn(4, HIER_N_SEGMENTS, HIER_KMER_DIM)
        ab_idx = torch.randint(0, _N_ANTIBIOTICS, (4,))
        model(genome, ab_idx)
        sums = model._attention_weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_permutation_invariance(self, model):
        """
        El modelo debe ser invariante al orden de los 64 segmentos.

        Esta es la propiedad central que diferencia AMRHierSet de AMRHierBiGRU:
        como los segmentos se procesan como un conjunto, reordenarlos no cambia
        el logit. Esto es correcto para ensamblajes draft con orden arbitrario.
        """
        model.eval()
        genome = torch.randn(2, HIER_N_SEGMENTS, HIER_KMER_DIM)
        ab_idx = torch.zeros(2, dtype=torch.long)
        perm = torch.randperm(HIER_N_SEGMENTS)
        genome_shuffled = genome[:, perm, :]
        with torch.no_grad():
            out1 = model(genome, ab_idx)
            out2 = model(genome_shuffled, ab_idx)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_no_sequential_modules(self, model):
        """El encoder no debe contener módulos RNN."""
        for name, module in model.named_modules():
            assert not isinstance(module, (nn.GRU, nn.LSTM, nn.RNN)), (
                f"Módulo secuencial inesperado: {name} ({type(module).__name__})"
            )

    def test_embedding_dim(self, model):
        """La capa de embedding debe tener la dimensión configurada."""
        assert model.antibiotic_embedding.embedding_dim == ANTIBIOTIC_EMBEDDING_DIM
        assert model.antibiotic_embedding.num_embeddings == _N_ANTIBIOTICS

    def test_attention_varies_by_antibiotic(self, model):
        """
        El mismo genoma con dos antibióticos distintos debe producir pesos de
        atención distintos — garantiza que ab_emb está conectado a self.attn.
        Si una refactorización desconecta el antibiótico de la atención, este
        test falla aunque los demás sigan en verde.

        Los embeddings se controlan explícitamente (0 vs 1) para garantizar
        queries distintas, independientemente de la inicialización aleatoria.
        """
        model.eval()
        with torch.no_grad():
            # Embeddings maximalmente distintos: 0 → zeros, 1 → ones
            # Con cross-attention (score = h·q_a), q_a=0 → scores uniformes;
            # q_a≠0 → scores distintos por segmento según h. Basta con que
            # los embeddings difieran para que las queries difieran.
            model.antibiotic_embedding.weight.data[0].fill_(0.0)
            model.antibiotic_embedding.weight.data[1].fill_(1.0)

        genome = torch.randn(1, HIER_N_SEGMENTS, HIER_KMER_DIM)
        ab0 = torch.tensor([0])
        ab1 = torch.tensor([1])
        with torch.no_grad():
            model(genome, ab0)
            attn0 = model._attention_weights.clone()
            model(genome, ab1)
            attn1 = model._attention_weights.clone()
        assert not torch.allclose(attn0, attn1), (
            "La atención debe variar según el antibiótico (conditioned attention)"
        )

    def test_from_antibiotic_index(self, tmp_path):
        """El factory method debe instanciar el modelo correctamente desde un CSV."""
        csv_path = tmp_path / "antibiotic_index.csv"
        pandas.DataFrame({
            "antibiotic": ["ab1", "ab2", "ab3"],
            "index": [0, 1, 2],
        }).to_csv(csv_path, index=False)
        loaded = AMRHierSet.from_antibiotic_index(str(csv_path))
        assert isinstance(loaded, AMRHierSet)
        assert loaded.antibiotic_embedding.num_embeddings == 3
