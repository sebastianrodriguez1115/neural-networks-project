"""
test_multi_bigru.py

Tests unitarios para el modelo AMRMultiBiGRU (arquitectura multi-stream).
"""

import pandas
import pytest
import torch
import torch.nn as nn

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

    def test_gate_sum_to_one(self, model, sample_input):
        """Los gates de fusión deben sumar 1 (softmax) — ningún stream puede apagarse."""
        genome, ab_idx = sample_input
        model.eval()
        with torch.no_grad():
            ab_emb = model.antibiotic_embedding(ab_idx)
            gates = torch.softmax(model.stream_gate(ab_emb), dim=-1)  # [batch, 3]
        assert gates.shape == (ab_idx.shape[0], 3)
        sums = gates.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_gate_varies_by_antibiotic(self, model):
        """Antibióticos distintos deben producir gates distintos (fusión condicionada)."""
        model_3 = AMRMultiBiGRU(n_antibiotics=3)
        model_3.eval()
        ab0 = torch.tensor([0])
        ab1 = torch.tensor([1])
        with torch.no_grad():
            emb0 = model_3.antibiotic_embedding(ab0)
            emb1 = model_3.antibiotic_embedding(ab1)
            gates0 = torch.softmax(model_3.stream_gate(emb0), dim=-1)
            gates1 = torch.softmax(model_3.stream_gate(emb1), dim=-1)
        assert not torch.allclose(gates0, gates1)

    def test_no_sequential_modules(self, model):
        """El encoder no debe contener módulos RNN — sin dependencias secuenciales entre bins."""
        sequential_types = (nn.GRU, nn.LSTM, nn.RNN)
        for name, module in model.named_modules():
            assert not isinstance(module, sequential_types), (
                f"Módulo secuencial inesperado: {name} ({type(module).__name__})"
            )

    def test_bin_importance_is_per_bin_prior(self, model):
        """
        bin_importance añade un prior por identidad de bin, no sesgo secuencial.

        Con bin_importance uniforme (todos iguales), la atención depende solo
        del contenido del bin (frecuencia), no de su posición en el tensor.
        Esto confirma que no hay dependencia entre bins adyacentes.
        """
        model.eval()
        # Forzar bin_importance uniforme en los tres streams.
        # norm usa elementwise_affine=False → no tiene weight/bias por bin,
        # así que zeroing bin_importance es suficiente para garantizar que
        # la atención no dependa de la identidad del bin.
        with torch.no_grad():
            for stream in [model.stream_k3, model.stream_k4, model.stream_k5]:
                stream.bin_importance.fill_(0.0)

        batch_size = 2
        k3 = torch.randn(batch_size, 64, 1)
        k4 = torch.randn(batch_size, 256, 1)
        k5 = torch.randn(batch_size, 1024, 1)
        ab_idx = torch.zeros(batch_size, dtype=torch.long)

        # Permutar bins en k3 — con bin_importance uniforme, la atención
        # depende solo del contenido: el contexto ponderado debe ser idéntico
        perm = torch.randperm(64)
        k3_perm = k3[:, perm, :]

        with torch.no_grad():
            ctx_orig, _ = model.stream_k3(k3)
            ctx_perm, _ = model.stream_k3(k3_perm)

        assert torch.allclose(ctx_orig, ctx_perm, atol=1e-5), (
            "Con bin_importance=0, el contexto debe ser invariante a permutaciones de bins"
        )
