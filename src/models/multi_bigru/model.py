"""
AMRMultiBiGRU — Modelo Multi-Stream BiGRU para predicción de AMR.

Procesa cada histograma de k-meros (k=3, 4, 5) con una BiGRU separada, eliminando
el padding y permitiendo que la atención opere sobre información real [Ngiam11].
"""

import pandas
import torch
import torch.nn as nn
from typing import Sequence

from models.bigru.model import BahdanauAttention
from data_pipeline.constants import ANTIBIOTIC_EMBEDDING_DIM


# Hiperparámetros del Multi-Stream BiGRU
# GRU hidden_size reducido para mantener el conteo total de parámetros
# comparable al BiGRU v1 (~177K).
STREAM_GRU_HIDDEN = 64               # hidden size por GRU individual
STREAM_GRU_OUTPUT = STREAM_GRU_HIDDEN * 2  # 128 — forward + backward [Schuster97]
NUM_STREAMS = 3                       # una GRU por k ∈ {3, 4, 5}

# Atención [Bahdanau15]
STREAM_ATTENTION_DIM = 64             # dimensión interna del espacio de atención

# Clasificador
# Entrada: 3 contextos × 128 dims + 49 embedding = 433
CLASSIFIER_INPUT = NUM_STREAMS * STREAM_GRU_OUTPUT + ANTIBIOTIC_EMBEDDING_DIM
CLASSIFIER_HIDDEN = 128

# Regularización [Srivastava14]
DROPOUT = 0.3


class KmerStream(nn.Module):
    """
    Stream individual: BiGRU + Atención para un histograma de k-meros.

    Cada stream procesa una secuencia de longitud variable (64, 256, o 1024)
    sin padding, permitiendo que la atención se enfoque en información real
    en lugar de ceros [Haykin, Cap. 15].
    """

    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self._seq_len = seq_len  # Almacenado para debug y repr
        self.gru = nn.GRU(
            input_size=1,                    # 1 feature por timestep
            hidden_size=STREAM_GRU_HIDDEN,   # 64
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = BahdanauAttention(
            hidden_dim=STREAM_GRU_OUTPUT,    # 128
            attention_dim=STREAM_ATTENTION_DIM,  # 64
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Procesa un histograma de k-meros.

        Parámetros:
            x: [batch, seq_len, 1] (frecuencia normalizada del k-mero)

        Retorna:
            (context, attention_weights)
            context: [batch, 128] (vector de contexto BiGRU)
            attention_weights: [batch, seq_len] (distribución de energía)
        """
        # BiGRU [Cho14; Schuster97]
        gru_out, _ = self.gru(x)      # [batch, seq_len, 128]
        context, alpha = self.attention(gru_out)
        return context, alpha


class AMRMultiBiGRU(nn.Module):
    """
    Arquitectura Multi-Stream BiGRU + Atención para predicción de AMR.

    Procesa cada histograma de k-meros (k=3,4,5) con una BiGRU separada para
    eliminar el ruido del padding y capturar interacciones multiescala [Ngiam11].
    """

    def __init__(self, n_antibiotics: int) -> None:
        super().__init__()

        # Un stream BiGRU+Attention por cada tamaño de k-mero [Lugo21, p. 647]
        self.stream_k3 = KmerStream(seq_len=64)
        self.stream_k4 = KmerStream(seq_len=256)
        self.stream_k5 = KmerStream(seq_len=1024)

        # Embedding de antibiótico [Haykin, Cap. 7.1]
        self.antibiotic_embedding = nn.Embedding(
            n_antibiotics, ANTIBIOTIC_EMBEDDING_DIM
        )

        # Clasificador con compresión progresiva [Goodfellow16, Cap. 14.4]
        self.classifier = nn.Sequential(
            nn.Linear(CLASSIFIER_INPUT, CLASSIFIER_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(CLASSIFIER_HIDDEN, 1),
        )

        # Pesos de atención por stream para interpretabilidad
        self._attention_weights: dict[str, torch.Tensor] | None = None

    def forward(
        self,
        genome: Sequence[torch.Tensor],
        antibiotic_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Paso forward del modelo multi-stream.

        Parámetros:
            genome: secuencia (list o tuple) de 3 tensores (k3, k4, k5):
                k3: [batch, 64, 1]
                k4: [batch, 256, 1]
                k5: [batch, 1024, 1]
            antibiotic_idx: [batch]

        Retorna:
            logits: [batch, 1]
        """
        k3, k4, k5 = genome

        # 1. Cada stream procesa su histograma de forma independiente [Bahdanau15]
        ctx3, alpha3 = self.stream_k3(k3)
        ctx4, alpha4 = self.stream_k4(k4)
        ctx5, alpha5 = self.stream_k5(k5)

        # Almacenar para análisis posterior
        self._attention_weights = {
            "k3": alpha3.detach(),
            "k4": alpha4.detach(),
            "k5": alpha5.detach(),
        }

        # 2. Fusión tardía [Ngiam11]: concatenar contextos + embedding
        ab_emb = self.antibiotic_embedding(antibiotic_idx)
        x = torch.cat([ctx3, ctx4, ctx5, ab_emb], dim=1)

        # 3. Clasificación final
        return self.classifier(x)

    @classmethod
    def from_antibiotic_index(cls, path: str) -> "AMRMultiBiGRU":
        """Factory method para instanciación desde la CLI."""
        df = pandas.read_csv(path)
        return cls(n_antibiotics=len(df))
