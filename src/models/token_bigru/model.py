"""
AMRTokenBiGRU — Modelo BiGRU con tokenización de k-meros.

A diferencia de los modelos BiGRU anteriores que procesan histogramas de
frecuencia, este modelo recibe la secuencia de k-meros como tokens discretos
y aprende embeddings densos [Mikolov13]. Esto preserva el orden y contexto
posicional del genoma, permitiendo que la BiGRU capture patrones secuenciales
reales [Cho14; Schuster97].
"""

import pandas
import torch
import torch.nn as nn

from data_pipeline.constants import (
    ANTIBIOTIC_EMBEDDING_DIM,
    TOKEN_VOCAB_SIZE,
    TOKEN_PAD_ID,
    TOKEN_EMBED_DIM,
)
from models.bigru.model import BahdanauAttention

# BiGRU [Cho14; Lugo21, p. 648]
GRU_HIDDEN_SIZE = 128            # Mismo que BiGRU original
GRU_NUM_LAYERS = 2               # Dos capas — necesario para activar dropout recurrente [Srivastava14]
GRU_OUTPUT_DIM = GRU_HIDDEN_SIZE * 2  # 256 — forward + backward [Schuster97]

# Atención [Bahdanau15]
ATTENTION_DIM = 128              # Dimensión interna del espacio de atención

# Clasificador
CLASSIFIER_HIDDEN = 128          # Capa densa tras concatenación

# Regularización [Srivastava14]
DROPOUT = 0.3                    # Mismo que MEJORA1 — clasificador pequeño


class AMRTokenBiGRU(nn.Module):
    """
    Arquitectura BiGRU + Attention con tokenización de k-meros.

    A diferencia de AMRBiGRU que procesa histogramas [1024, 3], este modelo:
    1. Recibe una secuencia de token IDs [batch, seq_len] (enteros)
    2. Los mapea a embeddings densos vía nn.Embedding [Mikolov13]
    3. Procesa la secuencia de embeddings con una BiGRU [Cho14; Schuster97]
    4. Comprime con atención aditiva [Bahdanau15]
    5. Clasifica con la misma cabeza que los modelos anteriores

    Esto devuelve la BiGRU a su uso idiomático: procesar secuencias de
    tokens discretos, como en NLP [Cho14; Bahdanau15], en lugar de
    recorrer bins de un histograma.

    Arquitectura:
        tokens [batch, 4096] → Embedding(257, 64) → [batch, 4096, 64]
        → BiGRU(128, layers=2, dropout=0.3) → [batch, 4096, 256] → Attention → ctx [batch, 256]
        → Concat(ab_emb [49]) → [batch, 305] → MLP → logit [batch, 1]
    """

    def __init__(self, n_antibiotics: int) -> None:
        super().__init__()

        # Embedding de k-meros: mapea tokens discretos a vectores densos [Mikolov13].
        # vocab_size = 257 (256 k-meros válidos + 1 token de padding).
        # padding_idx=256: el token de padding se mapea al vector cero,
        # para que no contribuya a la representación [Goodfellow16, Cap. 12.4].
        self.kmer_embedding = nn.Embedding(
            num_embeddings=TOKEN_VOCAB_SIZE + 1,  # 257
            embedding_dim=TOKEN_EMBED_DIM,         # 64
            padding_idx=TOKEN_PAD_ID,              # 256
        )

        # Embedding de antibiótico [Haykin, Cap. 7.1]
        self.antibiotic_embedding = nn.Embedding(
            n_antibiotics, ANTIBIOTIC_EMBEDDING_DIM
        )

        # BiGRU [Cho14; Schuster97; Lugo21, p. 648]
        # input_size=64 (dimensión del embedding) en lugar de 3 (histograma).
        # dropout=DROPOUT: aplicado a la salida de cada capa excepto la última
        # [Srivastava14]. Requiere num_layers >= 2 — con una sola capa nn.GRU
        # ignora el parámetro dropout.
        self.gru = nn.GRU(
            input_size=TOKEN_EMBED_DIM,      # 64
            hidden_size=GRU_HIDDEN_SIZE,     # 128
            num_layers=GRU_NUM_LAYERS,       # 2
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT,                 # 0.3 entre capas recurrentes
        )

        # Atención aditiva [Bahdanau15]
        self.attention = BahdanauAttention(
            hidden_dim=GRU_OUTPUT_DIM,       # 256
            attention_dim=ATTENTION_DIM,     # 128
        )

        # Clasificador [Lugo21, p. 648; Goodfellow16, Cap. 14.4]
        # Entrada: 256 (contexto BiGRU) + 49 (embedding antibiótico) = 305
        self.classifier = nn.Sequential(
            nn.Linear(GRU_OUTPUT_DIM + ANTIBIOTIC_EMBEDDING_DIM, CLASSIFIER_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(CLASSIFIER_HIDDEN, 1),
        )

        # Almacén de pesos de atención para interpretabilidad
        self._attention_weights: torch.Tensor | None = None

    def forward(self, tokens: torch.Tensor, antibiotic_idx: torch.Tensor) -> torch.Tensor:
        """
        Paso forward del modelo.

        Parámetros:
            tokens: [batch, seq_len] — secuencia de token IDs (long)
            antibiotic_idx: [batch]

        Retorna:
            logits: [batch, 1]
        """
        # 1. Embedding: tokens discretos → vectores densos [Mikolov13]
        # [batch, 4096] → [batch, 4096, 64]
        x = self.kmer_embedding(tokens)

        # 2. BiGRU procesa la secuencia en ambas direcciones [Schuster97]
        # [batch, 4096, 64] → [batch, 4096, 256]
        gru_out, _ = self.gru(x)

        # 3. Atención comprime la secuencia en un vector de contexto [Bahdanau15]
        # [batch, 4096, 256] → context [batch, 256], alpha [batch, 4096]
        context, attn_weights = self.attention(gru_out)
        self._attention_weights = attn_weights.detach()

        # 4. Fusión: contexto genómico + identidad del antibiótico
        ab_emb = self.antibiotic_embedding(antibiotic_idx)  # [batch, 49]
        x = torch.cat([context, ab_emb], dim=1)              # [batch, 305]

        # 5. Clasificación final
        return self.classifier(x)

    @classmethod
    def from_antibiotic_index(cls, path: str) -> "AMRTokenBiGRU":
        """Factory method: lee n_antibiotics desde antibiotic_index.csv."""
        df = pandas.read_csv(path)
        return cls(n_antibiotics=len(df))
