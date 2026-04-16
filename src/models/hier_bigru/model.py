"""
AMRHierBiGRU — Modelo Hierarchical BiGRU para predicción de AMR.

Procesa el genoma como una secuencia de histogramas locales (segmentos) para
capturar la presencia de genes de resistencia en su contexto geográfico,
garantizando 100% de cobertura sin el sesgo del subsampling [Haykin, Cap. 15].
"""

import pandas
import torch
import torch.nn as nn

from models.bigru.model import BahdanauAttention
from data_pipeline.constants import ANTIBIOTIC_EMBEDDING_DIM, HIER_KMER_DIM


# Hiperparámetros [Lugo21; Srivastava14; Pascanu13]
GRU_HIDDEN = 128                 # unidades ocultas por dirección
GRU_LAYERS = 2                   # apilado para capturar dependencias abstractas
GRU_OUTPUT_DIM = GRU_HIDDEN * 2  # 256 (bidireccional)

ATTENTION_DIM = 128              # espacio latente de atención
CLASSIFIER_HIDDEN = 128          # capa densa tras concatenación
DROPOUT = 0.3                    # regularización estándar del proyecto


class AMRHierBiGRU(nn.Module):
    """
    Arquitectura Hierarchical BiGRU + Atención.

    1. El genoma se representa como HIER_N_SEGMENTS segmentos contiguos de k-meros (k=4).
    2. Una BiGRU profunda de 2 capas procesa la secuencia de segmentos.
    3. Un mecanismo de atención identifica qué segmentos contienen información
       relevante para la resistencia (ej. ubicación de un plásmido o gen).
    4. El contexto genómico se fusiona con la identidad del antibiótico.
    """

    def __init__(self, n_antibiotics: int) -> None:
        """
        Inicializa las capas del modelo.

        Parámetros:
            n_antibiotics: número total de antibióticos para el embedding.
        """
        super().__init__()

        # Embedding de antibiótico [Haykin, Cap. 7.1]
        self.antibiotic_embedding = nn.Embedding(
            n_antibiotics, ANTIBIOTIC_EMBEDDING_DIM
        )

        # BiGRU Profunda [Cho14; Schuster97]
        # num_layers=2 permite aprender jerarquías de características y 
        # habilita el uso de dropout recurrente entre capas [Srivastava14].
        self.gru = nn.GRU(
            input_size=HIER_KMER_DIM,    # 256
            hidden_size=GRU_HIDDEN,       # 128
            num_layers=GRU_LAYERS,        # 2
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if GRU_LAYERS > 1 else 0,
        )

        # Mecanismo de atención aditiva [Bahdanau15] para resumir los segmentos.
        self.attention = BahdanauAttention(
            hidden_dim=GRU_OUTPUT_DIM, attention_dim=ATTENTION_DIM
        )

        # Clasificador final con compresión progresiva [Goodfellow16, Cap. 14.4]
        # Entrada: 256 (contexto BiGRU) + 49 (embedding antibiótico) = 305
        self.classifier = nn.Sequential(
            nn.Linear(GRU_OUTPUT_DIM + ANTIBIOTIC_EMBEDDING_DIM, CLASSIFIER_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(CLASSIFIER_HIDDEN, 1),
        )

        # Almacén de pesos para interpretabilidad genómica
        self._attention_weights: torch.Tensor | None = None

    def forward(self, genome: torch.Tensor, antibiotic_idx: torch.Tensor) -> torch.Tensor:
        """
        Paso forward del modelo jerárquico.

        Parámetros:
            genome: [batch, HIER_N_SEGMENTS, 256] — HIER_N_SEGMENTS segmentos de histogramas k=4.
            antibiotic_idx: [batch] — índices de antibióticos.

        Retorna:
            logits: [batch, 1]
        """
        # 1. BiGRU procesa la secuencia de histogramas locales
        # gru_out shape: [batch, HIER_N_SEGMENTS, 256]
        gru_out, _ = self.gru(genome)

        # 2. Atención comprime los HIER_N_SEGMENTS segmentos en un vector de contexto global
        # context shape: [batch, 256]
        context, attn_weights = self.attention(gru_out)
        self._attention_weights = attn_weights.detach()

        # 3. Fusión multimodal: contexto genómico + identidad del antibiótico
        ab_emb = self.antibiotic_embedding(antibiotic_idx)  # [batch, 49]
        x = torch.cat([context, ab_emb], dim=1)  # [batch, 305]

        # 4. Clasificación final (logit)
        return self.classifier(x)

    @classmethod
    def from_antibiotic_index(cls, path: str) -> "AMRHierBiGRU":
        """Factory method para instanciación desde la CLI."""
        df = pandas.read_csv(path)
        return cls(n_antibiotics=len(df))
