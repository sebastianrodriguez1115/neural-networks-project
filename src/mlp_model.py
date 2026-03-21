"""
mlp_model.py — Perceptrón Multicapa para predicción de AMR.

MLP con embedding de antibiótico para clasificación binaria
Resistant/Susceptible.
"""

import pandas
import torch
from torch import nn

from data_pipeline.constants import TOTAL_KMER_DIM

# Dimensión del embedding de antibiótico: min(50, (n_antibiotics // 2) + 1)
# Con 96 antibióticos en el dataset completo → 49. Ver docs/2_eda.md.
ANTIBIOTIC_EMBEDDING_DIM = 49

# Arquitectura MLP (docs/4_models.md)
HIDDEN_1 = 512
HIDDEN_2 = 128
DROPOUT = 0.3


class AMRMLP(nn.Module):
    """
    Perceptrón Multicapa para predicción de resistencia antimicrobiana.

    Recibe un vector genómico (histograma de k-meros, 1344 dims) y un índice
    de antibiótico. El antibiótico se transforma en un embedding aprendido que
    se concatena con el vector genómico antes de pasar por las capas densas.

    Arquitectura:
        Concat(genome[1344], antibiotic_emb[49]) → 1393
        → Linear(1393, 512) + ReLU + Dropout(0.3)
        → Linear(512, 128)  + ReLU + Dropout(0.3)
        → Linear(128, 1)    → logit

    La salida es un logit (sin sigmoid). Usar BCEWithLogitsLoss para entrenar.
    """

    def __init__(self, n_antibiotics: int) -> None:
        """
        Parámetros:
            n_antibiotics: número total de antibióticos en el dataset
                           (define el tamaño de la tabla de embeddings)
        """
        super().__init__()
        self.antibiotic_embedding = nn.Embedding(n_antibiotics, ANTIBIOTIC_EMBEDDING_DIM)

        input_dim = TOTAL_KMER_DIM + ANTIBIOTIC_EMBEDDING_DIM

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_2, 1),
        )

    def forward(self, genome: torch.Tensor, antibiotic_idx: torch.Tensor) -> torch.Tensor:
        """
        Propagación hacia adelante.

        Parámetros:
            genome: tensor (batch, 1344) con vectores genómicos normalizados
            antibiotic_idx: tensor (batch,) con índices enteros de antibiótico

        Retorna:
            logits: tensor (batch, 1) — un logit por muestra
        """
        ab_emb = self.antibiotic_embedding(antibiotic_idx)  # (batch, 49)
        x = torch.cat([genome, ab_emb], dim=1)              # (batch, 1393)
        return self.classifier(x)                            # (batch, 1)

    @classmethod
    def from_antibiotic_index(cls, path: str) -> "AMRMLP":
        """
        Factory method: construye el modelo leyendo el número de antibióticos
        desde el CSV del pipeline (antibiotic_index.csv).
        """
        df = pandas.read_csv(path)
        return cls(n_antibiotics=len(df))
