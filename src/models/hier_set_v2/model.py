"""
AMRHierSetV2 — HierSet con multi-head cross-attention + histogramas multi-escala.

Dos cambios ortogonales respecto a AMRHierSet (v1):

1. **Input multi-escala:** cada segmento recibe la concatenación de histogramas
   k=3, k=4 y k=5 (1344 dims en vez de 256). El MLP baseline ya mostró que la
   información multi-escala aporta (composición nucleotídica + motivos cortos
   + motivos largos) [Lugo21].

2. **Multi-head cross-attention (H=4, d_head=32):** la atención condicionada
   por antibiótico se divide en H subespacios paralelos. Cada cabeza puede
   especializarse en un patrón distinto de pesos sobre los segmentos
   (enzimas inactivadoras, bombas de eflujo, modificación del target,
   impermeabilidad). Equivalente escalar a v1 cuando H=1, sin costo extra
   de parámetros en la proyección de query [Vaswani17].

Score por cabeza: score_h(s, a) = h_s^{(h)} · q_a^{(h)} / sqrt(d_head).
"""

import math

import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_pipeline.constants import ANTIBIOTIC_EMBEDDING_DIM, HIER_KMER_DIM_MULTI


# Hiperparámetros
D_MODEL = 128
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS  # 32
CLASSIFIER_HIDDEN = 128
DROPOUT = 0.3

CLASSIFIER_INPUT = D_MODEL + ANTIBIOTIC_EMBEDDING_DIM


class AMRHierSetV2(nn.Module):
    """
    HierSet v2: multi-head cross-attention sobre histogramas multi-escala.

    Pipeline:
        1. LayerNorm sobre [1344] por segmento.
        2. Proyección lineal 1344 → D_MODEL (128) + ReLU + Dropout.
        3. Multi-head cross-attention con query derivada del antibiótico:
           scores por cabeza → softmax → weighted sum → concat(cabezas).
        4. Fusión con embedding del antibiótico + clasificador.
    """

    def __init__(self, n_antibiotics: int) -> None:
        super().__init__()

        self.antibiotic_embedding = nn.Embedding(n_antibiotics, ANTIBIOTIC_EMBEDDING_DIM)

        # Normalización del histograma multi-escala por segmento (1344 dims)
        self.norm = nn.LayerNorm(HIER_KMER_DIM_MULTI)

        # Proyección independiente por segmento (única Linear que crece vs v1)
        self.proj = nn.Linear(HIER_KMER_DIM_MULTI, D_MODEL)
        self.proj_dropout = nn.Dropout(DROPOUT)

        # Query multi-cabeza: una Linear(49 → 128) que se reshape a [H, D_HEAD]
        self.attn_query = nn.Linear(ANTIBIOTIC_EMBEDDING_DIM, D_MODEL, bias=False)

        self.classifier = nn.Sequential(
            nn.Linear(CLASSIFIER_INPUT, CLASSIFIER_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(CLASSIFIER_HIDDEN, 1),
        )

        # Pesos de atención para interpretabilidad: [batch, H, S]
        self._attention_weights: torch.Tensor | None = None

    def forward(self, genome: torch.Tensor, antibiotic_idx: torch.Tensor) -> torch.Tensor:
        """
        Parámetros:
            genome: [batch, HIER_N_SEGMENTS, HIER_KMER_DIM_MULTI] (1344 dims/segmento)
            antibiotic_idx: [batch]

        Retorna:
            logits: [batch, 1]
        """
        # 1-2. Normalización y proyección por segmento
        x = self.norm(genome)                                    # [B, S, 1344]
        h = self.proj_dropout(F.relu(self.proj(x)))              # [B, S, 128]

        # 3. Multi-head cross-attention
        B, S, _ = h.shape
        ab_emb = self.antibiotic_embedding(antibiotic_idx)       # [B, 49]
        query = self.attn_query(ab_emb).view(B, N_HEADS, D_HEAD) # [B, H, D_HEAD]
        h_heads = h.view(B, S, N_HEADS, D_HEAD)                  # [B, S, H, D_HEAD]

        # scores[b, h, s] = h_heads[b, s, h, :] · query[b, h, :] / sqrt(D_HEAD)
        scores = torch.einsum("bshd,bhd->bhs", h_heads, query) / math.sqrt(D_HEAD)
        alpha = F.softmax(scores, dim=2)                         # [B, H, S]

        # context[b, h, :] = sum_s alpha[b, h, s] * h_heads[b, s, h, :]
        context_heads = torch.einsum("bhs,bshd->bhd", alpha, h_heads)  # [B, H, D_HEAD]
        context = context_heads.reshape(B, D_MODEL)                    # [B, 128]

        self._attention_weights = alpha.detach()

        # 4. Fusión + clasificación
        return self.classifier(torch.cat([context, ab_emb], dim=1))

    @classmethod
    def from_antibiotic_index(cls, path: str) -> "AMRHierSetV2":
        """Factory method para instanciación desde la CLI."""
        return cls(n_antibiotics=len(pandas.read_csv(path)))
