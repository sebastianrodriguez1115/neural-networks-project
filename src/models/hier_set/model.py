"""
AMRHierSet — Modelo Hierarchical Set Encoder para predicción de AMR.

Trata los HIER_N_SEGMENTS segmentos recibidos como un conjunto (set): cada
segmento se proyecta de forma independiente y el modelo pondera su relevancia
mediante cross-attention query-key condicionada en el antibiótico, sin
dependencias secuenciales entre segmentos.

Diferencia clave con HierBiGRU: la BiGRU crea dependencias artificiales
entre segmentos adyacentes en el tensor (el estado oculto de s_i influye
sobre s_{i+1}). HierSet no impone esa dependencia.

Atención condicionada: score(s, a) = h_s · q_a / sqrt(D), donde
q_a = attn_query(ab_emb). A diferencia de la concatenación simple (que
añadiría un término constante entre segmentos, cancelado por softmax),
el producto escalar produce scores distintos por segmento según el antibiótico.

Alcance de la invarianza: el modelo es permutation-invariant sobre los
HIER_N_SEGMENTS segmentos que recibe. La pipeline (prepare-hier) construye
esos segmentos concatenando contigs en orden FASTA y cortando linealmente —
HierSet no añade sesgo secuencial adicional, pero no elimina el que ya
codifican las features.
"""

import math

import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_pipeline.constants import ANTIBIOTIC_EMBEDDING_DIM, HIER_KMER_DIM


# Hiperparámetros
D_MODEL = 128
CLASSIFIER_HIDDEN = 128
DROPOUT = 0.3

# Clasificador: 128 (contexto) + 49 (embedding antibiótico) = 177
CLASSIFIER_INPUT = D_MODEL + ANTIBIOTIC_EMBEDDING_DIM


class AMRHierSet(nn.Module):
    """
    Encoder de conjunto para histogramas segmentados.

    Pipeline:
        1. LayerNorm por segmento — normaliza la escala del histograma
        2. Proyección independiente: histograma [256] → representación [128] + Dropout
        3. Cross-attention query-key: el antibiótico genera una query y cada
           segmento es una key. score(s,a) = h_s·q_a/√D — varía por segmento
           Y antibiótico. Softmax → weighted sum → contexto [128].
        4. Fusión con embedding del antibiótico + clasificador

    El modelo no impone dependencias secuenciales entre los HIER_N_SEGMENTS
    segmentos: la representación de s_i no depende de s_{i-1} ni s_{i+1}. Es
    permutation-invariant sobre sus inputs tal como los recibe del pipeline.
    """

    def __init__(self, n_antibiotics: int) -> None:
        super().__init__()

        # Embedding de antibiótico
        self.antibiotic_embedding = nn.Embedding(n_antibiotics, ANTIBIOTIC_EMBEDDING_DIM)

        # Normalización del histograma por segmento
        self.norm = nn.LayerNorm(HIER_KMER_DIM)

        # Proyección independiente por segmento (sin dependencias entre segmentos)
        self.proj = nn.Linear(HIER_KMER_DIM, D_MODEL)
        self.proj_dropout = nn.Dropout(DROPOUT)

        # Cross-attention query-key: el antibiótico genera una query y cada
        # segmento es una key. La concatenación simple (h_s ++ ab) no funciona
        # porque el término antibiótico sería constante entre segmentos y
        # softmax lo cancelaría (shift-invariance). Aquí score(s,a) = h_s·q_a,
        # que sí varía entre segmentos según el antibiótico.
        self.attn_query = nn.Linear(ANTIBIOTIC_EMBEDDING_DIM, D_MODEL, bias=False)

        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(CLASSIFIER_INPUT, CLASSIFIER_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(CLASSIFIER_HIDDEN, 1),
        )

        # Pesos de atención para interpretabilidad
        self._attention_weights: torch.Tensor | None = None

    def forward(self, genome: torch.Tensor, antibiotic_idx: torch.Tensor) -> torch.Tensor:
        """
        Parámetros:
            genome: [batch, HIER_N_SEGMENTS, 256] — HIER_N_SEGMENTS segmentos de histogramas k=4
            antibiotic_idx: [batch]

        Retorna:
            logits: [batch, 1]
        """
        # 1. Normalizar cada segmento de forma independiente
        x = self.norm(genome)                                    # [batch, S, 256]

        # 2. Proyección independiente por segmento (sin estado compartido)
        h = self.proj_dropout(F.relu(self.proj(x)))              # [batch, S, 128]

        # 3. Cross-attention: antibiótico como query, segmentos como keys.
        #    score(s, a) = h_s · q_a / sqrt(D) — varía por segmento Y antibiótico.
        ab_emb = self.antibiotic_embedding(antibiotic_idx)           # [batch, 49]
        query = self.attn_query(ab_emb)                              # [batch, 128]
        scores = torch.bmm(h, query.unsqueeze(-1)).squeeze(-1)       # [batch, S]
        scores = scores / math.sqrt(D_MODEL)
        alpha = F.softmax(scores, dim=1)                             # [batch, S]
        context = (alpha.unsqueeze(-1) * h).sum(dim=1)           # [batch, 128]
        self._attention_weights = alpha.detach()

        # 4. Fusión con embedding del antibiótico y clasificación
        return self.classifier(torch.cat([context, ab_emb], dim=1))

    @classmethod
    def from_antibiotic_index(cls, path: str) -> "AMRHierSet":
        """Factory method para instanciación desde la CLI."""
        return cls(n_antibiotics=len(pandas.read_csv(path)))
