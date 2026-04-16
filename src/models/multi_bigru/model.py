"""
AMRMultiBiGRU — Modelo Multi-Stream para predicción de AMR.

Procesa cada histograma de k-meros (k=3, 4, 5) con un encoder sin dependencias
secuenciales entre bins (proyección element-wise + attention pooling). A diferencia
de la BiGRU, la representación de cada bin no está condicionada por qué otros bins
la precedieron en el tensor — elimina el sesgo artificial de orden sobre índices
de k-meros [Goodfellow16, Cap. 10].

Nota sobre invarianza: los bins tienen identidad fija (cada índice siempre
representa el mismo k-mero, vía rolling hash). El modelo aprende importancias
por bin (bin_importance) que son biológicamente válidas — no es equivalente al
sesgo secuencial de la BiGRU porque no hay dependencia entre bins adyacentes.
La fusión entre streams está condicionada por el antibiótico [Ngiam11].
"""

import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

from data_pipeline.constants import ANTIBIOTIC_EMBEDDING_DIM


# Hiperparámetros
STREAM_D_MODEL = 128             # dimensión del espacio de representación por stream
NUM_STREAMS = 3                  # una por k ∈ {3, 4, 5}

# Clasificador
# Entrada: 128 (contexto fusionado) + 49 (embedding antibiótico) = 177
CLASSIFIER_INPUT = STREAM_D_MODEL + ANTIBIOTIC_EMBEDDING_DIM
CLASSIFIER_HIDDEN = 128

# Regularización [Srivastava14]
DROPOUT = 0.3


class KmerStream(nn.Module):
    """
    Stream individual: encoder sin dependencias secuenciales + attention pooling.

    Cada bin del histograma se proyecta de forma independiente — no hay estado
    oculto que fluya de bin_i a bin_{i+1}. Esto elimina el sesgo artificial que
    introduce la BiGRU sobre índices de k-meros [Goodfellow16, Cap. 10].

    bin_importance asigna un prior aprendido por bin (= por k-mero específico).
    No rompe la independencia entre bins: bin_i y bin_j no se influyen
    mutuamente. Lo que agrega es que k-meros diagnósticos adquieren scores
    más altos independientemente de su frecuencia.

    Pipeline:
        1. LayerNorm(elementwise_affine=False) — normaliza la escala sin
           introducir pesos/bias por bin. bin_importance es el único prior
           aprendido por identidad de bin.
        2. Proyección element-wise: frecuencia escalar → vector de representación
        3. Attention pooling: cada bin ponderado por relevancia aprendida
    """

    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self._seq_len = seq_len
        # 1. LayerNorm sobre los bins del histograma (elementwise_affine=False:
        #    sin pesos/bias por bin — bin_importance es el único prior aprendido
        #    por identidad de bin, no LayerNorm)
        self.norm = nn.LayerNorm(seq_len, elementwise_affine=False)
        # 2. Proyección element-wise: mismos pesos para todos los bins
        self.freq_proj = nn.Linear(1, STREAM_D_MODEL)
        # Prior aprendido por bin: k-meros diagnósticos reciben scores más altos.
        # No crea dependencia secuencial — cada bin sigue procesándose de forma
        # independiente, solo se añade un escalar fijo por identidad de bin.
        self.bin_importance = nn.Parameter(torch.zeros(seq_len))
        # Proyección para el score de atención
        self.attn = nn.Linear(STREAM_D_MODEL, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parámetros:
            x: [batch, seq_len, 1] — frecuencias del histograma

        Retorna:
            context: [batch, STREAM_D_MODEL]
            alpha:   [batch, seq_len] — pesos de atención por bin
        """
        # 1. Normalizar el histograma dentro del modelo
        x_normed = self.norm(x.squeeze(-1)).unsqueeze(-1)        # [batch, seq_len, 1]
        # 2. Proyección element-wise (sin dependencia secuencial entre bins)
        h = F.relu(self.freq_proj(x_normed))                     # [batch, seq_len, 128]
        # 3. Score: representación aprendida + prior global del bin (por identidad)
        scores = self.attn(h).squeeze(-1) + self.bin_importance  # [batch, seq_len]
        alpha = F.softmax(scores, dim=1)                         # [batch, seq_len]
        context = (alpha.unsqueeze(-1) * h).sum(dim=1)           # [batch, 128]
        return context, alpha


class AMRMultiBiGRU(nn.Module):
    """
    Arquitectura Multi-Stream + Fusión condicionada por antibiótico.

    Tres streams independientes (k=3,4,5) procesan cada histograma sin
    dependencias secuenciales entre bins. La fusión usa softmax para garantizar
    que los tres streams compitan y al menos uno permanezca activo — evita el
    camino degenerado donde el modelo ignora el genoma y usa solo el embedding
    del antibiótico. Los pesos de fusión (gates) reflejan qué escala de k-meros
    es más diagnóstica para cada antibiótico [Ngiam11].
    """

    def __init__(self, n_antibiotics: int) -> None:
        super().__init__()

        # Un stream por tamaño de k-mero
        self.stream_k3 = KmerStream(seq_len=64)
        self.stream_k4 = KmerStream(seq_len=256)
        self.stream_k5 = KmerStream(seq_len=1024)

        # Embedding de antibiótico [Haykin, Cap. 7.1]
        self.antibiotic_embedding = nn.Embedding(
            n_antibiotics, ANTIBIOTIC_EMBEDDING_DIM
        )

        # Pesos de fusión condicionados por antibiótico (softmax → suman 1,
        # garantiza que al menos un stream permanece activo)
        self.stream_gate = nn.Linear(ANTIBIOTIC_EMBEDDING_DIM, NUM_STREAMS)

        # Clasificador: contexto fusionado (128) + embedding antibiótico (49) = 177
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
        Parámetros:
            genome: (k3, k4, k5)
                k3: [batch, 64, 1]
                k4: [batch, 256, 1]
                k5: [batch, 1024, 1]
            antibiotic_idx: [batch]

        Retorna:
            logits: [batch, 1]
        """
        k3, k4, k5 = genome

        # 1. Cada stream procesa su histograma de forma independiente
        ctx3, alpha3 = self.stream_k3(k3)
        ctx4, alpha4 = self.stream_k4(k4)
        ctx5, alpha5 = self.stream_k5(k5)

        self._attention_weights = {
            "k3": alpha3.detach(),
            "k4": alpha4.detach(),
            "k5": alpha5.detach(),
        }

        # 2. Embedding del antibiótico
        ab_emb = self.antibiotic_embedding(antibiotic_idx)  # [batch, 49]

        # 3. Fusión condicionada por antibiótico (softmax: los tres streams compiten,
        # ninguno puede apagarse completamente como ocurriría con sigmoid independiente)
        gates = F.softmax(self.stream_gate(ab_emb), dim=-1)      # [batch, 3]
        contexts = torch.stack([ctx3, ctx4, ctx5], dim=1)        # [batch, 3, 128]
        fused = (gates.unsqueeze(-1) * contexts).sum(dim=1)      # [batch, 128]

        # 4. Clasificación final
        x = torch.cat([fused, ab_emb], dim=1)  # [batch, 177]
        return self.classifier(x)

    @classmethod
    def from_antibiotic_index(cls, path: str) -> "AMRMultiBiGRU":
        """Factory method para instanciación desde la CLI."""
        df = pandas.read_csv(path)
        return cls(n_antibiotics=len(df))
