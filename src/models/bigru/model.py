"""
AMRBiGRU — Modelo de Deep Learning para predicción de resistencia antimicrobiana.

Implementa una arquitectura BiGRU con mecanismo de atención aditiva (Bahdanau),
basada en el diseño de [Lugo21].
"""

import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_pipeline.constants import ANTIBIOTIC_EMBEDDING_DIM


# Hiperparámetros de la BiGRU [Lugo21, p. 648]:
# "a bidirectional GRU hidden layer with 128 units"
GRU_INPUT_SIZE = 3               # 3 features por timestep (k=3,4,5) [Lugo21, p. 647]
GRU_HIDDEN_SIZE = 128            # [Lugo21, p. 648]
GRU_NUM_LAYERS = 1               # [Lugo21, p. 648]: una sola capa recurrente
GRU_OUTPUT_DIM = GRU_HIDDEN_SIZE * 2  # 256 — forward + backward [Schuster97]

# Atención [Bahdanau15] / [Luong15]
ATTENTION_DIM = 128              # dimensión interna del espacio de atención

# Clasificador
CLASSIFIER_HIDDEN = 128          # capa densa tras concatenación

# Regularización mediante Dropout [Srivastava14; Haykin, Cap. 4.14].
# MEJORA1: reducido de 0.5 [Lugo21, p. 651] a 0.3 para estabilizar el
# entrenamiento. Dropout=0.5 causó oscilaciones excesivas en val_F1
# (rango 0.72–0.85). Para clasificadores pequeños (305→128→1),
# 0.3 proporciona regularización suficiente con menor varianza [Srivastava14].
DROPOUT = 0.3


class BahdanauAttention(nn.Module):
    """
    Mecanismo de atención aditiva [Bahdanau15].

    Calcula un vector de contexto como una suma ponderada de los estados
    ocultos de la BiGRU, permitiendo al modelo enfocarse en las partes
    más relevantes de la secuencia genómica.
    """

    def __init__(self, hidden_dim: int, attention_dim: int) -> None:
        """
        Inicializa las proyecciones para el cálculo de energía.

        Parámetros:
            hidden_dim: dimensión de los estados ocultos (256 para BiGRU 128)
            attention_dim: dimensión del espacio latente de atención
        """
        super().__init__()
        # Proyección de estados ocultos al espacio de atención
        self.W_a = nn.Linear(hidden_dim, attention_dim, bias=False)
        # Vector de contexto aprendido que produce el score escalar
        self.v_a = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, gru_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula el vector de contexto y los pesos de atención.

        Implementa las ecuaciones de [Bahdanau15, §A.1.2]:
        1. e_t = v_a^T * tanh(W_a * h_t)
        2. alpha_t = softmax(e_t)
        3. context = sum(alpha_t * h_t)

        Parámetros:
            gru_outputs: [batch, seq_len, hidden_dim]

        Retorna:
            (context, attention_weights)
        """
        # 1. Atención aditiva [Bahdanau15, Eq. 6]: e_t = v_a^T · tanh(W_a · h_t)
        # Calcula la "energía" de cada timestep — cuánta información relevante
        # contiene para la predicción de resistencia.
        energy = self.v_a(torch.tanh(self.W_a(gru_outputs)))  # [batch, 1024, 1]
        energy = energy.squeeze(-1)  # [batch, 1024]

        # 2. Distribución de probabilidad sobre la secuencia [Goodfellow16, Cap. 6.2.2.3]
        # La función softmax crea pesos normalizados que compiten por relevancia [Haykin, Cap. 9.7].
        alpha = F.softmax(energy, dim=1)  # [batch, 1024]

        # 3. Suma ponderada para obtener el vector de contexto [Lugo21, p. 648]
        # context = sum_t (alpha_t * h_t)
        # alpha.unsqueeze(1) -> [batch, 1, 1024]
        # gru_outputs -> [batch, 1024, 256]
        # bmm -> [batch, 1, 256]
        context = torch.bmm(alpha.unsqueeze(1), gru_outputs).squeeze(1)  # [batch, 256]

        return context, alpha


class AMRBiGRU(nn.Module):
    """
    Arquitectura BiGRU + Atención para predicción de resistencia.

    Basada en [Lugo21, Fig. 2, p. 651]:
    Input [1024, 3] -> BiGRU(128) -> Attention -> Concat(AB_Emb) -> MLP -> Logit
    """

    def __init__(self, n_antibiotics: int) -> None:
        """
        Inicializa las capas del modelo.

        Parámetros:
            n_antibiotics: número total de antibióticos para el embedding.
        """
        super().__init__()

        # Mapeo de categorías a espacio continuo [Haykin, Cap. 7.1]
        self.antibiotic_embedding = nn.Embedding(
            n_antibiotics, ANTIBIOTIC_EMBEDDING_DIM
        )

        # Red Recurrente [Cho14; Haykin, Cap. 15] con 128 unidades [Lugo21, p. 648].
        # La bidireccionalidad [Schuster97] captura contexto en ambas direcciones.
        self.gru = nn.GRU(
            input_size=GRU_INPUT_SIZE,
            hidden_size=GRU_HIDDEN_SIZE,
            num_layers=GRU_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
        )

        # Mecanismo de atención aditiva [Bahdanau15] para comprimir la secuencia.
        self.attention = BahdanauAttention(
            hidden_dim=GRU_OUTPUT_DIM, attention_dim=ATTENTION_DIM
        )

        # Clasificador con compresión progresiva (cuello de botella [Goodfellow16, Cap. 14.4])
        # Entrada: 256 (contexto BiGRU) + 49 (embedding antibiótico) = 305
        self.classifier = nn.Sequential(
            nn.Linear(GRU_OUTPUT_DIM + ANTIBIOTIC_EMBEDDING_DIM, CLASSIFIER_HIDDEN),
            nn.ReLU(),
            # Regularización [Lugo21, p. 651; Srivastava14; Haykin, Cap. 4.14]
            nn.Dropout(DROPOUT),
            nn.Linear(CLASSIFIER_HIDDEN, 1),
        )

        # Almacén de pesos para interpretabilidad [Lugo21, p. 648]
        self._attention_weights: torch.Tensor | None = None

    def forward(self, genome: torch.Tensor, antibiotic_idx: torch.Tensor) -> torch.Tensor:
        """
        Paso forward del modelo.

        Parámetros:
            genome: [batch, 1024, 3]
            antibiotic_idx: [batch]

        Retorna:
            logits: [batch, 1]
        """
        # 1. BiGRU procesa la secuencia en ambas direcciones [Schuster97]
        # gru_out shape: [batch, 1024, 256]
        gru_out, _ = self.gru(genome)

        # 2. Atención comprime la secuencia en un vector de contexto [Bahdanau15]
        # context shape: [batch, 256]
        context, attn_weights = self.attention(gru_out)

        # Almacenar para análisis posterior (detach para liberar memoria de grafos)
        self._attention_weights = attn_weights.detach()

        # 3. Fusión multimodal: contexto genómico + identidad del antibiótico
        ab_emb = self.antibiotic_embedding(antibiotic_idx)  # [batch, 49]
        x = torch.cat([context, ab_emb], dim=1)  # [batch, 305]

        # 4. Clasificación final
        return self.classifier(x)

    @classmethod
    def from_antibiotic_index(cls, path: str) -> "AMRBiGRU":
        """
        Crea una instancia del modelo leyendo el número de antibióticos de un CSV.
        Patrón de fábrica para facilitar la instanciación desde la CLI.
        """
        df = pandas.read_csv(path)
        return cls(n_antibiotics=len(df))
