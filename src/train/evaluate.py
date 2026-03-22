"""
evaluate.py — Evaluación de modelos y cálculo de métricas.

Funciones para inferencia sin gradientes, cálculo de métricas de
clasificación binaria, y búsqueda del umbral óptimo de decisión.
"""

import numpy
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[numpy.ndarray, numpy.ndarray, float]:
    """
    Ejecuta inferencia sobre un DataLoader sin modificar pesos.

    Recorre todos los batches en modo eval (sin gradientes ni dropout),
    acumula las probabilidades predichas, las etiquetas reales, y la
    pérdida promedio.

    Retorna:
        (probabilities, targets, loss) donde:
        - probabilities: array (n_samples,) con P(Resistant) ∈ [0, 1]
        - targets: array (n_samples,) con etiquetas reales (0.0 o 1.0)
        - loss: pérdida promedio sobre todos los batches
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for genome, antibiotic_idx, label in loader:
            genome = genome.to(device)
            antibiotic_idx = antibiotic_idx.to(device)
            label = label.to(device).unsqueeze(1)

            logits = model(genome, antibiotic_idx)
            loss = criterion(logits, label)
            total_loss += loss.item()
            n_batches += 1

            # Convertir logits a probabilidades con sigmoid
            probabilities = torch.sigmoid(logits).cpu().numpy().ravel()
            all_probabilities.append(probabilities)
            all_targets.append(label.cpu().numpy().ravel())

    return (
        numpy.concatenate(all_probabilities),
        numpy.concatenate(all_targets),
        total_loss / n_batches,
    )


def compute_metrics(
    targets: numpy.ndarray,
    probabilities: numpy.ndarray,
    loss: float,
    threshold: float = 0.5,
) -> dict:
    """
    Calcula métricas de clasificación binaria a partir de arrays numpy.

    Métricas calculadas:
        - loss: pérdida (pasada directamente, no recalculada)
        - accuracy: proporción de predicciones correctas
        - precision: de los que predijo Resistant, cuántos lo son realmente
        - recall (sensibilidad): de los realmente Resistant, cuántos detectó
        - f1: media armónica de precision y recall
        - auc_roc: área bajo la curva ROC (calidad del ranking de probabilidades)

    Parámetros:
        targets: etiquetas reales (0.0 o 1.0)
        probabilities: probabilidades predichas ∈ [0, 1]
        loss: pérdida promedio (calculada durante inferencia)
        threshold: umbral de decisión para convertir probabilidades en clases
    """
    predictions = (probabilities >= threshold).astype(int)
    return {
        "loss": loss,
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(
            targets,
            predictions,
            zero_division=0,  # type: ignore[arg-type]
        ),
        "recall": recall_score(
            targets,
            predictions,
            zero_division=0,  # type: ignore[arg-type]
        ),
        "f1": f1_score(
            targets,
            predictions,
            zero_division=0,  # type: ignore[arg-type]
        ),
        "auc_roc": roc_auc_score(targets, probabilities),
    }


def find_optimal_threshold(
    targets: numpy.ndarray,
    probabilities: numpy.ndarray,
) -> float:
    """
    Encuentra el umbral de decisión que maximiza F1-score.

    Con clases desbalanceadas, el umbral por defecto de 0.5 no es
    necesariamente el mejor punto de decisión (Haykin, 2009, §1.4 —
    Clasificador de Bayes). Este método evalúa cada probabilidad única
    como candidato de umbral y selecciona el que produce el mayor F1.

    Retorna:
        Umbral óptimo ∈ [0, 1].
    """
    unique_thresholds = numpy.unique(probabilities)
    best_f1 = -1.0
    best_threshold = 0.5

    for t in unique_thresholds:
        predictions = (probabilities >= t).astype(int)
        f1 = f1_score(
            targets,
            predictions,
            zero_division=0,  # type: ignore[arg-type]
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    return best_threshold


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """
    Evalúa el modelo sobre un conjunto (val o test) sin modificar pesos.

    Compone tres pasos:
        1. Inferencia: recolecta probabilidades y etiquetas (collect_predictions)
        2. Métricas: calcula accuracy, precision, recall, F1, AUC-ROC (compute_metrics)
        3. Umbral óptimo: busca el threshold que maximiza F1 (find_optimal_threshold)

    Retorna:
        dict con claves: loss, accuracy, precision, recall, f1, auc_roc,
        optimal_threshold.
    """
    probabilities, targets, loss = collect_predictions(model, loader, criterion, device)
    metrics = compute_metrics(targets, probabilities, loss, threshold)
    metrics["optimal_threshold"] = find_optimal_threshold(targets, probabilities)
    return metrics
