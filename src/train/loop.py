"""
loop.py — Ciclo de entrenamiento y utilidades asociadas.

Implementa el ciclo completo de aprendizaje supervisado (Haykin, 2009, Cap. 4):
el modelo recibe ejemplos etiquetados, calcula el error respecto a la etiqueta
conocida, y ajusta sus pesos mediante retropropagación del error para minimizar
la función de pérdida.
"""

import json
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import pandas
import torch
from torch import nn
from torch.utils.data import DataLoader

from .evaluate import evaluate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """
    Fija semillas en todos los generadores aleatorios para reproducibilidad.

    Garantiza que la trayectoria de pesos durante el entrenamiento sea
    idéntica entre ejecuciones (Haykin, 2009, §4.4): misma inicialización
    de pesos → misma secuencia de actualizaciones → mismos resultados.

    Afecta: random (Python), numpy, torch CPU, torch CUDA y cuDNN.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Detección de dispositivo
# ---------------------------------------------------------------------------


def detect_device() -> torch.device:
    """
    Detecta el mejor acelerador disponible: CUDA → MPS → CPU.

    Retorna un torch.device listo para usar con .to(device).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Gráficas de entrenamiento
# ---------------------------------------------------------------------------


def _plot_history(history: pandas.DataFrame, output_path: Path) -> None:
    """
    Genera gráficas de loss y F1 por época y las guarda como PNG.

    Dos subplots:
        1. Loss (train y val) vs épocas — muestra convergencia y posible
           overfitting cuando las curvas divergen (Haykin, 2009, Fig. 4.17)
        2. F1 (val) vs épocas — métrica de interés clínico
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Loss
    ax1.plot(history["epoch"], history["train_loss"], label="Train")
    ax1.plot(history["epoch"], history["val_loss"], label="Val")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.set_title("Pérdida por época")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: F1
    ax2.plot(history["epoch"], history["val_f1"], label="Val F1", color="green")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("F1")
    ax2.set_title("F1 por época")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Una época de entrenamiento
# ---------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float | None = None,
) -> float:
    """
    Ejecuta una época completa de entrenamiento (Haykin, 2009, §4.3–4.4).

    Para cada mini-batch:
        1. Propagación hacia adelante: calcula la salida de la red
        2. Cálculo del error: compara salida con etiqueta real (BCEWithLogitsLoss)
        3. Retropropagación: calcula gradientes ∂E/∂w para cada peso
        4. Actualización: Adam ajusta los pesos en dirección del gradiente

    El uso de mini-batches (en vez de todo el dataset) produce una estimación
    ruidosa del gradiente que ayuda a escapar mínimos locales (Haykin, §4.3).

    Retorna:
        loss promedio sobre todos los batches de la época.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for genome, antibiotic_idx, label in loader:
        # Mover datos al dispositivo (GPU/CPU)
        # MEJORA: Soporte para multi-stream (tupla de tensores) [Goodfellow16, Cap. 15]
        if isinstance(genome, (tuple, list)):
            genome = tuple(g.to(device) for g in genome)
        else:
            genome = genome.to(device)

        antibiotic_idx = antibiotic_idx.to(device)
        # label shape: [batch] → [batch, 1] para coincidir con la salida del modelo
        label = label.to(device).unsqueeze(1)

        # 1. Propagación hacia adelante: calcula la salida de la red (logits)
        logits = model(genome, antibiotic_idx)

        # 2. Cálculo del error: compara salida con etiqueta real
        loss = criterion(logits, label)

        # Limpiar gradientes acumulados antes de la retropropagación
        optimizer.zero_grad()

        # 3. Retropropagación: calcula gradientes ∂E/∂w para cada peso
        loss.backward()

        # Gradient clipping [Pascanu13]: limita la norma L2 del gradiente
        # global para prevenir la explosión de gradientes en redes recurrentes.
        # Con BPTT [Haykin, Cap. 15.3] sobre 1024 timesteps, los gradientes
        # pueden crecer exponencialmente sin esta protección [Goodfellow16, Cap. 10.7].
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # 4. Actualización: Adam ajusta los pesos en dirección del gradiente
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Evaluación final sobre test set
# ---------------------------------------------------------------------------


def _final_evaluation(
    model: nn.Module,
    best_model_path: Path,
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """
    Evalúa el mejor modelo guardado sobre el test set.

    Este paso es la evaluación definitiva del modelo entrenado
    (Haykin, 2009, §4.13 — validación cruzada). Se ejecuta una sola vez
    al final del entrenamiento, usando el checkpoint con mejor F1 en
    validación. El proceso:

        1. Carga los pesos del mejor checkpoint (mejor val F1)
        2. Evalúa sobre validación para obtener el umbral óptimo de decisión
        3. Evalúa sobre test con ese umbral (estimación final de generalización)
        4. Guarda las métricas en metrics.json

    Retorna:
        dict con métricas finales sobre test set.
    """
    # Restaurar los pesos del mejor modelo encontrado durante entrenamiento
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    # Buscar el umbral de decisión que maximiza F1 en validación.
    # Con clases desbalanceadas, 0.5 no es necesariamente óptimo.
    val_final = evaluate(model, val_loader, criterion, device)
    optimal_threshold = val_final["optimal_threshold"]
    logger.info("Umbral óptimo (val): %.4f", optimal_threshold)

    # Evaluar sobre test set con el umbral calibrado en validación.
    # Esta es la estimación final de cómo generalizará el modelo.
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        threshold=optimal_threshold,
    )
    test_metrics["threshold_used"] = optimal_threshold

    # Persistir métricas para análisis posterior
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, indent=2))

    logger.info(
        "Métricas en test: F1=%.4f, Recall=%.4f, AUC=%.4f",
        test_metrics["f1"],
        test_metrics["recall"],
        test_metrics["auc_roc"],
    )
    logger.info("Resultados guardados en %s", output_dir)

    return test_metrics


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    lr: float = 0.001,
    epochs: int = 100,
    patience: int = 10,
    output_dir: str | Path = "results/mlp",
    max_grad_norm: float | None = None,
    weight_decay: float = 0.0,
) -> dict:
    """
    Ciclo completo de entrenamiento con early stopping (Haykin, 2009, §4.13).

    El proceso sigue el paradigma de aprendizaje con un maestro:
        - En cada época, el modelo ve todos los ejemplos de entrenamiento
          y ajusta sus pesos para reducir el error (train_epoch).
        - Después de cada época, se evalúa sobre validación (evaluate)
          para monitorear la generalización.
        - Early stopping: si la pérdida de validación no mejora durante
          `patience` épocas consecutivas, se detiene el entrenamiento.
          Esto evita la fase de sobreentrenamiento donde la red memoriza
          ruido en vez de aprender patrones generales (Haykin, Fig. 4.17).
        - Se guarda el checkpoint con mejor F1 en validación, que es la
          métrica de interés clínico (no la loss).

    Al terminar, evalúa sobre test set y genera:
        - best_model.pt   — pesos del mejor modelo
        - metrics.json    — métricas finales sobre test
        - history.csv     — métricas por época
        - history.png     — gráficas de convergencia

    Parámetros:
        model: red neuronal (AMRMLP)
        train_loader, val_loader, test_loader: DataLoaders por partición
        criterion: función de pérdida (BCEWithLogitsLoss con pos_weight)
        device: dispositivo de cómputo
        lr: tasa de aprendizaje para Adam (default 0.001)
        weight_decay: regularización L2 (default 0.0)
        epochs: número máximo de épocas (default 100)
        patience: épocas sin mejora antes de parar (default 10)
        output_dir: directorio para guardar resultados

    Retorna:
        dict con métricas finales sobre test set.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)

    # AdamW (Loshchilov & Hutter, 2019) desacopla el weight decay del gradiente,
    # permitiendo una regularización más efectiva. Es la opción preferida
    # para arquitecturas recurrentes y transformers [Goodfellow16, Cap. 7].
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Estado de early stopping y checkpointing
    best_val_loss = float("inf")
    best_val_f1 = -1.0
    epochs_without_improvement = 0
    best_model_path = output_dir / "best_model.pt"

    # Historial de métricas por época
    history_rows: list[dict] = []

    logger.info(
        "Iniciando entrenamiento: %d épocas máx, patience=%d, lr=%.4f, device=%s",
        epochs,
        patience,
        lr,
        device,
    )

    for epoch in range(1, epochs + 1):
        # --- Entrenamiento ---
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            max_grad_norm=max_grad_norm,
        )

        # --- Evaluación sobre validación ---
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Registrar métricas de esta época
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc_roc": val_metrics["auc_roc"],
        }
        history_rows.append(row)

        logger.info(
            "Época %3d/%d — train_loss: %.4f | val_loss: %.4f | val_f1: %.4f | val_recall: %.4f",
            epoch,
            epochs,
            train_loss,
            val_metrics["loss"],
            val_metrics["f1"],
            val_metrics["recall"],
        )

        # --- Checkpoint: guardar si es el mejor F1 en validación ---
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_model_path)
            logger.info("  → Nuevo mejor modelo (val F1=%.4f), guardado.", best_val_f1)

        # --- Early stopping: monitorear val loss ---
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info(
                "Early stopping en época %d (sin mejora en val_loss por %d épocas).",
                epoch,
                patience,
            )
            break

    # --- Guardar historial ---
    history = pandas.DataFrame(history_rows)
    history.to_csv(output_dir / "history.csv", index=False)

    # --- Gráficas ---
    _plot_history(history, output_dir / "history.png")

    # --- Evaluación final sobre test set con el mejor modelo ---
    test_metrics = _final_evaluation(
        model=model,
        best_model_path=best_model_path,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        output_dir=output_dir,
    )

    return test_metrics
