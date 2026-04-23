"""
Evaluación offline de umbrales por antibiótico sobre HierSet v1.

Carga el checkpoint ya entrenado, corre inferencia sobre val/test,
y compara el F1 con umbral global vs. umbrales por antibiótico.
No reentrena el modelo.

Uso:
    uv run python scripts/per_antibiotic_threshold.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy
import pandas
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

# Permitir importar desde src/ sin instalar el paquete
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.hier_set.dataset import HierSetDataset  # noqa: E402
from models.hier_set.model import AMRHierSet  # noqa: E402
from train.evaluate import find_optimal_threshold  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT = ROOT / "results" / "hier_set" / "best_model.pt"
MIN_SAMPLES = 30


def collect(model, loader, device):
    """Devuelve (probs, targets, ab_idx) como numpy arrays."""
    model.eval()
    probs_list, targets_list, ab_list = [], [], []
    with torch.no_grad():
        for genome, ab_idx, label in loader:
            genome = genome.to(device)
            ab_idx_dev = ab_idx.to(device)
            logits = model(genome, ab_idx_dev)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            probs_list.append(probs)
            targets_list.append(label.numpy().ravel())
            ab_list.append(ab_idx.numpy().ravel())
    return (
        numpy.concatenate(probs_list),
        numpy.concatenate(targets_list),
        numpy.concatenate(ab_list),
    )


def per_antibiotic_thresholds(targets, probs, ab_idx, global_threshold, min_samples):
    """Umbral óptimo por antibiótico; fallback al global si hay pocas muestras."""
    thresholds = {}
    summary = []
    for ab in numpy.unique(ab_idx):
        mask = ab_idx == ab
        t_ab, p_ab = targets[mask], probs[mask]
        n_pos = int(t_ab.sum())
        n_neg = int((1 - t_ab).sum())
        if n_pos < min_samples or n_neg < min_samples:
            thresholds[int(ab)] = float(global_threshold)
            summary.append((int(ab), len(t_ab), n_pos, n_neg, float(global_threshold), True))
        else:
            t = find_optimal_threshold(t_ab, p_ab)
            thresholds[int(ab)] = t
            summary.append((int(ab), len(t_ab), n_pos, n_neg, t, False))
    return thresholds, summary


def apply_per_ab(probs, ab_idx, thresholds, fallback):
    """Vectoriza la aplicación de un umbral distinto por muestra."""
    per_sample = numpy.array(
        [thresholds.get(int(idx), fallback) for idx in ab_idx]
    )
    return (probs >= per_sample).astype(int)


def metrics(targets, preds, probs, label):
    return {
        "label": label,
        "f1": f1_score(targets, preds, zero_division=0),
        "precision": precision_score(targets, preds, zero_division=0),
        "recall": recall_score(targets, preds, zero_division=0),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Cargando datasets...")
    val_ds = HierSetDataset(DATA_DIR, split="val")
    test_ds = HierSetDataset(DATA_DIR, split="test")
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    print("Cargando modelo...")
    model = AMRHierSet.from_antibiotic_index(
        str(DATA_DIR / "antibiotic_index.csv")
    ).to(device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state)

    print("Inferencia val...")
    val_probs, val_targets, val_ab = collect(model, val_loader, device)
    print("Inferencia test...")
    test_probs, test_targets, test_ab = collect(model, test_loader, device)

    # --- Umbral global sobre val ---
    global_t = find_optimal_threshold(val_targets, val_probs)
    print(f"\nUmbral global (val): {global_t:.4f}")

    # --- Umbrales por antibiótico sobre val ---
    thresholds, summary = per_antibiotic_thresholds(
        val_targets, val_probs, val_ab, global_t, MIN_SAMPLES
    )
    n_tuned = sum(1 for row in summary if not row[5])
    n_fallback = len(summary) - n_tuned
    print(f"Antibióticos tuneados: {n_tuned} / {len(summary)} (fallback: {n_fallback})")

    # --- Métricas en test ---
    preds_global = (test_probs >= global_t).astype(int)
    preds_per_ab = apply_per_ab(test_probs, test_ab, thresholds, global_t)

    results = [
        metrics(test_targets, preds_global, test_probs, "global"),
        metrics(test_targets, preds_per_ab, test_probs, "per_antibiotic"),
    ]

    print("\n== TEST SET ==")
    print(f"{'Esquema':<20} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    for r in results:
        print(
            f"{r['label']:<20} {r['f1']:.4f}   {r['precision']:.4f}     {r['recall']:.4f}"
        )

    delta_f1 = results[1]["f1"] - results[0]["f1"]
    print(f"\nΔF1 (per_ab - global): {delta_f1:+.4f}")

    # --- Breakdown por antibiótico (opcional, útil para entender) ---
    ab_index = pandas.read_csv(DATA_DIR / "antibiotic_index.csv")
    idx_to_name = dict(zip(ab_index["index"], ab_index["antibiotic"]))

    print("\nTop 10 antibióticos con mayor cambio de umbral vs global:")
    rows = []
    for ab, n, n_pos, n_neg, t, fb in summary:
        if fb:
            continue
        rows.append((idx_to_name.get(ab, f"idx={ab}"), n, n_pos, t, t - global_t))
    rows.sort(key=lambda x: abs(x[4]), reverse=True)
    print(f"{'antibiotic':<25} {'n':>6} {'n_pos':>6} {'thresh':>8} {'Δ':>8}")
    for name, n, n_pos, t, delta in rows[:10]:
        print(f"{name:<25} {n:>6} {n_pos:>6} {t:>8.4f} {delta:>+8.4f}")

    # Guardar resultados para reproducibilidad
    output = {
        "global_threshold": float(global_t),
        "per_antibiotic_thresholds": {str(k): v for k, v in thresholds.items()},
        "min_samples": MIN_SAMPLES,
        "n_tuned": n_tuned,
        "n_fallback": n_fallback,
        "test_metrics": {r["label"]: r for r in results},
        "delta_f1": delta_f1,
    }
    out_path = ROOT / "results" / "hier_set" / "per_antibiotic_thresholds.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nGuardado en: {out_path}")


if __name__ == "__main__":
    main()
