"""
test_train.py

Tests unitarios de las funciones de entrenamiento en el paquete src/train/.
"""

import json

import numpy
import pandas
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline.constants import RANDOM_SEED, TOTAL_KMER_DIM
from mlp_model import AMRMLP
from train import (
    collect_predictions,
    compute_metrics,
    detect_device,
    evaluate,
    find_optimal_threshold,
    set_seed,
    train,
    train_epoch,
)


_N_ANTIBIOTICS = 5
_BATCH_SIZE = 8
_N_SAMPLES = 24  # Divisible por batch_size


@pytest.fixture()
def device():
    """Dispositivo CPU para tests (determinista y portátil)."""
    return torch.device("cpu")


@pytest.fixture()
def model():
    """Modelo MLP pequeño para tests."""
    set_seed(RANDOM_SEED)
    return AMRMLP(n_antibiotics=_N_ANTIBIOTICS)


@pytest.fixture()
def criterion(device):
    """Función de pérdida BCEWithLogitsLoss sin pos_weight."""
    return torch.nn.BCEWithLogitsLoss()


@pytest.fixture()
def synthetic_loader():
    """DataLoader sintético que imita la interfaz de AMRDataset."""
    genomes = torch.randn(_N_SAMPLES, TOTAL_KMER_DIM)
    ab_idxs = torch.randint(0, _N_ANTIBIOTICS, (_N_SAMPLES,))
    labels = torch.randint(0, 2, (_N_SAMPLES,)).float()
    dataset = TensorDataset(genomes, ab_idxs, labels)
    return DataLoader(dataset, batch_size=_BATCH_SIZE)


class TestSetSeed:
    """Tests de reproducibilidad con set_seed."""

    def test_torch_deterministic(self):
        """Dos inicializaciones con la misma semilla producen pesos idénticos."""
        set_seed(RANDOM_SEED)
        m1 = AMRMLP(n_antibiotics=_N_ANTIBIOTICS)

        set_seed(RANDOM_SEED)
        m2 = AMRMLP(n_antibiotics=_N_ANTIBIOTICS)

        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)

    def test_numpy_deterministic(self):
        """numpy.random produce la misma secuencia tras set_seed."""
        set_seed(RANDOM_SEED)
        a = numpy.random.rand(10)

        set_seed(RANDOM_SEED)
        b = numpy.random.rand(10)

        numpy.testing.assert_array_equal(a, b)


class TestTrainOneEpoch:
    """Tests de train_epoch."""

    def test_returns_float(self, model, synthetic_loader, criterion, device):
        """La función retorna un float (loss promedio)."""
        optimizer = torch.optim.Adam(model.parameters())
        loss = train_epoch(model, synthetic_loader, optimizer, criterion, device)
        assert isinstance(loss, float)

    def test_loss_is_finite(self, model, synthetic_loader, criterion, device):
        """La loss es un número finito (no NaN ni inf)."""
        optimizer = torch.optim.Adam(model.parameters())
        loss = train_epoch(model, synthetic_loader, optimizer, criterion, device)
        assert numpy.isfinite(loss)

    def test_weights_change(self, model, synthetic_loader, criterion, device):
        """Después de una época, los pesos del modelo han cambiado."""
        optimizer = torch.optim.Adam(model.parameters())
        weights_before = [p.clone() for p in model.parameters()]

        train_epoch(model, synthetic_loader, optimizer, criterion, device)

        changed = any(
            not torch.equal(before, after)
            for before, after in zip(weights_before, model.parameters())
        )
        assert changed


class TestCollectPredictions:
    """Tests de collect_predictions."""

    def test_returns_three_elements(self, model, synthetic_loader, criterion, device):
        """Retorna tupla de (probabilities, targets, loss)."""
        result = collect_predictions(model, synthetic_loader, criterion, device)
        assert len(result) == 3

    def test_probabilities_in_zero_one(self, model, synthetic_loader, criterion, device):
        """Las probabilidades están en [0, 1] (sigmoid aplicado)."""
        probabilities, _, _ = collect_predictions(model, synthetic_loader, criterion, device)
        assert probabilities.min() >= 0.0
        assert probabilities.max() <= 1.0

    def test_targets_are_binary(self, model, synthetic_loader, criterion, device):
        """Los targets son 0.0 o 1.0."""
        _, targets, _ = collect_predictions(model, synthetic_loader, criterion, device)
        assert set(numpy.unique(targets)).issubset({0.0, 1.0})

    def test_loss_is_finite(self, model, synthetic_loader, criterion, device):
        """La loss es un número finito."""
        _, _, loss = collect_predictions(model, synthetic_loader, criterion, device)
        assert numpy.isfinite(loss)

    def test_no_weight_change(self, model, synthetic_loader, criterion, device):
        """collect_predictions() no modifica los pesos del modelo."""
        weights_before = [p.clone() for p in model.parameters()]
        collect_predictions(model, synthetic_loader, criterion, device)
        for before, after in zip(weights_before, model.parameters()):
            assert torch.equal(before, after)


class TestComputeMetrics:
    """Tests de compute_metrics (numpy puro, sin modelo)."""

    def test_perfect_predictions(self):
        """Métricas perfectas cuando las predicciones son correctas."""
        targets = numpy.array([0.0, 0.0, 1.0, 1.0])
        probabilities = numpy.array([0.1, 0.2, 0.8, 0.9])
        metrics = compute_metrics(targets, probabilities, loss=0.1, threshold=0.5)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0

    def test_returns_expected_keys(self):
        """El dict contiene las 6 claves de métricas + loss."""
        targets = numpy.array([0.0, 1.0, 0.0, 1.0])
        probabilities = numpy.array([0.3, 0.7, 0.4, 0.6])
        metrics = compute_metrics(targets, probabilities, loss=0.5)
        expected = {"loss", "accuracy", "precision", "recall", "f1", "auc_roc"}
        assert set(metrics.keys()) == expected


class TestFindOptimalThreshold:
    """Tests de find_optimal_threshold."""

    def test_perfect_separation(self):
        """Con clases perfectamente separadas, el umbral está entre ellas."""
        targets = numpy.array([0.0, 0.0, 1.0, 1.0])
        probabilities = numpy.array([0.1, 0.2, 0.8, 0.9])
        threshold = find_optimal_threshold(targets, probabilities)
        assert 0.2 <= threshold <= 0.9

    def test_returns_float_in_range(self):
        """El umbral es un float en [0, 1]."""
        targets = numpy.array([0.0, 1.0, 0.0, 1.0])
        probabilities = numpy.array([0.3, 0.7, 0.6, 0.4])
        threshold = find_optimal_threshold(targets, probabilities)
        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0


class TestEvaluate:
    """Tests de evaluate (composición de las tres funciones)."""

    def test_returns_expected_keys(self, model, synthetic_loader, criterion, device):
        """El dict de métricas contiene todas las claves esperadas."""
        metrics = evaluate(model, synthetic_loader, criterion, device)
        expected = {"loss", "accuracy", "precision", "recall", "f1", "auc_roc", "optimal_threshold"}
        assert set(metrics.keys()) == expected

    def test_metrics_in_valid_range(self, model, synthetic_loader, criterion, device):
        """Todas las métricas están en rangos válidos."""
        metrics = evaluate(model, synthetic_loader, criterion, device)
        assert metrics["loss"] >= 0
        for key in ("accuracy", "precision", "recall", "f1", "auc_roc"):
            assert 0 <= metrics[key] <= 1, f"{key}={metrics[key]} fuera de [0, 1]"
        assert 0 <= metrics["optimal_threshold"] <= 1


class TestTrain:
    """Tests del orquestador train()."""

    def test_produces_outputs(self, model, synthetic_loader, criterion, device, tmp_path):
        """train() genera los 4 archivos de salida esperados."""
        metrics = train(
            model=model,
            train_loader=synthetic_loader,
            val_loader=synthetic_loader,
            test_loader=synthetic_loader,
            criterion=criterion,
            device=device,
            lr=0.01,
            epochs=3,
            patience=5,
            output_dir=tmp_path,
        )

        assert (tmp_path / "best_model.pt").exists()
        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "history.csv").exists()
        assert (tmp_path / "history.png").exists()

    def test_metrics_json_valid(self, model, synthetic_loader, criterion, device, tmp_path):
        """metrics.json contiene JSON válido con las claves esperadas."""
        train(
            model=model,
            train_loader=synthetic_loader,
            val_loader=synthetic_loader,
            test_loader=synthetic_loader,
            criterion=criterion,
            device=device,
            epochs=2,
            patience=5,
            output_dir=tmp_path,
        )

        data = json.loads((tmp_path / "metrics.json").read_text())
        assert "f1" in data
        assert "recall" in data
        assert "threshold_used" in data

    def test_history_csv_has_rows(self, model, synthetic_loader, criterion, device, tmp_path):
        """history.csv tiene al menos una fila por época ejecutada."""
        n_epochs = 3
        train(
            model=model,
            train_loader=synthetic_loader,
            val_loader=synthetic_loader,
            test_loader=synthetic_loader,
            criterion=criterion,
            device=device,
            epochs=n_epochs,
            patience=n_epochs + 1,
            output_dir=tmp_path,
        )

        df = pandas.read_csv(tmp_path / "history.csv")
        assert len(df) == n_epochs
        assert "train_loss" in df.columns
        assert "val_f1" in df.columns

    def test_early_stopping(self, device, tmp_path):
        """Con patience=1 y un modelo que no mejora, se detiene antes de max epochs."""
        set_seed(RANDOM_SEED)
        model = AMRMLP(n_antibiotics=_N_ANTIBIOTICS)

        # Dataset donde todas las labels son 0 — el modelo converge rápido
        # y luego val_loss deja de mejorar
        genomes = torch.randn(_N_SAMPLES, TOTAL_KMER_DIM)
        ab_idxs = torch.zeros(_N_SAMPLES, dtype=torch.long)
        labels = torch.zeros(_N_SAMPLES)
        loader = DataLoader(TensorDataset(genomes, ab_idxs, labels), batch_size=_BATCH_SIZE)

        criterion = torch.nn.BCEWithLogitsLoss()

        train(
            model=model,
            train_loader=loader,
            val_loader=loader,
            test_loader=loader,
            criterion=criterion,
            device=device,
            epochs=100,
            patience=1,
            output_dir=tmp_path,
        )

        df = pandas.read_csv(tmp_path / "history.csv")
        assert len(df) < 100, "Early stopping debería detener antes de 100 épocas"


class TestDetectDevice:
    """Tests de detect_device."""

    def test_returns_torch_device(self):
        """Retorna un objeto torch.device válido."""
        device = detect_device()
        assert isinstance(device, torch.device)
