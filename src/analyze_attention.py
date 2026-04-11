"""
analyze_attention.py — Análisis de interpretabilidad del mecanismo de atención.

Calcula la energía promedio asignada por el mecanismo de atención a las distintas
regiones de la entrada (k=3, 4, 5) para entender qué longitudes de k-meros son
más informativas para el modelo.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import torch
import typer
from torch.utils.data import DataLoader

from models.bigru.model import AMRBiGRU
from models.bigru.dataset import BiGRUDataset
from train import detect_device

app = typer.Typer()
logger = logging.getLogger(__name__)

@app.command()
def analyze_attention(
    data_dir: Path = typer.Option("data/processed", help="Directorio de datos."),
    model_path: Path = typer.Option("results/bigru/best_model.pt", help="Ruta al modelo."),
    output_path: Path = typer.Option("results/bigru/attention_analysis.png", help="Ruta de salida."),
):
    data_dir = Path(data_dir)
    model_path = Path(model_path)
    device = detect_device()

    # 1. Cargar modelo
    logger.info(f"Cargando modelo desde {model_path}...")
    model = AMRBiGRU.from_antibiotic_index(str(data_dir / "antibiotic_index.csv"))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 2. Cargar datos de prueba
    logger.info("Cargando dataset de prueba...")
    test_ds = AMRDataset(data_dir, split="test", model_type="bigru")
    loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # 3. Recolectar pesos de atención
    # Shape esperado de pesos: [batch, 1024]
    all_weights = []
    
    logger.info(f"Analizando {len(test_ds)} muestras...")
    with torch.no_grad():
        for genome, ab_idx, _ in loader:
            genome = genome.to(device)
            ab_idx = ab_idx.to(device)
            
            # Forward pass (los pesos se guardan internamente en model._attention_weights)
            model(genome, ab_idx)
            all_weights.append(model._attention_weights.cpu().numpy())

    # Concatenar todos los pesos: [N_test, 1024]
    weights_matrix = numpy.concatenate(all_weights, axis=0)
    # Media sobre todas las muestras: [1024]
    mean_weights = weights_matrix.mean(axis=0)

    # 4. Graficar
    plt.figure(figsize=(12, 6))
    
    # Definir regiones de k [Lugo21, p. 647]
    # k=3: 0-63, k=4: 0-255, k=5: 0-1023 (todos paddeados a 1024)
    # Dado que todos los histogramas están apilados, el eje X (1024 timesteps)
    # representa la posición en el histograma respectivo.
    
    plt.plot(mean_weights, label="Peso de Atención Promedio", color="blue", alpha=0.7)
    
    # Resaltar límites teóricos de información
    plt.axvline(x=64, color="red", linestyle="--", label="Fin k=3 (64)")
    plt.axvline(x=256, color="green", linestyle="--", label="Fin k=4 (256)")
    
    plt.fill_between(range(64), mean_weights[:64], color="red", alpha=0.1, label="Zona Informativa k=3")
    plt.fill_between(range(64, 256), mean_weights[64:256], color="green", alpha=0.1, label="Zona Informativa k=4")
    plt.fill_between(range(256, 1024), mean_weights[256:1024], color="blue", alpha=0.1, label="Zona Informativa k=5")

    plt.title("Importancia Relativa por Posición del Histograma (Atención Bahdanau)")
    plt.xlabel("Índice del Histograma (0-1023)")
    plt.ylabel("Peso de Atención (Promedio)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Análisis guardado en: {output_path}")

    # 5. Calcular importancia por zona
    # El mecanismo de atención suma 1.0 sobre los 1024 timesteps.
    # Queremos ver cuánta de esa "probabilidad" se concentra en cada región.
    total_energy = mean_weights.sum()
    energy_k3 = mean_weights[:64].sum() / total_energy
    energy_k4 = mean_weights[64:256].sum() / total_energy
    energy_k5 = mean_weights[256:].sum() / total_energy

    logger.info("\nDistribución de la 'Energía' de Atención:")
    logger.info(f"  Zona 0-63   (Activa en k=3, 4, 5): {energy_k3:.2%}")
    logger.info(f"  Zona 64-255 (Activa en k=4, 5):    {energy_k4:.2%}")
    logger.info(f"  Zona 256-1023 (Solo k=5):          {energy_k5:.2%}")
    
    # Nota interpretativa: Como los k están apilados, la zona 0-63 contiene 
    # información de k=3 Y de las cabeceras de k=4 y k=5. 
    # La zona 256-1023 es "pureza k=5".


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()
