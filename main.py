"""
main.py

Punto de entrada CLI del proyecto de predicción de AMR.

Comandos:
    download-amr              Descarga etiquetas AMR de BV-BRC para organismos ESKAPE
    download-genomes          Descarga genomas FASTA para los genome_id del CSV de etiquetas
    eda                       Análisis exploratorio del dataset de etiquetas AMR
    export-contradictions-cmd Exporta pares con etiquetas contradictorias a CSV
    prepare-data              Preprocesa datos: limpia, extrae k-meros, split, normaliza
    prepare-tokens            Extrae secuencias de tokens para el modelo Token BiGRU
    train-mlp                 Entrena el MLP sobre los datos preprocesados
    train-bigru               Entrena la BiGRU + Attention sobre los datos preprocesados
    train-token-bigru         Entrena la Token BiGRU (deep NN con tokens)
    train-multi-bigru         Entrena la Multi-Stream BiGRU (arquitectura experta por k)

Uso:
    uv run python main.py --help
    uv run python main.py download-amr
    uv run python main.py eda --labels data/processed/amr_labels.csv
    uv run python main.py prepare-tokens
    uv run python main.py train-token-bigru
    uv run python main.py train-mlp
"""

import json
import logging
from pathlib import Path

import pandas
import torch
import typer
from torch.utils.data import DataLoader

from bvbrc import download_multiple_genomes_fasta, fetch_amr_labels
from data_pipeline import run_pipeline, extract_and_save_tokens
from data_pipeline.constants import (
    RANDOM_SEED,
    TOKEN_KMER_K,
    TOKEN_MAX_LEN,
)
from models.mlp.dataset import MLPDataset
from models.mlp.model import AMRMLP
from models.bigru.dataset import BiGRUDataset
from models.bigru.model import AMRBiGRU
from models.multi_bigru.dataset import MultiBiGRUDataset
from models.multi_bigru.model import AMRMultiBiGRU
from models.token_bigru.dataset import TokenBiGRUDataset
from models.token_bigru.model import AMRTokenBiGRU
from eda import export_contradictions, run_eda
from train import detect_device, set_seed, train as run_training


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = typer.Typer(
    help="Herramientas CLI para el proyecto de predicción de resistencia antimicrobiana.",
    no_args_is_help=True,
)


def _sample_genome_ids(amr_labels: pandas.DataFrame, n_per_species: int) -> list[str]:
    """Selecciona hasta n genomas por especie, estratificados por fenotipo mayoritario."""
    amr_dedup = amr_labels.drop_duplicates(subset=["genome_id", "antibiotic"])
    sample_ids = []

    for _, group in amr_dedup.groupby("taxon_id"):
        genome_phenotype = (
            group.groupby("genome_id")["resistant_phenotype"]
            .agg(lambda x: x.value_counts().index[0])
            .reset_index()
        )
        n_each = n_per_species // 2
        resistant = genome_phenotype[genome_phenotype["resistant_phenotype"] == "Resistant"]["genome_id"]
        susceptible = genome_phenotype[genome_phenotype["resistant_phenotype"] == "Susceptible"]["genome_id"]

        sample_ids.extend(resistant.sample(min(n_each, len(resistant)), random_state=42).tolist())
        sample_ids.extend(susceptible.sample(min(n_each, len(susceptible)), random_state=42).tolist())

    return [str(gid) for gid in sample_ids]


@app.command(help="Descarga etiquetas AMR (Resistant/Susceptible) de BV-BRC para todos los organismos ESKAPE y las guarda como CSV.")
def download_amr(
    output: Path = typer.Option(
        Path("data/processed/amr_labels.csv"),
        help="Ruta donde guardar el CSV de etiquetas AMR.",
    ),
):
    """
    Descarga etiquetas AMR de BV-BRC para todos los organismos ESKAPE.

    Filtra por evidencia de laboratorio y fenotipos binarios (Resistant/Susceptible).
    Guarda el resultado como CSV en la ruta indicada.
    """
    typer.echo(f"Descargando etiquetas AMR → {output}")
    fetch_amr_labels(output_path=output)
    typer.echo(f"Listo. Etiquetas guardadas en: {output}")


@app.command(help="Descarga archivos FASTA de los genomas listados en el CSV de etiquetas. Usa --sample-per-species para una muestra estratificada por especie y fenotipo.")
def download_genomes(
    labels: Path = typer.Option(
        Path("data/processed/amr_labels.csv"),
        help="Ruta al CSV de etiquetas AMR (fuente de genome_id).",
    ),
    output_dir: Path = typer.Option(
        Path("data/raw/fasta"),
        help="Directorio donde guardar los archivos FASTA.",
    ),
    sample_per_species: int = typer.Option(
        None,
        help="Si se indica, descarga como máximo N genomas por especie, estratificados por fenotipo (mitad Resistant, mitad Susceptible). Si no se indica, descarga todos.",
    ),
):
    """
    Descarga genomas FASTA para los genome_id presentes en el CSV de etiquetas.

    Solo descarga los genomas que tengan al menos una etiqueta AMR válida.
    Omite genomas cuyo archivo .fna ya exista en output_dir.
    Usa --sample-per-species para descargar una muestra estratificada por especie
    (mitad Resistant, mitad Susceptible), útil para EDA sin descargar el dataset completo.
    """
    if not labels.exists():
        typer.echo(f"Error: no se encontró el archivo de etiquetas: {labels}", err=True)
        raise typer.Exit(code=1)

    amr_labels = pandas.read_csv(labels, dtype={"genome_id": str})

    if sample_per_species is not None:
        genome_ids = _sample_genome_ids(amr_labels, sample_per_species)
        typer.echo(f"Modo muestra: {sample_per_species} genomas/especie → {len(genome_ids)} genome IDs seleccionados")
    else:
        genome_ids = amr_labels["genome_id"].astype(str).unique().tolist()

    typer.echo(f"Genome IDs a descargar: {len(genome_ids)}")
    typer.echo(f"Destino: {output_dir}")

    results = download_multiple_genomes_fasta(
        genome_ids=genome_ids,
        output_directory=output_dir,
    )

    typer.echo(f"Descarga finalizada. Exitosos: {len(results)}/{len(genome_ids)}")


@app.command(help="Análisis exploratorio del dataset: distribución por especie, balance de clases, ranking de antibióticos, outliers, baseline benchmark y calidad genómica.")
def eda(
    labels: Path = typer.Option(
        Path("data/processed/amr_labels.csv"),
        help="Ruta al CSV de etiquetas AMR.",
    ),
    top_n_antibiotics: int = typer.Option(
        20,
        help="Número de antibióticos a mostrar en el ranking.",
    ),
    genomes_dir: Path = typer.Option(
        ...,
        help="Directorio con archivos .fna para análisis genómico.",
    ),
):
    """
    Análisis exploratorio del dataset de etiquetas AMR.

    Muestra: resumen general, distribución por especie, balance de clases,
    ranking de antibióticos, calidad de datos, outliers, baseline benchmark
    y análisis de secuencias genómicas.
    """
    if not labels.exists():
        typer.echo(f"Error: no se encontró el archivo de etiquetas: {labels}", err=True)
        raise typer.Exit(code=1)

    run_eda(labels_path=labels, top_n=top_n_antibiotics, genomes_dir=genomes_dir)


@app.command(help="Exporta los pares (genome_id, antibiotic) con etiquetas contradictorias (Resistant y Susceptible en registros distintos) a un CSV para inspección.")
def export_contradictions_cmd(
    labels: Path = typer.Option(
        Path("data/processed/amr_labels.csv"),
        help="Ruta al CSV de etiquetas AMR.",
    ),
    output: Path = typer.Option(
        Path("data/processed/contradictory_labels.csv"),
        help="Ruta donde guardar el CSV de pares contradictorios.",
    ),
):
    """
    Exporta pares (genome_id, antibiotic) con etiquetas contradictorias a CSV.

    Un par es contradictorio cuando el mismo genoma fue testeado contra el mismo
    antibiótico y produjo resultados Resistant y Susceptible en registros distintos.
    """
    if not labels.exists():
        typer.echo(f"Error: no se encontró el archivo de etiquetas: {labels}", err=True)
        raise typer.Exit(code=1)

    n_pairs = export_contradictions(labels_path=labels, output_path=output)
    typer.echo(f"Pares contradictorios encontrados: {n_pairs}")
    typer.echo(f"Reporte guardado en: {output}")


@app.command(help="Pre-procesa los datos: limpia etiquetas, extrae k-meros, divide en train/val/test y normaliza features.")
def prepare_data(
    labels: Path = typer.Option(
        Path("data/processed/amr_labels.csv"),
        help="Ruta al CSV de etiquetas AMR.",
    ),
    fasta_dir: Path = typer.Option(
        Path("data/raw/fasta"),
        help="Directorio con archivos .fna de genomas.",
    ),
    output_dir: Path = typer.Option(
        Path("data/processed"),
        help="Directorio donde guardar los outputs del pipeline.",
    ),
    n_jobs: int = typer.Option(
        1,
        help="Número de procesos paralelos para extracción de k-meros. "
             "Usa -1 para el 80% de los CPUs disponibles.",
    ),
):
    """
    Ejecuta el pipeline completo de preprocesamiento:

    1. Elimina pares contradictorios y duplicados del CSV de etiquetas
    2. Filtra genomas por calidad (longitud mínima 0.5 Mb)
    3. Crea índice antibiótico → entero
    4. Divide genome_ids en train/val/test (70/15/15, estratificado)
    5. Extrae histogramas de k-meros (k=3,4,5) por genoma
    6. Normaliza con estadísticas del train set
    7. Guarda features (.npy), etiquetas limpias, splits e índice
    """
    if not labels.exists():
        typer.echo(f"Error: no se encontró el archivo de etiquetas: {labels}", err=True)
        raise typer.Exit(code=1)
    if not fasta_dir.is_dir():
        typer.echo(f"Error: no se encontró el directorio de genomas: {fasta_dir}", err=True)
        raise typer.Exit(code=1)

    run_pipeline(labels_path=labels, fasta_dir=fasta_dir, output_dir=output_dir, n_jobs=n_jobs)
    typer.echo("Pipeline completado.")


@app.command(help="Extrae secuencias de tokens de k-meros para el modelo Token BiGRU.")
def prepare_tokens(
    data_dir: Path = typer.Option(
        Path("data/processed"),
        help="Directorio con outputs del pipeline (splits.csv, etc.).",
    ),
    fasta_dir: Path = typer.Option(
        Path("data/raw/fasta"),
        help="Directorio con archivos .fna de genomas.",
    ),
    k: int = typer.Option(
        TOKEN_KMER_K,
        "--k",
        help="Tamaño del k-mero para tokenización.",
    ),
    max_len: int = typer.Option(
        TOKEN_MAX_LEN,
        "--max-len",
        help="Longitud máxima de la secuencia de tokens.",
    ),
    n_jobs: int = typer.Option(
        1,
        help="Número de procesos paralelos. Usa -1 para el 80% de los CPUs.",
    ),
):
    """
    Extrae secuencias de tokens de k-meros para el modelo AMRTokenBiGRU.

    Requiere haber ejecutado prepare-data previamente. Lee los genome_ids
    del splits.csv existente y genera archivos .npy en token_bigru/.
    """
    splits_path = data_dir / "splits.csv"
    if not splits_path.exists():
        typer.echo(f"Error: no se encontró splits.csv en {data_dir}. Ejecuta prepare-data primero.", err=True)
        raise typer.Exit(code=1)

    splits = pandas.read_csv(splits_path, dtype={"genome_id": str})
    genome_list = sorted(splits["genome_id"].unique())

    typer.echo(f"Extrayendo tokens para {len(genome_list)} genomas...")
    extract_and_save_tokens(
        genome_ids=genome_list,
        fasta_dir=fasta_dir,
        output_dir=data_dir,
        k=k,
        max_len=max_len,
        n_jobs=n_jobs,
    )
    typer.echo(f"Tokens guardados en: {data_dir}/token_bigru/")


@app.command(help="Entrena el MLP (shallow NN) sobre los datos preprocesados y evalúa sobre test set.")
def train_mlp(
    data_dir: Path = typer.Option(
        Path("data/processed"),
        help="Directorio con outputs del pipeline (splits.csv, mlp/, etc.).",
    ),
    output_dir: Path = typer.Option(
        Path("results/mlp"),
        help="Directorio donde guardar modelo, métricas y gráficas.",
    ),
    epochs: int = typer.Option(100, help="Número máximo de épocas."),
    batch_size: int = typer.Option(32, help="Tamaño del mini-batch."),
    lr: float = typer.Option(0.001, help="Tasa de aprendizaje para Adam."),
    patience: int = typer.Option(10, help="Épocas sin mejora para early stopping."),
):
    """
    Entrena el Perceptrón Multicapa (AMRMLP) para predicción de AMR.

    Carga los datos preprocesados, construye el modelo, ejecuta el loop
    de entrenamiento con early stopping, y guarda el mejor modelo junto
    con métricas y gráficas de convergencia en output-dir.
    """
    # Reproducibilidad
    set_seed(RANDOM_SEED)

    # Detectar dispositivo
    device = detect_device()
    typer.echo(f"Dispositivo: {device}")

    # Cargar datasets
    typer.echo("Cargando datos (MLP)...")
    train_ds = MLPDataset(data_dir, split="train")
    val_ds = MLPDataset(data_dir, split="val")
    test_ds = MLPDataset(data_dir, split="test")
    typer.echo(f"Muestras — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Modelo
    model = AMRMLP.from_antibiotic_index(str(data_dir / "antibiotic_index.csv"))
    typer.echo(f"Modelo: {sum(p.numel() for p in model.parameters())} parámetros")

    # Función de pérdida con pos_weight para desbalance de clases
    pos_weight = MLPDataset.load_pos_weight(data_dir)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device),
    )
    typer.echo(f"pos_weight: {pos_weight:.4f}")

    # Entrenar
    test_metrics = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        lr=lr,
        epochs=epochs,
        patience=patience,
        output_dir=output_dir,
    )

    typer.echo(f"\nResultados en test set:")
    typer.echo(f"  F1:      {test_metrics['f1']:.4f}")
    typer.echo(f"  Recall:  {test_metrics['recall']:.4f}")
    typer.echo(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
    typer.echo(f"  Umbral:  {test_metrics['threshold_used']:.4f}")
    typer.echo(f"\nGuardado en: {output_dir}")


@app.command(help="Entrena la BiGRU + Attention (deep NN) sobre los datos preprocesados y evalúa sobre test set.")
def train_bigru(
    data_dir: Path = typer.Option(
        Path("data/processed"),
        help="Directorio con outputs del pipeline (splits.csv, bigru/, etc.).",
    ),
    output_dir: Path = typer.Option(
        Path("results/bigru"),
        help="Directorio donde guardar modelo, métricas y gráficas.",
    ),
    epochs: int = typer.Option(100, help="Número máximo de épocas."),
    batch_size: int = typer.Option(32, help="Tamaño del mini-batch."),
    lr: float = typer.Option(0.001, help="Tasa de aprendizaje para Adam."),
    patience: int = typer.Option(10, help="Épocas sin mejora para early stopping."),
    pos_weight_scale: float = typer.Option(
        2.5,
        "--pos-weight-scale",
        help=(
            "Factor multiplicador del pos_weight base para sesgar hacia recall. "
            "Valores > 1 penalizan más los falsos negativos [King20]."
        ),
    ),
):
    """
    Entrena la Red Neuronal Recurrente (AMRBiGRU) para predicción de AMR.

    Implementa una arquitectura BiGRU con mecanismo de atención aditiva [Bahdanau15]
    basada en [Lugo21]. Carga la representación distribuida de k-meros (matrices 2D),
    ejecuta el loop de entrenamiento con gradient clipping [Pascanu13] y early stopping.
    """
    # Reproducibilidad [Haykin, §4.4]
    set_seed(RANDOM_SEED)

    # Detectar dispositivo
    device = detect_device()
    typer.echo(f"Dispositivo: {device}")

    # Cargar datasets con representación distribuida [Lugo21, p. 647]
    typer.echo("Cargando datos (BiGRU)...")
    train_ds = BiGRUDataset(data_dir, split="train")
    val_ds = BiGRUDataset(data_dir, split="val")
    test_ds = BiGRUDataset(data_dir, split="test")
    typer.echo(
        f"Muestras — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}"
    )

    # DataLoaders — [Goodfellow16, Cap. 8.1.3]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Modelo [Lugo21; Schuster97; Cho14; Bahdanau15]
    model = AMRBiGRU.from_antibiotic_index(str(data_dir / "antibiotic_index.csv"))
    typer.echo(f"Modelo: {sum(p.numel() for p in model.parameters())} parámetros")

    # Función de pérdida con pos_weight escalado para priorizar recall [Haykin, Cap. 1.4]
    # MEJORA1: pos_weight_scale > 1 aumenta la penalización de falsos negativos [King20].
    base_pos_weight = BiGRUDataset.load_pos_weight(data_dir)
    scaled_pos_weight = base_pos_weight * pos_weight_scale
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([scaled_pos_weight], device=device),
    )
    typer.echo(f"pos_weight base: {base_pos_weight:.4f} → escalado: {scaled_pos_weight:.4f}")

    # Guardar parámetros para trazabilidad
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "model_type": "bigru",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "patience": patience,
        "pos_weight_scale": pos_weight_scale,
        "max_grad_norm": 1.0,
        "device": str(device),
    }
    (output_dir / "params.json").write_text(json.dumps(params, indent=2))

    # Entrenar con Gradient Clipping [Pascanu13] para prevenir explosión de gradientes en RNNs
    test_metrics = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        lr=lr,
        epochs=epochs,
        patience=patience,
        output_dir=output_dir,
        max_grad_norm=1.0,
    )

    typer.echo(f"\nResultados en test set:")
    typer.echo(f"  F1:      {test_metrics['f1']:.4f}")
    typer.echo(f"  Recall:  {test_metrics['recall']:.4f}")
    typer.echo(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
    typer.echo(f"  Umbral:  {test_metrics['threshold_used']:.4f}")
    typer.echo(f"\nGuardado en: {output_dir}")


@app.command(help="Entrena el modelo Token BiGRU (deep NN con tokens) sobre los datos preprocesados.")
def train_token_bigru(
    data_dir: Path = typer.Option(
        Path("data/processed"),
        help="Directorio con outputs del pipeline (splits.csv, token_bigru/, etc.).",
    ),
    output_dir: Path = typer.Option(
        Path("results/token_bigru"),
        help="Directorio donde guardar modelo, métricas y gráficas.",
    ),
    epochs: int = typer.Option(100, help="Número máximo de épocas."),
    batch_size: int = typer.Option(32, help="Tamaño del mini-batch."),
    lr: float = typer.Option(0.0005, help="Tasa de aprendizaje para Adam."),
    patience: int = typer.Option(10, help="Épocas sin mejora para early stopping."),
    pos_weight_scale: float = typer.Option(
        1.5,
        "--pos-weight-scale",
        help="Factor multiplicador del pos_weight base [King20].",
    ),
    weight_decay: float = typer.Option(
        1e-4,
        "--weight-decay",
        help="Regularización L2 en Adam [Goodfellow16, Cap. 7].",
    ),
):
    """
    Entrena la arquitectura BiGRU + Attention con tokenización de k-meros.

    A diferencia del BiGRU base, este modelo procesa una secuencia real
    de tokens extraídos del genoma [Mikolov13; Cho14]. Utiliza un
    mecanismo de atención aditiva [Bahdanau15] para identificar las
    regiones genómicas más informativas para la resistencia.
    """
    set_seed(RANDOM_SEED)
    device = detect_device()
    typer.echo(f"Dispositivo: {device}")

    # Cargar datasets con model_type='token_bigru'
    typer.echo("Cargando datos (Token BiGRU)...")
    train_ds = TokenBiGRUDataset(data_dir, split="train")
    val_ds = TokenBiGRUDataset(data_dir, split="val")
    test_ds = TokenBiGRUDataset(data_dir, split="test")
    typer.echo(f"Muestras — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Modelo con embedding de k-meros
    model = AMRTokenBiGRU.from_antibiotic_index(str(data_dir / "antibiotic_index.csv"))
    typer.echo(f"Modelo: {sum(p.numel() for p in model.parameters())} parámetros")

    # Función de pérdida con penalización asimétrica [King20]
    base_pos_weight = TokenBiGRUDataset.load_pos_weight(data_dir)
    scaled_pos_weight = base_pos_weight * pos_weight_scale
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([scaled_pos_weight], device=device),
    )
    typer.echo(f"pos_weight base: {base_pos_weight:.4f} → escalado: {scaled_pos_weight:.4f}")
    typer.echo(f"weight_decay: {weight_decay:.1e}")

    # Trazabilidad
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "model_type": "token_bigru",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "patience": patience,
        "pos_weight_scale": pos_weight_scale,
        "weight_decay": weight_decay,
        "max_grad_norm": 1.0,
        "device": str(device),
    }
    (output_dir / "params.json").write_text(json.dumps(params, indent=2))

    # Entrenar con Gradient Clipping [Pascanu13] y L2 [Goodfellow16, Cap. 7]
    test_metrics = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        lr=lr,
        epochs=epochs,
        patience=patience,
        output_dir=output_dir,
        max_grad_norm=1.0,
        weight_decay=weight_decay,
    )

    typer.echo(f"\nResultados en test set:")
    typer.echo(f"  F1:      {test_metrics['f1']:.4f}")
    typer.echo(f"  Recall:  {test_metrics['recall']:.4f}")
    typer.echo(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
    typer.echo(f"  Umbral:  {test_metrics['threshold_used']:.4f}")
    typer.echo(f"\nGuardado en: {output_dir}")


@app.command(help="Entrena la Multi-Stream BiGRU (arquitectura experta por k) sobre los datos preprocesados.")
def train_multi_bigru(
    data_dir: Path = typer.Option(
        Path("data/processed"),
        help="Directorio con vectores MLP (reutilizados para segmentación).",
    ),
    output_dir: Path = typer.Option(
        Path("results/multi_bigru"),
        help="Directorio donde guardar modelo, métricas y gráficas.",
    ),
    epochs: int = typer.Option(100, help="Número máximo de épocas."),
    batch_size: int = typer.Option(32, help="Tamaño del mini-batch."),
    lr: float = typer.Option(0.001, help="Tasa de aprendizaje para Adam."),
    patience: int = typer.Option(10, help="Épocas sin mejora para early stopping."),
    pos_weight_scale: float = typer.Option(
        2.5,
        "--pos-weight-scale",
        help="Factor multiplicador del pos_weight base [King20].",
    ),
):
    """
    Entrena la Multi-Stream BiGRU para predicción de AMR.

    Procesa cada k-mero (3, 4, 5) con una BiGRU independiente para eliminar
    el ruido del padding [Ngiam11]. Reutiliza los vectores del MLP para
    eficiencia de almacenamiento.
    """
    set_seed(RANDOM_SEED)
    device = detect_device()
    typer.echo(f"Dispositivo: {device}")

    # Cargar datasets
    typer.echo("Cargando datos (Multi-Stream BiGRU)...")
    train_ds = MultiBiGRUDataset(data_dir, split="train")
    val_ds = MultiBiGRUDataset(data_dir, split="val")
    test_ds = MultiBiGRUDataset(data_dir, split="test")
    typer.echo(f"Muestras — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Modelo multi-stream
    model = AMRMultiBiGRU.from_antibiotic_index(str(data_dir / "antibiotic_index.csv"))
    typer.echo(f"Modelo: {sum(p.numel() for p in model.parameters())} parámetros")

    # Función de pérdida
    base_pos_weight = MultiBiGRUDataset.load_pos_weight(data_dir)
    scaled_pos_weight = base_pos_weight * pos_weight_scale
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([scaled_pos_weight], device=device),
    )
    typer.echo(f"pos_weight base: {base_pos_weight:.4f} → escalado: {scaled_pos_weight:.4f}")

    # Trazabilidad
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "model_type": "multi_bigru",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "patience": patience,
        "pos_weight_scale": pos_weight_scale,
        "max_grad_norm": 1.0,
        "device": str(device),
    }
    (output_dir / "params.json").write_text(json.dumps(params, indent=2))

    # Entrenar
    test_metrics = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        lr=lr,
        epochs=epochs,
        patience=patience,
        output_dir=output_dir,
        max_grad_norm=1.0,
    )

    typer.echo(f"\nResultados en test set:")
    typer.echo(f"  F1:      {test_metrics['f1']:.4f}")
    typer.echo(f"  Recall:  {test_metrics['recall']:.4f}")
    typer.echo(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
    typer.echo(f"  Umbral:  {test_metrics['threshold_used']:.4f}")
    typer.echo(f"\nGuardado en: {output_dir}")


if __name__ == "__main__":
    app()
