"""
main.py

Punto de entrada CLI del proyecto de predicción de AMR.

Comandos:
    download-amr              Descarga etiquetas AMR de BV-BRC (ESKAPE por defecto, all-taxa opcional)
    download-genomes          Descarga genomas FASTA para los genome_id del CSV de etiquetas
    eda                       Análisis exploratorio del dataset de etiquetas AMR
    eda-expanded              EDA reproducible del dataset expandido all-taxa
    export-contradictions-cmd Exporta pares con etiquetas contradictorias a CSV
    prepare-labels-for-download Limpia labels antes de descargar FASTA expandido
    prepare-data              Preprocesa datos: limpia, extrae k-meros, split, normaliza
    prepare-tokens            Extrae secuencias de tokens para el modelo Token BiGRU
    prepare-hier              Extrae histogramas segmentados para el modelo Hierarchical BiGRU
    train-mlp                 Entrena el MLP sobre los datos preprocesados
    train-bigru               Entrena la BiGRU + Attention sobre los datos preprocesados
    train-token-bigru         Entrena la Token BiGRU (deep NN con tokens)
    train-multi-bigru         Entrena la Multi-Stream BiGRU (arquitectura experta por k)
    train-hier-bigru          Entrena el Hierarchical BiGRU sobre histogramas segmentados
    train-hier-set            Entrena el Hierarchical Set Encoder (sin dependencias secuenciales)
    evaluate-hier-set-checkpoint Evalúa un checkpoint HierSet sin reentrenar

Uso:
    uv run python main.py --help
    uv run python main.py download-amr
    uv run python main.py eda --labels data/processed/amr_labels.csv
    uv run python main.py prepare-tokens
    uv run python main.py prepare-hier
    uv run python main.py train-token-bigru
    uv run python main.py train-hier-bigru
    uv run python main.py train-mlp
"""

import json
import logging
from pathlib import Path

import pandas
import torch
import typer
from torch.utils.data import DataLoader

from bvbrc import ESKAPE_TAXON_IDS, download_multiple_genomes_fasta, fetch_amr_labels
from data_pipeline import (
    run_pipeline,
    prepare_labels_for_download as run_prepare_labels_for_download,
    extract_and_save_tokens,
    extract_and_save_hier,
    extract_and_save_hier_multi,
)
from data_pipeline.constants import (
    RANDOM_SEED,
    MIN_RECORDS_PER_ANTIBIOTIC,
    TOKEN_KMER_K,
    TOKEN_MAX_LEN,
    HIER_KMER_K,
    HIER_N_SEGMENTS,
)
from models.mlp.dataset import MLPDataset
from models.mlp.model import AMRMLP
from models.bigru.dataset import BiGRUDataset
from models.bigru.model import AMRBiGRU
from models.multi_bigru.dataset import MultiBiGRUDataset
from models.multi_bigru.model import AMRMultiBiGRU
from models.token_bigru.dataset import TokenBiGRUDataset
from models.token_bigru.model import AMRTokenBiGRU
from models.hier_bigru.dataset import HierBiGRUDataset
from models.hier_bigru.model import AMRHierBiGRU
from models.hier_set.dataset import HierSetDataset
from models.hier_set.model import AMRHierSet
from models.hier_set_v2.dataset import HierSetV2Dataset
from models.hier_set_v2.model import AMRHierSetV2
from eda import export_contradictions, run_eda, run_expanded_eda
from train import (
    collect_predictions,
    compute_metrics,
    detect_device,
    find_optimal_threshold,
    set_seed,
    train as run_training,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = typer.Typer(
    help="Herramientas CLI para el proyecto de predicción de resistencia antimicrobiana.",
    no_args_is_help=True,
)


def _resolve_threshold(checkpoint: Path, threshold: float | None) -> float:
    """Obtiene el umbral explícito o el guardado junto al checkpoint."""
    if threshold is not None:
        return threshold
    metrics_path = checkpoint.parent / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        if "threshold_used" in metrics:
            return float(metrics["threshold_used"])
    return 0.5


def _compute_metrics_safe(
    targets,
    probabilities,
    threshold: float,
) -> dict:
    """Calcula métricas y tolera grupos con una sola clase para AUC."""
    try:
        return compute_metrics(targets, probabilities, loss=float("nan"), threshold=threshold)
    except ValueError:
        predictions = (probabilities >= threshold).astype(int)
        tp = int(((predictions == 1) & (targets == 1)).sum())
        tn = int(((predictions == 0) & (targets == 0)).sum())
        fp = int(((predictions == 1) & (targets == 0)).sum())
        fn = int(((predictions == 0) & (targets == 1)).sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        return {
            "loss": float("nan"),
            "accuracy": (tp + tn) / len(targets) if len(targets) > 0 else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_roc": float("nan"),
        }


def _save_group_metrics(
    predictions: pandas.DataFrame,
    group_column: str,
    threshold: float,
    output_path: Path,
) -> None:
    """Guarda métricas por grupo para análisis de heterogeneidad."""
    rows = []
    for group_value, group in predictions.groupby(group_column, dropna=True):
        targets = group["target"].to_numpy()
        probabilities = group["probability"].to_numpy()
        metrics = _compute_metrics_safe(targets, probabilities, threshold)
        metrics.pop("loss", None)
        metrics.update(
            {
                group_column: group_value,
                "n_samples": len(group),
                "n_resistant": int(targets.sum()),
                "n_susceptible": int(len(targets) - targets.sum()),
            }
        )
        rows.append(metrics)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pandas.DataFrame(rows).sort_values("n_samples", ascending=False).to_csv(
        output_path,
        index=False,
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


@app.command(help="Descarga etiquetas AMR (Resistant/Susceptible) de BV-BRC y las guarda como CSV.")
def download_amr(
    output: Path = typer.Option(
        Path("data/processed/amr_labels.csv"),
        help="Ruta donde guardar el CSV de etiquetas AMR.",
    ),
    all_taxa: bool = typer.Option(
        False,
        "--all-taxa",
        help="Descarga etiquetas AMR de todos los taxones, sin limitarse a ESKAPE.",
    ),
    exclude_eskape: bool = typer.Option(
        False,
        "--exclude-eskape",
        help="Excluye taxones ESKAPE del resultado. Requiere --all-taxa.",
    ),
    typing_method: str | None = typer.Option(
        None,
        "--typing-method",
        help="Filtra por método de laboratorio, por ejemplo 'Broth dilution'.",
    ),
):
    """
    Descarga etiquetas AMR de BV-BRC.

    Por defecto usa el alcance ESKAPE histórico del proyecto. Con --all-taxa
    descarga todos los taxones con etiquetas AMR válidas.
    Filtra por evidencia de laboratorio y fenotipos binarios Resistant/Susceptible.
    Guarda el resultado como CSV en la ruta indicada.
    """
    if exclude_eskape and not all_taxa:
        typer.echo("Error: --exclude-eskape requiere --all-taxa.", err=True)
        raise typer.Exit(code=1)

    scope = "todos los taxones" if all_taxa else "organismos ESKAPE"
    if exclude_eskape:
        scope = "taxones no-ESKAPE"

    typer.echo(f"Descargando etiquetas AMR ({scope}) → {output}")
    exclude_taxon_ids = list(ESKAPE_TAXON_IDS.values()) if exclude_eskape else None
    fetch_amr_labels(
        output_path=output,
        all_taxa=all_taxa,
        exclude_taxon_ids=exclude_taxon_ids,
        typing_method=typing_method,
    )
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
    n_jobs: int = typer.Option(
        1,
        help="Número de workers para descargar FASTA. Usa -1 para el 80% de los CPUs.",
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
        n_jobs=n_jobs,
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


@app.command(help="EDA reproducible del dataset AMR expandido all-taxa.")
def eda_expanded(
    raw_labels: Path = typer.Option(
        Path("data/expanded/raw/amr_labels_all_taxa.csv"),
        help="CSV raw combinado all-taxa.",
    ),
    processed_dir: Path = typer.Option(
        Path("data/expanded/processed"),
        help="Directorio con labels_for_download.csv, cleaned_labels.csv y splits.csv.",
    ),
    non_eskape_labels: Path | None = typer.Option(
        Path("data/expanded/raw/amr_labels_non_eskape.csv"),
        help="CSV raw no-ESKAPE. Si no existe, se omite.",
    ),
    top_n_taxa: int = typer.Option(
        10,
        help="Número de taxones a mostrar en el ranking.",
    ),
    top_n_antibiotics: int = typer.Option(
        10,
        help="Número de antibióticos a mostrar en el ranking.",
    ),
):
    """Ejecuta el EDA del dataset expandido usando artefactos procesados."""
    if not raw_labels.exists():
        typer.echo(f"Error: no se encontró el CSV raw: {raw_labels}", err=True)
        raise typer.Exit(code=1)
    if not processed_dir.is_dir():
        typer.echo(f"Error: no se encontró el directorio procesado: {processed_dir}", err=True)
        raise typer.Exit(code=1)
    for filename in ["labels_for_download.csv", "cleaned_labels.csv", "splits.csv"]:
        path = processed_dir / filename
        if not path.exists():
            typer.echo(f"Error: falta {path}", err=True)
            raise typer.Exit(code=1)

    run_expanded_eda(
        raw_labels_path=raw_labels,
        processed_dir=processed_dir,
        non_eskape_labels_path=non_eskape_labels,
        top_n_taxa=top_n_taxa,
        top_n_antibiotics=top_n_antibiotics,
    )


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


@app.command(help="Limpia etiquetas AMR antes de descargar FASTA para el dataset expandido.")
def prepare_labels_for_download(
    labels: Path = typer.Option(
        Path("data/expanded/raw/amr_labels_all_taxa.csv"),
        help="Ruta al CSV crudo de etiquetas AMR expandidas.",
    ),
    output: Path = typer.Option(
        Path("data/expanded/processed/labels_for_download.csv"),
        help="Ruta donde guardar las etiquetas limpias para descargar FASTA.",
    ),
    min_records_per_antibiotic: int = typer.Option(
        MIN_RECORDS_PER_ANTIBIOTIC,
        "--min-records-per-antibiotic",
        help="Mínimo de registros útiles finales por antibiótico.",
    ),
):
    """
    Limpia etiquetas antes de la descarga masiva de FASTA.

    Reutiliza LabelCleaner: conserva Broth dilution, elimina contradicciones,
    deduplica pares genome_id-antibiotic y filtra antibióticos con baja frecuencia.
    """
    if not labels.exists():
        typer.echo(f"Error: no se encontró el archivo de etiquetas: {labels}", err=True)
        raise typer.Exit(code=1)
    if labels.resolve() == output.resolve():
        typer.echo("Error: --output debe ser distinto de --labels.", err=True)
        raise typer.Exit(code=1)

    cleaned = run_prepare_labels_for_download(
        labels_path=labels,
        output_path=output,
        min_records_per_antibiotic=min_records_per_antibiotic,
    )
    typer.echo(f"Etiquetas limpias guardadas en: {output}")
    typer.echo(f"  Registros:    {len(cleaned):,}")
    typer.echo(f"  Genomas:      {cleaned['genome_id'].nunique():,}")
    typer.echo(f"  Taxones:      {cleaned['taxon_id'].nunique():,}")
    typer.echo(f"  Antibióticos: {cleaned['antibiotic'].nunique():,}")


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
    locked_splits: Path | None = typer.Option(
        None,
        "--locked-splits",
        help="CSV con columnas genome_id y split para conservar particiones existentes.",
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
    if locked_splits is not None and not locked_splits.exists():
        typer.echo(f"Error: no se encontró el archivo de splits congelados: {locked_splits}", err=True)
        raise typer.Exit(code=1)

    run_pipeline(
        labels_path=labels,
        fasta_dir=fasta_dir,
        output_dir=output_dir,
        n_jobs=n_jobs,
        locked_splits_path=locked_splits,
    )
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


@app.command(help="Extrae histogramas segmentados para el modelo Hierarchical BiGRU.")
def prepare_hier(
    data_dir: Path = typer.Option(
        Path("data/processed"),
        help="Directorio con outputs del pipeline (splits.csv, etc.).",
    ),
    fasta_dir: Path = typer.Option(
        Path("data/raw/fasta"),
        help="Directorio con archivos .fna de genomas.",
    ),
    n_jobs: int = typer.Option(
        1,
        help="Número de procesos paralelos. Usa -1 para el 80% de los CPUs.",
    ),
):
    """
    Extrae histogramas segmentados (tiled histograms) para AMRHierBiGRU.

    Divide cada genoma en HIER_N_SEGMENTS segmentos y calcula el histograma de k=4
    para cada uno. Garantiza cobertura del 100% del genoma. Los resultados se guardan
    en data/processed/hier_bigru/ como archivos .npy.

    NOTA: si se cambia HIER_N_SEGMENTS en constants.py, este comando debe volver a
    ejecutarse — los .npy existentes quedarán con shape incompatible y el dataset
    fallará al cargarlos.
    """
    splits_path = data_dir / "splits.csv"
    if not splits_path.exists():
        typer.echo(f"Error: no se encontró splits.csv en {data_dir}. Ejecuta prepare-data primero.", err=True)
        raise typer.Exit(code=1)

    if not fasta_dir.is_dir():
        typer.echo(f"Error: el directorio de FASTA no existe: {fasta_dir}", err=True)
        raise typer.Exit(code=1)

    splits = pandas.read_csv(splits_path, dtype={"genome_id": str})
    genome_list = sorted(splits["genome_id"].unique())

    typer.echo(f"Extrayendo histogramas segmentados para {len(genome_list)} genomas...")
    extract_and_save_hier(
        genome_ids=genome_list,
        fasta_dir=fasta_dir,
        output_dir=data_dir,
        n_jobs=n_jobs,
    )
    typer.echo(f"Features guardadas en: {data_dir}/hier_bigru/")


@app.command(help="Extrae histogramas multi-escala segmentados (k=3,4,5) para HierSet v2.")
def prepare_hier_multi(
    data_dir: Path = typer.Option(
        Path("data/processed"),
        help="Directorio con outputs del pipeline (splits.csv, etc.).",
    ),
    fasta_dir: Path = typer.Option(
        Path("data/raw/fasta"),
        help="Directorio con archivos .fna de genomas.",
    ),
    n_jobs: int = typer.Option(
        1,
        help="Número de procesos paralelos. Usa -1 para el 80% de los CPUs.",
    ),
):
    """
    Extrae histogramas multi-escala segmentados (k=3,4,5) para AMRHierSetV2.

    Divide cada genoma en HIER_N_SEGMENTS segmentos y, por cada segmento, concatena
    los histogramas k=3 (64 dims), k=4 (256 dims) y k=5 (1024 dims) → 1344 dims
    por segmento. Los resultados se guardan en data/processed/hier_set_v2/.

    NOTA: si se cambian HIER_N_SEGMENTS o HIER_KMER_SIZES en constants.py, este
    comando debe volver a ejecutarse.
    """
    splits_path = data_dir / "splits.csv"
    if not splits_path.exists():
        typer.echo(f"Error: no se encontró splits.csv en {data_dir}. Ejecuta prepare-data primero.", err=True)
        raise typer.Exit(code=1)

    if not fasta_dir.is_dir():
        typer.echo(f"Error: el directorio de FASTA no existe: {fasta_dir}", err=True)
        raise typer.Exit(code=1)

    splits = pandas.read_csv(splits_path, dtype={"genome_id": str})
    genome_list = sorted(splits["genome_id"].unique())

    typer.echo(f"Extrayendo histogramas multi-escala para {len(genome_list)} genomas...")
    extract_and_save_hier_multi(
        genome_ids=genome_list,
        fasta_dir=fasta_dir,
        output_dir=data_dir,
        n_jobs=n_jobs,
    )
    typer.echo(f"Features guardadas en: {data_dir}/hier_set_v2/")


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
    lr_patience: int = typer.Option(5, help="Épocas sin mejora para reducir LR."),
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
        lr_patience=lr_patience,
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
    lr_patience: int = typer.Option(5, help="Épocas sin mejora para reducir LR."),
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
        "lr_patience": lr_patience,
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
        lr_patience=lr_patience,
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
    lr_patience: int = typer.Option(5, help="Épocas sin mejora para reducir LR."),
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
        "lr_patience": lr_patience,
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
        lr_patience=lr_patience,
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
    lr_patience: int = typer.Option(5, help="Épocas sin mejora para reducir LR."),
    pos_weight_scale: float = typer.Option(
        2.5,
        "--pos-weight-scale",
        help="Factor multiplicador del pos_weight base [King20].",
    ),
    weight_decay: float = typer.Option(
        1e-4,
        "--weight-decay",
        help="Regularización L2 con AdamW [Loshchilov19].",
    ),
):
    """
    Entrena el modelo Multi-Stream para predicción de AMR.

    Procesa cada histograma de k-meros (k=3,4,5) con un encoder sin dependencias
    secuenciales entre bins (proyección element-wise + attention pooling). La
    fusión entre streams está condicionada por el antibiótico [Ngiam11].
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
        "lr_patience": lr_patience,
        "pos_weight_scale": pos_weight_scale,
        "weight_decay": weight_decay,
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
        lr_patience=lr_patience,
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


@app.command(help="Entrena el modelo Hierarchical BiGRU sobre histogramas segmentados.")
def train_hier_bigru(
    data_dir: Path = typer.Option(
        Path("data/processed"),
        help="Directorio con outputs del pipeline (splits.csv, hier_bigru/, etc.).",
    ),
    output_dir: Path = typer.Option(
        Path("results/hier_bigru"),
        help="Directorio donde guardar modelo, métricas y gráficas.",
    ),
    epochs: int = typer.Option(100, help="Número máximo de épocas."),
    batch_size: int = typer.Option(32, help="Tamaño del mini-batch."),
    lr: float = typer.Option(0.001, help="Tasa de aprendizaje para AdamW."),
    patience: int = typer.Option(15, help="Épocas sin mejora para early stopping."),
    lr_patience: int = typer.Option(5, help="Épocas sin mejora para reducir LR."),
    pos_weight_scale: float = typer.Option(
        2.5,
        "--pos-weight-scale",
        help="Factor multiplicador del pos_weight base [King20].",
    ),
    weight_decay: float = typer.Option(
        1e-4,
        "--weight-decay",
        help="Regularización L2 [Goodfellow16, Cap. 7].",
    ),
):
    """
    Entrena la arquitectura Hierarchical BiGRU + Atención.

    Procesa el genoma como una secuencia de 64 histogramas locales, permitiendo
    que el mecanismo de atención se enfoque en segmentos específicos (ej. genes
    de resistencia) sin perder cobertura global.
    """
    set_seed(RANDOM_SEED)
    device = detect_device()
    typer.echo(f"Dispositivo: {device}")

    # Cargar datasets
    typer.echo("Cargando datos (Hierarchical BiGRU)...")
    train_ds = HierBiGRUDataset(data_dir, split="train")
    val_ds = HierBiGRUDataset(data_dir, split="val")
    test_ds = HierBiGRUDataset(data_dir, split="test")
    typer.echo(f"Muestras — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Modelo jerárquico
    model = AMRHierBiGRU.from_antibiotic_index(str(data_dir / "antibiotic_index.csv"))
    typer.echo(f"Modelo: {sum(p.numel() for p in model.parameters())} parámetros")

    # Función de pérdida
    base_pos_weight = HierBiGRUDataset.load_pos_weight(data_dir)
    scaled_pos_weight = base_pos_weight * pos_weight_scale
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([scaled_pos_weight], device=device),
    )
    typer.echo(f"pos_weight base: {base_pos_weight:.4f} → escalado: {scaled_pos_weight:.4f}")

    # Trazabilidad
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "model_type": "hier_bigru",
        "hier_n_segments": HIER_N_SEGMENTS,
        "hier_kmer_k": HIER_KMER_K,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "patience": patience,
        "lr_patience": lr_patience,
        "pos_weight_scale": pos_weight_scale,
        "weight_decay": weight_decay,
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
        lr_patience=lr_patience,
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


@app.command(help="Entrena el Hierarchical Set Encoder (sin dependencias secuenciales entre segmentos).")
def train_hier_set(
    data_dir: Path = typer.Option(
        Path("data/processed"),
        help="Directorio con outputs del pipeline (splits.csv, hier_bigru/, etc.).",
    ),
    output_dir: Path = typer.Option(
        Path("results/hier_set"),
        help="Directorio donde guardar modelo, métricas y gráficas.",
    ),
    epochs: int = typer.Option(100, help="Número máximo de épocas."),
    batch_size: int = typer.Option(32, help="Tamaño del mini-batch."),
    lr: float = typer.Option(0.001, help="Tasa de aprendizaje para AdamW."),
    patience: int = typer.Option(15, help="Épocas sin mejora para early stopping."),
    lr_patience: int = typer.Option(5, help="Épocas sin mejora para reducir LR."),
    pos_weight_scale: float = typer.Option(
        2.5,
        "--pos-weight-scale",
        help="Factor multiplicador del pos_weight base [King20].",
    ),
    weight_decay: float = typer.Option(
        1e-3,
        "--weight-decay",
        help="Regularización L2 con AdamW [Loshchilov19].",
    ),
):
    """
    Entrena el Hierarchical Set Encoder para predicción de AMR.

    Procesa los HIER_N_SEGMENTS segmentos de histogramas sin dependencias
    secuenciales entre ellos: proyección independiente por segmento + attention
    pooling condicionado en el antibiótico. A diferencia de HierBiGRU, no asume
    que segmentos adyacentes en el tensor sean biológicamente relacionados.
    """
    set_seed(RANDOM_SEED)
    device = detect_device()
    typer.echo(f"Dispositivo: {device}")

    typer.echo("Cargando datos (Hierarchical Set Encoder)...")
    train_ds = HierSetDataset(data_dir, split="train")
    val_ds = HierSetDataset(data_dir, split="val")
    test_ds = HierSetDataset(data_dir, split="test")
    typer.echo(f"Muestras — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = AMRHierSet.from_antibiotic_index(str(data_dir / "antibiotic_index.csv"))
    typer.echo(f"Modelo: {sum(p.numel() for p in model.parameters())} parámetros")

    base_pos_weight = HierSetDataset.load_pos_weight(data_dir)
    scaled_pos_weight = base_pos_weight * pos_weight_scale
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([scaled_pos_weight], device=device),
    )
    typer.echo(f"pos_weight base: {base_pos_weight:.4f} → escalado: {scaled_pos_weight:.4f}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "model_type": "hier_set",
        "hier_n_segments": HIER_N_SEGMENTS,
        "hier_kmer_k": HIER_KMER_K,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "patience": patience,
        "lr_patience": lr_patience,
        "pos_weight_scale": pos_weight_scale,
        "weight_decay": weight_decay,
        "device": str(device),
    }
    (output_dir / "params.json").write_text(json.dumps(params, indent=2))

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
        lr_patience=lr_patience,
        output_dir=output_dir,
        weight_decay=weight_decay,
    )

    typer.echo(f"\nResultados en test set:")
    typer.echo(f"  F1:      {test_metrics['f1']:.4f}")
    typer.echo(f"  Recall:  {test_metrics['recall']:.4f}")
    typer.echo(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
    typer.echo(f"  Umbral:  {test_metrics['threshold_used']:.4f}")
    typer.echo(f"\nGuardado en: {output_dir}")


@app.command(help="Evalúa un checkpoint HierSet sobre un split completo o un subset locked/new.")
def evaluate_hier_set_checkpoint(
    checkpoint: Path = typer.Option(
        ...,
        help="Ruta al checkpoint best_model.pt de HierSet.",
    ),
    data_dir: Path = typer.Option(
        Path("data/expanded/processed"),
        help="Directorio con splits.csv, cleaned_labels.csv, antibiotic_index.csv y hier_bigru/.",
    ),
    split: str = typer.Option(
        "test",
        help="Split a evaluar: train, val o test.",
    ),
    subset: str | None = typer.Option(
        None,
        "--subset",
        help="Valor de split_source a evaluar, por ejemplo locked o new. Si se omite, usa todo el split.",
    ),
    threshold: float | None = typer.Option(
        None,
        "--threshold",
        help="Umbral de decisión. Si se omite, usa threshold_used de metrics.json junto al checkpoint o 0.5.",
    ),
    batch_size: int = typer.Option(32, help="Tamaño del mini-batch para inferencia."),
    output: Path | None = typer.Option(
        None,
        help="Ruta donde guardar métricas JSON. Por defecto se guarda junto al checkpoint.",
    ),
    per_antibiotic: bool = typer.Option(
        False,
        "--per-antibiotic",
        help="Guarda métricas por antibiótico en CSV.",
    ),
    per_taxon: bool = typer.Option(
        False,
        "--per-taxon",
        help="Guarda métricas por taxon_id en CSV. Requiere metadata con taxon_id.",
    ),
    metadata_labels: Path | None = typer.Option(
        None,
        "--metadata-labels",
        help="CSV con genome_id, antibiotic y taxon_id. Por defecto usa labels_for_download.csv si existe en data_dir.",
    ),
):
    """Evalúa un checkpoint HierSet sin entrenar ni recalibrar sobre test."""
    valid_splits = {"train", "val", "test"}
    if split not in valid_splits:
        typer.echo(f"Error: --split debe ser uno de {sorted(valid_splits)}.", err=True)
        raise typer.Exit(code=1)
    if not checkpoint.exists():
        typer.echo(f"Error: no se encontró el checkpoint: {checkpoint}", err=True)
        raise typer.Exit(code=1)
    if not data_dir.is_dir():
        typer.echo(f"Error: no se encontró el directorio de datos: {data_dir}", err=True)
        raise typer.Exit(code=1)

    threshold_used = _resolve_threshold(checkpoint, threshold)
    if threshold_used < 0.0 or threshold_used > 1.0:
        typer.echo("Error: el umbral debe estar en [0, 1].", err=True)
        raise typer.Exit(code=1)

    set_seed(RANDOM_SEED)
    device = detect_device()
    typer.echo(f"Dispositivo: {device}")
    typer.echo(
        f"Evaluando HierSet: split={split}, subset={subset or 'all'}, threshold={threshold_used:.4f}"
    )

    dataset = HierSetDataset(data_dir, split=split, split_source=subset)
    loader = DataLoader(dataset, batch_size=batch_size)
    typer.echo(f"Muestras: {len(dataset)}")

    model = AMRHierSet.from_antibiotic_index(str(data_dir / "antibiotic_index.csv"))
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device)

    params_path = checkpoint.parent / "params.json"
    pos_weight_scale = 2.5
    if params_path.exists():
        params = json.loads(params_path.read_text())
        pos_weight_scale = float(params.get("pos_weight_scale", pos_weight_scale))
    base_pos_weight = HierSetDataset.load_pos_weight(data_dir)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([base_pos_weight * pos_weight_scale], device=device),
    )

    probabilities, targets, loss = collect_predictions(model, loader, criterion, device)
    metrics = compute_metrics(targets, probabilities, loss, threshold=threshold_used)
    metrics["optimal_threshold"] = find_optimal_threshold(targets, probabilities)
    metrics["threshold_used"] = threshold_used
    metrics["split"] = split
    metrics["subset"] = subset or "all"
    metrics["n_samples"] = len(dataset)

    predictions = dataset.records
    predictions["target"] = targets
    predictions["probability"] = probabilities
    predictions["prediction"] = (probabilities >= threshold_used).astype(int)

    if output is None:
        output = checkpoint.parent / f"metrics_{split}_{subset or 'all'}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2))

    if per_antibiotic:
        antibiotic_output = output.with_name(output.stem + "_by_antibiotic.csv")
        _save_group_metrics(predictions, "antibiotic", threshold_used, antibiotic_output)
        typer.echo(f"Métricas por antibiótico: {antibiotic_output}")

    if per_taxon:
        if metadata_labels is None:
            metadata_labels = data_dir / "labels_for_download.csv"
        if not metadata_labels.exists():
            typer.echo(
                f"Error: no se encontró metadata para taxon_id: {metadata_labels}",
                err=True,
            )
            raise typer.Exit(code=1)
        metadata = pandas.read_csv(metadata_labels, dtype={"genome_id": str})
        required = {"genome_id", "antibiotic", "taxon_id"}
        missing = required - set(metadata.columns)
        if missing:
            typer.echo(
                "Error: metadata_labels debe contener columnas: " + ", ".join(sorted(missing)),
                err=True,
            )
            raise typer.Exit(code=1)
        metadata = metadata[["genome_id", "antibiotic", "taxon_id"]].drop_duplicates(
            subset=["genome_id", "antibiotic"],
        )
        predictions = predictions.merge(
            metadata,
            on=["genome_id", "antibiotic"],
            how="left",
        )
        missing_taxon = int(predictions["taxon_id"].isna().sum())
        if missing_taxon:
            typer.echo(f"Advertencia: {missing_taxon} muestras sin taxon_id.")
        taxon_output = output.with_name(output.stem + "_by_taxon.csv")
        _save_group_metrics(predictions, "taxon_id", threshold_used, taxon_output)
        typer.echo(f"Métricas por taxon_id: {taxon_output}")

    typer.echo(f"F1:      {metrics['f1']:.4f}")
    typer.echo(f"Recall:  {metrics['recall']:.4f}")
    typer.echo(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    typer.echo(f"Umbral:  {metrics['threshold_used']:.4f}")
    typer.echo(f"Guardado en: {output}")


@app.command(help="Entrena el HierSet v2 (multi-head attention + histogramas multi-escala).")
def train_hier_set_v2(
    data_dir: Path = typer.Option(
        Path("data/processed"),
        help="Directorio con outputs del pipeline (splits.csv, hier_set_v2/, etc.).",
    ),
    output_dir: Path = typer.Option(
        Path("results/hier_set_v2"),
        help="Directorio donde guardar modelo, métricas y gráficas.",
    ),
    epochs: int = typer.Option(100, help="Número máximo de épocas."),
    batch_size: int = typer.Option(32, help="Tamaño del mini-batch."),
    lr: float = typer.Option(0.001, help="Tasa de aprendizaje para AdamW."),
    patience: int = typer.Option(15, help="Épocas sin mejora para early stopping."),
    lr_patience: int = typer.Option(5, help="Épocas sin mejora para reducir LR."),
    pos_weight_scale: float = typer.Option(
        2.5,
        "--pos-weight-scale",
        help="Factor multiplicador del pos_weight base [King20].",
    ),
    weight_decay: float = typer.Option(
        1e-3,
        "--weight-decay",
        help="Regularización L2 con AdamW [Loshchilov19].",
    ),
):
    """
    Entrena AMRHierSetV2 — multi-head cross-attention (H=4) sobre histogramas
    multi-escala (k=3,4,5) segmentados. Mismos hiperparámetros de entrenamiento
    que train-hier-set para comparación justa con v1.
    """
    set_seed(RANDOM_SEED)
    device = detect_device()
    typer.echo(f"Dispositivo: {device}")

    typer.echo("Cargando datos (HierSet v2)...")
    train_ds = HierSetV2Dataset(data_dir, split="train")
    val_ds = HierSetV2Dataset(data_dir, split="val")
    test_ds = HierSetV2Dataset(data_dir, split="test")
    typer.echo(f"Muestras — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = AMRHierSetV2.from_antibiotic_index(str(data_dir / "antibiotic_index.csv"))
    typer.echo(f"Modelo: {sum(p.numel() for p in model.parameters())} parámetros")

    base_pos_weight = HierSetV2Dataset.load_pos_weight(data_dir)
    scaled_pos_weight = base_pos_weight * pos_weight_scale
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([scaled_pos_weight], device=device),
    )
    typer.echo(f"pos_weight base: {base_pos_weight:.4f} → escalado: {scaled_pos_weight:.4f}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "model_type": "hier_set_v2",
        "hier_n_segments": HIER_N_SEGMENTS,
        "hier_kmer_sizes": [3, 4, 5],
        "n_heads": 4,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "patience": patience,
        "lr_patience": lr_patience,
        "pos_weight_scale": pos_weight_scale,
        "weight_decay": weight_decay,
        "device": str(device),
    }
    (output_dir / "params.json").write_text(json.dumps(params, indent=2))

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
        lr_patience=lr_patience,
        output_dir=output_dir,
        weight_decay=weight_decay,
    )

    typer.echo(f"\nResultados en test set:")
    typer.echo(f"  F1:      {test_metrics['f1']:.4f}")
    typer.echo(f"  Recall:  {test_metrics['recall']:.4f}")
    typer.echo(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
    typer.echo(f"  Umbral:  {test_metrics['threshold_used']:.4f}")
    typer.echo(f"\nGuardado en: {output_dir}")


if __name__ == "__main__":
    app()
