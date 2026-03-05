"""
main.py

Punto de entrada CLI del proyecto de predicción de AMR.

Comandos:
    download-amr      Descarga etiquetas AMR de BV-BRC para organismos ESKAPE
    download-genomes  Descarga genomas FASTA para los genome_id del CSV de etiquetas
    eda               Análisis exploratorio del dataset de etiquetas AMR

Uso:
    uv run python main.py --help
    uv run python main.py download-amr
    uv run python main.py eda --labels data/processed/amr_labels.csv
"""

import logging
from pathlib import Path

import pandas
import typer

from bvbrc import download_multiple_genomes_fasta, fetch_amr_labels
from eda import run_eda


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = typer.Typer(
    help="Herramientas CLI para el proyecto de predicción de resistencia antimicrobiana.",
    no_args_is_help=True,
)


@app.command()
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


@app.command()
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
    """
    if not labels.exists():
        typer.echo(f"Error: no se encontró el archivo de etiquetas: {labels}", err=True)
        raise typer.Exit(code=1)

    amr_labels = pandas.read_csv(labels)

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


def _sample_genome_ids(amr_labels: pandas.DataFrame, n_per_species: int) -> list[str]:
    """Selecciona hasta n genomas por especie, estratificados por fenotipo mayoritario."""
    amr_dedup = amr_labels.drop_duplicates(subset=["genome_id", "antibiotic"])
    sample_ids = []

    for taxon_id, group in amr_dedup.groupby("taxon_id"):
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


@app.command()
def eda(
    labels: Path = typer.Option(
        Path("data/processed/amr_labels.csv"),
        help="Ruta al CSV de etiquetas AMR.",
    ),
    top_n: int = typer.Option(
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

    run_eda(labels_path=labels, top_n=top_n, genomes_dir=genomes_dir)


if __name__ == "__main__":
    app()
