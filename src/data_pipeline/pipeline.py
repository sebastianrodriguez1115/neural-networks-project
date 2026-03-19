import logging
from pathlib import Path

import numpy
import pandas

from .cleaning import GenomeFilter, LabelCleaner
from .features import (
    KmerExtractor,
    build_antibiotic_index,
    mlp_vector_to_bigru_matrix,
    normalize_features,
    split_genomes,
)

logger = logging.getLogger(__name__)


def _clean_labels(labels_path: Path, output_dir: Path) -> pandas.DataFrame:
    logger.info("Step 1: Cleaning labels")
    cleaned = LabelCleaner(labels_path).clean()
    cleaned_path = output_dir / "cleaned_labels.csv"
    cleaned[["genome_id", "antibiotic", "resistant_phenotype"]].to_csv(
        cleaned_path, index=False
    )
    logger.info(f"Cleaned labels saved to: {cleaned_path}")
    return cleaned


def _filter_genomes(cleaned: pandas.DataFrame, fasta_dir: Path) -> pandas.DataFrame:
    logger.info("Step 2: Genomic quality filter")
    all_genome_ids = cleaned["genome_id"].unique().tolist()
    valid_genomes = GenomeFilter(fasta_dir).filter(all_genome_ids)
    filtered = cleaned[cleaned["genome_id"].isin(valid_genomes)].reset_index(
        drop=True
    )
    logger.info(
        f"Labels after genomic filter: {len(filtered)} records, "
        f"{filtered['genome_id'].nunique()} genomes"
    )
    return filtered


def _save_antibiotic_index(
    cleaned: pandas.DataFrame, output_dir: Path
) -> None:
    antibiotic_idx = build_antibiotic_index(cleaned["antibiotic"])
    idx_path = output_dir / "antibiotic_index.csv"
    antibiotic_idx.to_csv(idx_path, index=False)
    logger.info(
        f"Antibiotic index ({len(antibiotic_idx)} entries) saved to: {idx_path}"
    )


def _split_genomes(
    cleaned: pandas.DataFrame, output_dir: Path
) -> tuple[pandas.DataFrame, set[str]]:
    logger.info("Step 3: Train/val/test split")
    splits = split_genomes(cleaned)
    splits_path = output_dir / "splits.csv"
    splits.to_csv(splits_path, index=False)
    logger.info(f"Splits saved to: {splits_path}")

    train_ids = set(splits[splits["split"] == "train"]["genome_id"])
    train_labels = cleaned[cleaned["genome_id"].isin(train_ids)]
    n_susceptible = (train_labels["resistant_phenotype"] == "Susceptible").sum()
    n_resistant = (train_labels["resistant_phenotype"] == "Resistant").sum()
    pos_weight = n_susceptible / n_resistant if n_resistant > 0 else 1.0
    logger.info(
        f"pos_weight (train set): {pos_weight:.4f} "
        f"(Susceptible={n_susceptible}, Resistant={n_resistant})"
    )
    return splits, train_ids


def _extract_kmers(genome_ids: list[str], fasta_dir: Path) -> dict[str, numpy.ndarray]:
    logger.info("Step 4: K-mer extraction")
    mlp_vectors: dict[str, numpy.ndarray] = {}
    for i, genome_id in enumerate(genome_ids):
        fasta_path = fasta_dir / f"{genome_id}.fna"
        logger.info(f"[{i + 1}/{len(genome_ids)}] Extracting k-mers: {genome_id}")
        extractor = KmerExtractor(fasta_path)
        extractor.extract()
        mlp_vectors[genome_id] = extractor.to_mlp_vector()
    return mlp_vectors


def _normalize_and_save(
    mlp_vectors: dict[str, numpy.ndarray],
    train_ids: set[str],
    genome_ids: list[str],
    output_dir: Path,
) -> None:
    logger.info("Step 5: Normalization (train set stats)")
    mlp_normalized, mlp_mean, mlp_std = normalize_features(mlp_vectors, train_ids)

    logger.info("Step 6: Saving features")
    mlp_dir = output_dir / "mlp"
    bigru_dir = output_dir / "bigru"
    mlp_dir.mkdir(parents=True, exist_ok=True)
    bigru_dir.mkdir(parents=True, exist_ok=True)

    for genome_id in genome_ids:
        mlp_vec = mlp_normalized[genome_id]
        numpy.save(mlp_dir / f"{genome_id}.npy", mlp_vec)
        numpy.save(bigru_dir / f"{genome_id}.npy", mlp_vector_to_bigru_matrix(mlp_vec))

    numpy.save(output_dir / "mlp_mean.npy", mlp_mean)
    numpy.save(output_dir / "mlp_std.npy", mlp_std)
    logger.info(f"Features saved to: {output_dir}")


def run_pipeline(
    labels_path: Path,
    fasta_dir: Path,
    output_dir: Path,
) -> None:
    """Runs the full data preprocessing pipeline."""
    labels_path = Path(labels_path)
    fasta_dir = Path(fasta_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned = _clean_labels(labels_path, output_dir)
    cleaned = _filter_genomes(cleaned, fasta_dir)
    _save_antibiotic_index(cleaned, output_dir)
    _, train_ids = _split_genomes(cleaned, output_dir)
    genome_list = sorted(cleaned["genome_id"].unique())
    mlp_vectors = _extract_kmers(genome_list, fasta_dir)
    _normalize_and_save(mlp_vectors, train_ids, genome_list, output_dir)

    logger.info(
        f"Pipeline complete. {len(genome_list)} genomes processed. "
        f"Output: {output_dir}"
    )
