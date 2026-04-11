import concurrent.futures
import json
import logging
import os
from functools import partial
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


def _filter_genomes(
    cleaned: pandas.DataFrame, fasta_dir: Path, output_dir: Path
) -> pandas.DataFrame:
    logger.info("Step 2: Genomic quality filter")
    all_genome_ids = cleaned["genome_id"].unique().tolist()
    genome_filter = GenomeFilter(fasta_dir)
    valid_genomes = genome_filter.filter(all_genome_ids)
    filtered = cleaned[cleaned["genome_id"].isin(valid_genomes)].reset_index(
        drop=True
    )
    logger.info(
        f"Labels after genomic filter: {len(filtered)} records, "
        f"{filtered['genome_id'].nunique()} genomes"
    )
    _save_discarded_genomes(genome_filter, output_dir)
    return filtered


def _save_discarded_genomes(genome_filter: GenomeFilter, output_dir: Path) -> None:
    rows = []
    for genome_id in genome_filter.missing:
        rows.append({"genome_id": genome_id, "reason": "missing_fasta"})
    for genome_id in genome_filter.short:
        rows.append({"genome_id": genome_id, "reason": "below_min_length"})
    if not rows:
        return
    path = output_dir / "discarded_genomes.csv"
    pandas.DataFrame(rows).to_csv(path, index=False)
    logger.info(f"Discarded genomes ({len(rows)}) saved to: {path}")


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

    stats = {
        "n_resistant": int(n_resistant),
        "n_susceptible": int(n_susceptible),
        "pos_weight": pos_weight,
    }
    stats_path = output_dir / "train_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info(f"Train stats saved to: {stats_path}")

    return splits, train_ids


def _extract_single_genome(genome_id: str, fasta_dir: Path) -> tuple[str, numpy.ndarray]:
    """Extrae k-meros de un solo genoma. Función de nivel módulo para poder ser picklada."""
    extractor = KmerExtractor(fasta_dir / f"{genome_id}.fna")
    extractor.extract()
    return genome_id, extractor.to_mlp_vector()


def _extract_kmers(
    genome_ids: list[str],
    fasta_dir: Path,
    n_jobs: int = 1,
) -> dict[str, numpy.ndarray]:
    logger.info("Step 4: K-mer extraction")
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError(f"n_jobs debe ser >= 1 o -1 (80% de los CPUs), recibido: {n_jobs}")

    worker = partial(_extract_single_genome, fasta_dir=fasta_dir)

    if n_jobs == 1:
        mlp_vectors: dict[str, numpy.ndarray] = {}
        for i, (genome_id, vector) in enumerate(map(worker, genome_ids)):
            logger.info(f"[{i + 1}/{len(genome_ids)}] Extracting k-mers: {genome_id}")
            mlp_vectors[genome_id] = vector
        return mlp_vectors

    workers = max(1, int(os.cpu_count() * 0.8)) if n_jobs == -1 else n_jobs
    total = len(genome_ids)
    logger.info(f"Parallel k-mer extraction: {total} genomes, {workers} workers")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, gid): gid for gid in genome_ids}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            genome_id, vector = future.result()
            results.append((genome_id, vector))
            logger.info(f"[{i + 1}/{total}] K-mers extracted: {genome_id}")
    return dict(results)


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


def _extract_single_genome_tokens(
    genome_id: str, fasta_dir: Path, k: int, max_len: int
) -> tuple[str, numpy.ndarray]:
    """Extrae secuencia de tokens de un solo genoma."""
    extractor = KmerExtractor(fasta_dir / f"{genome_id}.fna")
    return genome_id, extractor.to_token_sequence(k=k, max_len=max_len)


def _extract_and_save_tokens(
    genome_ids: list[str],
    fasta_dir: Path,
    output_dir: Path,
    k: int,
    max_len: int,
    n_jobs: int = 1,
) -> None:
    """Extrae secuencias de tokens de k-meros y las guarda como .npy."""
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError(f"n_jobs debe ser >= 1 o -1 (80% de los CPUs), recibido: {n_jobs}")

    logger.info(f"Step 7: Token extraction (k={k}, max_len={max_len})")
    token_dir = output_dir / "token_bigru"
    token_dir.mkdir(parents=True, exist_ok=True)

    total = len(genome_ids)
    worker = partial(
        _extract_single_genome_tokens, fasta_dir=fasta_dir, k=k, max_len=max_len
    )

    if n_jobs == 1:
        for i, (genome_id, tokens) in enumerate(map(worker, genome_ids)):
            numpy.save(token_dir / f"{genome_id}.npy", tokens)
            if (i + 1) % 10 == 0 or (i + 1) == total:
                logger.info(f"[{i + 1}/{total}] Tokens extracted: {genome_id}")
    else:
        workers = max(1, int(os.cpu_count() * 0.8)) if n_jobs == -1 else n_jobs
        logger.info(f"Parallel token extraction: {total} genomes, {workers} workers")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(worker, gid): gid for gid in genome_ids}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                genome_id, tokens = future.result()
                numpy.save(token_dir / f"{genome_id}.npy", tokens)
                if (i + 1) % 10 == 0 or (i + 1) == total:
                    logger.info(f"[{i + 1}/{total}] Tokens extracted: {genome_id}")


def run_pipeline(
    labels_path: Path,
    fasta_dir: Path,
    output_dir: Path,
    n_jobs: int = 1,
) -> None:
    """Ejecuta el pipeline completo de preprocesamiento de datos.

    Args:
        n_jobs: Procesos paralelos para extracción de k-meros.
                1=secuencial (defecto), -1=80% de los CPUs disponibles.
    """
    labels_path = Path(labels_path)
    fasta_dir = Path(fasta_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned = _clean_labels(labels_path, output_dir)
    cleaned = _filter_genomes(cleaned, fasta_dir, output_dir)
    _save_antibiotic_index(cleaned, output_dir)
    _, train_ids = _split_genomes(cleaned, output_dir)
    genome_list = sorted(cleaned["genome_id"].unique())
    mlp_vectors = _extract_kmers(genome_list, fasta_dir, n_jobs=n_jobs)
    _normalize_and_save(mlp_vectors, train_ids, genome_list, output_dir)

    logger.info(
        f"Pipeline complete. {len(genome_list)} genomes processed. "
        f"Output: {output_dir}"
    )
