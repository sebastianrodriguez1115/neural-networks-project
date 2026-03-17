import logging
from pathlib import Path

import numpy
import pandas
from Bio import SeqIO
from sklearn.model_selection import train_test_split

from .constants import (
    BASE_TO_INDEX,
    BIGRU_PAD_DIM,
    KMER_DIMS,
    KMER_SIZES,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
)

logger = logging.getLogger(__name__)


class KmerExtractor:
    """Extracts k-mer frequency histograms from a genome FASTA file."""

    def __init__(self, fasta_path: Path):
        self._fasta_path = Path(fasta_path)
        self._histograms: dict[int, numpy.ndarray] = {}

    def extract(self) -> dict[int, numpy.ndarray]:
        sequences = self._read_sequences()
        for k, dim in zip(KMER_SIZES, KMER_DIMS):
            histogram = numpy.zeros(dim, dtype=numpy.float64)
            for sequence in sequences:
                self._count_kmers(sequence, k, histogram)
            self._histograms[k] = histogram
        return self._histograms

    def to_mlp_vector(self) -> numpy.ndarray:
        """Concatenate histograms (k=3,4,5) into a 1344-dim vector."""
        return numpy.concatenate([self._histograms[k] for k in KMER_SIZES])

    def _read_sequences(self) -> list[str]:
        return [
            str(record.seq).upper()
            for record in SeqIO.parse(self._fasta_path, "fasta")
        ]

    @staticmethod
    def _count_kmers(sequence: str, k: int, histogram: numpy.ndarray) -> None:
        """Count k-mers using a 2-bit rolling hash. O(n) per sequence.

        Each base is encoded as 2 bits (A=00, C=01, G=10, T=11). The hash
        slides one base at a time via left shift, OR, and AND with a bitmask.
        Runs of ambiguous bases (e.g. 'N') reset the hash.

        Technique: standard 2-bit encoding for DNA k-mer counting.
        See: Compeau & Pevzner, Bioinformatics Algorithms (2014), Ch. 9.
        """
        mask = (4**k) - 1
        current = 0
        valid_count = 0
        for base in sequence:
            base_index = BASE_TO_INDEX.get(base)
            if base_index is None:
                valid_count = 0
                current = 0
            else:
                current = ((current << 2) | base_index) & mask
                valid_count += 1
                if valid_count >= k:
                    histogram[current] += 1


def build_antibiotic_index(antibiotics: pandas.Series) -> pandas.DataFrame:
    """Creates a sorted mapping from antibiotic name to integer index."""
    unique = sorted(antibiotics.unique())
    return pandas.DataFrame({"antibiotic": unique, "index": range(len(unique))})


def split_genomes(
    labels: pandas.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    random_seed: int = RANDOM_SEED,
) -> pandas.DataFrame:
    """Stratified train/val/test split by genome_id.

    Stratifies on the majority phenotype per genome to preserve
    class distribution across splits.
    """
    genome_phenotype = (
        labels.groupby("genome_id")["resistant_phenotype"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"resistant_phenotype": "majority_phenotype"})
    )

    val_test_ratio = 1.0 - train_ratio
    train_ids, val_test_ids, _, val_test_labels = train_test_split(
        genome_phenotype["genome_id"],
        genome_phenotype["majority_phenotype"],
        test_size=val_test_ratio,
        stratify=genome_phenotype["majority_phenotype"],
        random_state=random_seed,
    )

    relative_val = val_ratio / val_test_ratio
    val_ids, test_ids = train_test_split(
        val_test_ids,
        test_size=1.0 - relative_val,
        stratify=val_test_labels,
        random_state=random_seed,
    )

    splits = pandas.concat(
        [
            pandas.DataFrame({"genome_id": train_ids, "split": "train"}),
            pandas.DataFrame({"genome_id": val_ids, "split": "val"}),
            pandas.DataFrame({"genome_id": test_ids, "split": "test"}),
        ],
        ignore_index=True,
    )
    logger.info(
        f"Split complete: "
        f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
    )
    return splits


def normalize_features(
    vectors: dict[str, numpy.ndarray],
    train_ids: set[str],
) -> tuple[dict[str, numpy.ndarray], numpy.ndarray, numpy.ndarray]:
    """Normalize using train-set mean and std. Returns (normalized, mean, std)."""
    train_matrix = numpy.stack([vectors[gid] for gid in sorted(train_ids)])
    mean = train_matrix.mean(axis=0)
    std = train_matrix.std(axis=0)
    std[std == 0] = 1.0
    normalized = {gid: (vec - mean) / std for gid, vec in vectors.items()}
    return normalized, mean, std


def mlp_vector_to_bigru_matrix(mlp_vector: numpy.ndarray) -> numpy.ndarray:
    """Convert a 1344-dim MLP vector into a [1024, 3] BiGRU input matrix.

    Splits the vector back into histograms (64, 256, 1024), pads each
    to 1024 with zeros, and stacks as columns.
    """
    columns = []
    offset = 0
    for dim in KMER_DIMS:
        histogram = mlp_vector[offset : offset + dim]
        padded = numpy.pad(histogram, (0, BIGRU_PAD_DIM - dim))
        columns.append(padded)
        offset += dim
    return numpy.stack(columns, axis=1)
