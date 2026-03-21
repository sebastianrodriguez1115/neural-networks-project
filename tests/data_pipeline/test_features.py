"""
test_features.py

Tests de data_pipeline/features.py:
    - KmerExtractor: dimensiones del vector, conteo de k-meros
    - build_antibiotic_index: mapeo ordenado
    - normalize_features: usa estadísticas del train set, no del dataset completo
    - mlp_vector_to_bigru_matrix: forma del output
    - split_genomes: ratios, todos los genomas asignados, estratificación
"""

import numpy
import pandas
import pytest

from data_pipeline.features import (
    KmerExtractor,
    build_antibiotic_index,
    mlp_vector_to_bigru_matrix,
    normalize_features,
    split_genomes,
)
from data_pipeline.constants import BIGRU_PAD_DIM, KMER_DIMS, KMER_SIZES, TOTAL_KMER_DIM, TRAIN_RATIO


# ── Helpers ────────────────────────────────────────────────────────────────────


def _write_fasta(path, sequence="ACGT" * 10):
    path.write_text(f">contig1\n{sequence}\n")


def _make_labels(n_resistant: int = 10, n_susceptible: int = 10) -> pandas.DataFrame:
    """Crea un DataFrame de etiquetas balanceado con un antibiótico por genoma."""
    rows = (
        [(f"r{i}", "amikacin", "Resistant") for i in range(n_resistant)]
        + [(f"s{i}", "amikacin", "Susceptible") for i in range(n_susceptible)]
    )
    return pandas.DataFrame(rows, columns=["genome_id", "antibiotic", "resistant_phenotype"])


# ── KmerExtractor ──────────────────────────────────────────────────────────────


def test_kmer_extractor_mlp_vector_dimension(tmp_path):
    fasta = tmp_path / "genome.fna"
    _write_fasta(fasta, "ACGT" * 100)

    extractor = KmerExtractor(fasta)
    extractor.extract()

    assert extractor.to_mlp_vector().shape == (TOTAL_KMER_DIM,)


def test_kmer_extractor_bigru_matrix_shape(tmp_path):
    fasta = tmp_path / "genome.fna"
    _write_fasta(fasta, "ACGT" * 100)

    extractor = KmerExtractor(fasta)
    extractor.extract()
    vec = extractor.to_mlp_vector()
    matrix = mlp_vector_to_bigru_matrix(vec)

    assert matrix.shape == (BIGRU_PAD_DIM, len(KMER_SIZES))


def test_kmer_extractor_counts_are_non_negative(tmp_path):
    fasta = tmp_path / "genome.fna"
    _write_fasta(fasta, "ACGTACGT")

    extractor = KmerExtractor(fasta)
    extractor.extract()

    assert (extractor.to_mlp_vector() >= 0).all()


def test_kmer_extractor_identical_sequences_give_identical_vectors(tmp_path):
    fasta1 = tmp_path / "g1.fna"
    fasta2 = tmp_path / "g2.fna"
    seq = "ACGTACGTACGT" * 50
    _write_fasta(fasta1, seq)
    _write_fasta(fasta2, seq)

    e1 = KmerExtractor(fasta1)
    e1.extract()
    e2 = KmerExtractor(fasta2)
    e2.extract()

    numpy.testing.assert_array_equal(e1.to_mlp_vector(), e2.to_mlp_vector())


# ── build_antibiotic_index ─────────────────────────────────────────────────────


def test_build_antibiotic_index_sorted_alphabetically():
    series = pandas.Series(["penicillin", "amikacin", "ampicillin"])
    result = build_antibiotic_index(series)

    assert list(result["antibiotic"]) == ["amikacin", "ampicillin", "penicillin"]


def test_build_antibiotic_index_zero_based_integers():
    series = pandas.Series(["b", "a", "c"])
    result = build_antibiotic_index(series)

    assert list(result["index"]) == [0, 1, 2]


def test_build_antibiotic_index_deduplicates():
    series = pandas.Series(["amikacin", "amikacin", "penicillin"])
    result = build_antibiotic_index(series)

    assert len(result) == 2


# ── normalize_features ─────────────────────────────────────────────────────────


def test_normalize_features_uses_train_mean_not_full_mean():
    vectors = {
        "g1": numpy.array([1.0, 2.0]),
        "g2": numpy.array([3.0, 4.0]),
        "g3": numpy.array([100.0, 200.0]),  # outlier exclusivo del test
    }
    train_ids = {"g1", "g2"}

    _, mean, _ = normalize_features(vectors, train_ids)

    assert mean[0] == pytest.approx(2.0)   # (1 + 3) / 2


def test_normalize_features_returns_dict_with_all_genome_ids():
    vectors = {"g1": numpy.array([1.0]), "g2": numpy.array([2.0])}
    train_ids = {"g1"}

    normalized, _, _ = normalize_features(vectors, train_ids)

    assert set(normalized.keys()) == {"g1", "g2"}


def test_normalize_features_zero_std_columns_become_one():
    vectors = {
        "g1": numpy.array([5.0, 1.0]),
        "g2": numpy.array([5.0, 3.0]),
    }
    train_ids = {"g1", "g2"}

    _, _, std = normalize_features(vectors, train_ids)

    assert std[0] == pytest.approx(1.0)  # std cero reemplazado por 1.0


# ── mlp_vector_to_bigru_matrix ─────────────────────────────────────────────────


def test_mlp_vector_to_bigru_matrix_shape():
    vec = numpy.zeros(TOTAL_KMER_DIM)
    matrix = mlp_vector_to_bigru_matrix(vec)

    assert matrix.shape == (BIGRU_PAD_DIM, len(KMER_SIZES))


def test_mlp_vector_to_bigru_matrix_padded_positions_are_zero():
    vec = numpy.ones(TOTAL_KMER_DIM)
    matrix = mlp_vector_to_bigru_matrix(vec)

    # histograma k=3: KMER_DIMS[0] dims rellenas a BIGRU_PAD_DIM — posiciones [KMER_DIMS[0]..] en col 0 deben ser 0
    assert matrix[KMER_DIMS[0], 0] == pytest.approx(0.0)
    # histograma k=4: KMER_DIMS[1] dims rellenas a BIGRU_PAD_DIM — posiciones [KMER_DIMS[1]..] en col 1 deben ser 0
    assert matrix[KMER_DIMS[1], 1] == pytest.approx(0.0)
    # histograma k=5: KMER_DIMS[2] dims exactas (== BIGRU_PAD_DIM) — sin relleno en col 2
    assert matrix[BIGRU_PAD_DIM - 1, 2] == pytest.approx(1.0)


# ── split_genomes ──────────────────────────────────────────────────────────────


def test_split_genomes_assigns_all_genomes():
    labels = _make_labels(n_resistant=10, n_susceptible=10)
    splits = split_genomes(labels)

    all_genome_ids = set(labels["genome_id"].unique())
    split_genome_ids = set(splits["genome_id"])
    assert split_genome_ids == all_genome_ids


def test_split_genomes_approximate_train_ratio():
    labels = _make_labels(n_resistant=10, n_susceptible=10)
    splits = split_genomes(labels)

    total = len(splits)
    train_count = (splits["split"] == "train").sum()
    assert train_count / total == pytest.approx(TRAIN_RATIO, abs=0.10)


def test_split_genomes_has_train_val_test():
    labels = _make_labels(n_resistant=10, n_susceptible=10)
    splits = split_genomes(labels)

    assert set(splits["split"].unique()) == {"train", "val", "test"}


def test_split_genomes_no_genome_in_multiple_splits():
    labels = _make_labels(n_resistant=10, n_susceptible=10)
    splits = split_genomes(labels)

    assert splits["genome_id"].nunique() == len(splits)
