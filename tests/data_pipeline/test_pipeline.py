"""
test_pipeline.py

Integration test for data_pipeline/pipeline.py.
Tests run_pipeline end-to-end with synthetic FASTA files and a minimal labels CSV.
Private orchestration functions are covered indirectly via run_pipeline.
"""

import numpy
import pandas
import pytest

from data_pipeline.pipeline import run_pipeline


# ── Helpers ────────────────────────────────────────────────────────────────────


def _write_fasta(path, n_bases: int = 500_000) -> None:
    """Writes a FASTA file with n_bases bp (default meets MIN_GENOME_LENGTH)."""
    sequence = ("ACGT" * (n_bases // 4 + 1))[:n_bases]
    path.write_text(f">contig1\n{sequence}\n")


def _make_labels_csv(path, genome_ids: list[str]) -> None:
    """Writes a labels CSV alternating Resistant/Susceptible by genome position."""
    rows = [
        (gid, "amikacin", "Resistant" if i % 2 == 0 else "Susceptible")
        for i, gid in enumerate(genome_ids)
    ]
    pandas.DataFrame(
        rows, columns=["genome_id", "antibiotic", "resistant_phenotype"]
    ).to_csv(path, index=False)


@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory):
    """Runs run_pipeline once for all tests in this module.

    Creates 20 genomes (10 R, 10 S), writes FASTA files and labels CSV,
    executes the full pipeline, and returns (output_dir, genome_ids).
    Uses module scope to avoid running the expensive pipeline 6 times.
    """
    tmp_path = tmp_path_factory.mktemp("pipeline")
    genome_ids = [f"1.{i}" for i in range(1, 21)]
    fasta_dir = tmp_path / "fastas"
    fasta_dir.mkdir()
    for gid in genome_ids:
        _write_fasta(fasta_dir / f"{gid}.fna")
    labels_path = tmp_path / "labels.csv"
    _make_labels_csv(labels_path, genome_ids)
    output_dir = tmp_path / "output"
    run_pipeline(labels_path, fasta_dir, output_dir)
    return output_dir, genome_ids


# ── run_pipeline ───────────────────────────────────────────────────────────────


def test_run_pipeline_creates_cleaned_labels(pipeline_output):
    output_dir, _ = pipeline_output
    assert (output_dir / "cleaned_labels.csv").exists()


def test_run_pipeline_creates_splits_csv(pipeline_output):
    output_dir, _ = pipeline_output
    splits = pandas.read_csv(output_dir / "splits.csv")
    assert set(splits["split"].unique()) == {"train", "val", "test"}


def test_run_pipeline_creates_antibiotic_index(pipeline_output):
    output_dir, _ = pipeline_output
    idx = pandas.read_csv(output_dir / "antibiotic_index.csv")
    assert "antibiotic" in idx.columns
    assert "index" in idx.columns


def test_run_pipeline_saves_mlp_features_per_genome(pipeline_output):
    output_dir, genome_ids = pipeline_output
    mlp_dir = output_dir / "mlp"
    for gid in genome_ids:
        vec = numpy.load(mlp_dir / f"{gid}.npy")
        assert vec.shape == (1344,)


def test_run_pipeline_saves_bigru_features_per_genome(pipeline_output):
    output_dir, genome_ids = pipeline_output
    bigru_dir = output_dir / "bigru"
    for gid in genome_ids:
        matrix = numpy.load(bigru_dir / f"{gid}.npy")
        assert matrix.shape == (1024, 3)


def test_run_pipeline_saves_normalization_stats(pipeline_output):
    output_dir, _ = pipeline_output
    assert (output_dir / "mlp_mean.npy").exists()
    assert (output_dir / "mlp_std.npy").exists()
