"""
test_cleaning.py

Tests for data_pipeline/cleaning.py:
    - LabelCleaner: contradictory pair removal, deduplication
    - GenomeFilter: missing FASTA filtering, short genome filtering, valid genome pass-through
"""

import textwrap
from pathlib import Path

import pandas
import pytest

from data_pipeline.cleaning import GenomeFilter, LabelCleaner


# ── Helpers ────────────────────────────────────────────────────────────────────


def _write_labels(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """Writes a minimal AMR labels CSV."""
    df = pandas.DataFrame(rows, columns=["genome_id", "antibiotic", "resistant_phenotype"])
    df.to_csv(path, index=False)


def _write_fasta(path: Path, sequence: str = "ACGT" * 125_000) -> None:
    """Writes a FASTA file. Default sequence is 500 000 bp (meets MIN_GENOME_LENGTH)."""
    path.write_text(f">contig1\n{sequence}\n")


# ── LabelCleaner ───────────────────────────────────────────────────────────────


def test_label_cleaner_removes_contradictory_pairs(tmp_path):
    csv = tmp_path / "labels.csv"
    _write_labels(csv, [
        ("1.1", "amikacin", "Resistant"),
        ("1.1", "amikacin", "Susceptible"),  # contradictory — both rows removed
        ("1.2", "amikacin", "Resistant"),    # clean — kept
    ])

    result = LabelCleaner(csv).clean()

    assert len(result) == 1
    assert result.iloc[0]["genome_id"] == "1.2"


def test_label_cleaner_keeps_consistent_pairs(tmp_path):
    csv = tmp_path / "labels.csv"
    _write_labels(csv, [
        ("1.1", "amikacin",   "Resistant"),
        ("1.1", "ampicillin", "Susceptible"),
    ])

    result = LabelCleaner(csv).clean()

    assert len(result) == 2


def test_label_cleaner_removes_consistent_duplicates(tmp_path):
    csv = tmp_path / "labels.csv"
    _write_labels(csv, [
        ("1.1", "amikacin", "Resistant"),
        ("1.1", "amikacin", "Resistant"),  # exact duplicate — one removed
    ])

    result = LabelCleaner(csv).clean()

    assert len(result) == 1


def test_label_cleaner_returns_empty_when_all_contradictory(tmp_path):
    csv = tmp_path / "labels.csv"
    _write_labels(csv, [
        ("1.1", "amikacin", "Resistant"),
        ("1.1", "amikacin", "Susceptible"),
    ])

    result = LabelCleaner(csv).clean()

    assert len(result) == 0


def test_label_cleaner_casts_genome_id_to_str(tmp_path):
    csv = tmp_path / "labels.csv"
    _write_labels(csv, [("1234567.1", "amikacin", "Resistant")])

    result = LabelCleaner(csv).clean()

    assert pandas.api.types.is_string_dtype(result["genome_id"])  # str dtype in pandas


# ── GenomeFilter ───────────────────────────────────────────────────────────────


def test_genome_filter_excludes_missing_fasta(tmp_path):
    fasta_dir = tmp_path / "fastas"
    fasta_dir.mkdir()
    # No FASTA files created

    valid = GenomeFilter(fasta_dir).filter(["genome_1"])

    assert "genome_1" not in valid


def test_genome_filter_excludes_short_genomes(tmp_path):
    fasta_dir = tmp_path / "fastas"
    fasta_dir.mkdir()
    _write_fasta(fasta_dir / "short.fna", sequence="ACGT" * 10)  # 40 bp — too short

    valid = GenomeFilter(fasta_dir, min_length=500_000).filter(["short"])

    assert "short" not in valid


def test_genome_filter_includes_valid_genomes(tmp_path):
    fasta_dir = tmp_path / "fastas"
    fasta_dir.mkdir()
    _write_fasta(fasta_dir / "valid.fna")  # 500 000 bp — meets threshold

    valid = GenomeFilter(fasta_dir).filter(["valid"])

    assert "valid" in valid


def test_genome_filter_handles_mixed_batch(tmp_path):
    fasta_dir = tmp_path / "fastas"
    fasta_dir.mkdir()
    _write_fasta(fasta_dir / "good.fna")
    _write_fasta(fasta_dir / "short.fna", sequence="ACGT")

    valid = GenomeFilter(fasta_dir, min_length=500_000).filter(["good", "short", "missing"])

    assert valid == {"good"}
