"""
test_cleaning.py

Tests de data_pipeline/cleaning.py:
    - LabelCleaner: eliminación de pares contradictorios, deduplicación
    - GenomeFilter: filtro por FASTA faltante, filtro por genoma corto, paso de genomas válidos
"""

from pathlib import Path

import pandas
import pytest

from data_pipeline.cleaning import GenomeFilter, LabelCleaner
from data_pipeline.constants import MIN_GENOME_LENGTH

_ACGT_PATTERN = "ACGT"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _write_labels(path: Path, rows: list) -> None:
    """Escribe un CSV mínimo de etiquetas AMR."""
    new_rows = []
    for row in rows:
        if len(row) == 3:
            new_rows.append((*row, "Broth dilution"))
        else:
            new_rows.append(row)
    
    df = pandas.DataFrame(new_rows, columns=[
        "genome_id", "antibiotic", "resistant_phenotype", "laboratory_typing_method"
    ])
    df.to_csv(path, index=False)


def _write_fasta(path: Path, sequence: str = _ACGT_PATTERN * (MIN_GENOME_LENGTH // len(_ACGT_PATTERN))) -> None:
    """Escribe un archivo FASTA. La secuencia por defecto supera MIN_GENOME_LENGTH."""
    path.write_text(f">contig1\n{sequence}\n")


# ── LabelCleaner ───────────────────────────────────────────────────────────────


def test_label_cleaner_filters_by_typing_method(tmp_path):
    csv = tmp_path / "labels.csv"
    _write_labels(csv, [
        ("1.1", "amikacin", "Resistant", "Broth dilution"),
        ("1.2", "amikacin", "Resistant", "Disk diffusion"),  # se elimina
        ("1.3", "amikacin", "Resistant", None),             # se elimina
    ])

    result = LabelCleaner(csv, min_records_per_antibiotic=1).clean()

    assert len(result) == 1
    assert result.iloc[0]["genome_id"] == "1.1"


def test_label_cleaner_removes_contradictory_pairs(tmp_path):
    csv = tmp_path / "labels.csv"
    _write_labels(csv, [
        ("1.1", "amikacin", "Resistant"),
        ("1.1", "amikacin", "Susceptible"),  # contradictorio — ambas filas se eliminan
        ("1.2", "amikacin", "Resistant"),    # limpio — se conserva
    ])

    result = LabelCleaner(csv, min_records_per_antibiotic=1).clean()

    assert len(result) == 1
    assert result.iloc[0]["genome_id"] == "1.2"


def test_label_cleaner_keeps_consistent_pairs(tmp_path):
    csv = tmp_path / "labels.csv"
    _write_labels(csv, [
        ("1.1", "amikacin",   "Resistant"),
        ("1.1", "ampicillin", "Susceptible"),
    ])

    result = LabelCleaner(csv, min_records_per_antibiotic=1).clean()

    assert len(result) == 2


def test_label_cleaner_removes_consistent_duplicates(tmp_path):
    csv = tmp_path / "labels.csv"
    _write_labels(csv, [
        ("1.1", "amikacin", "Resistant"),
        ("1.1", "amikacin", "Resistant"),  # duplicado exacto — uno se elimina
    ])

    result = LabelCleaner(csv, min_records_per_antibiotic=1).clean()

    assert len(result) == 1


def test_label_cleaner_returns_empty_when_all_contradictory(tmp_path):
    csv = tmp_path / "labels.csv"
    _write_labels(csv, [
        ("1.1", "amikacin", "Resistant"),
        ("1.1", "amikacin", "Susceptible"),
    ])

    result = LabelCleaner(csv, min_records_per_antibiotic=1).clean()

    assert len(result) == 0


def test_label_cleaner_casts_genome_id_to_str(tmp_path):
    csv = tmp_path / "labels.csv"
    _write_labels(csv, [("1234567.1", "amikacin", "Resistant")])

    result = LabelCleaner(csv, min_records_per_antibiotic=1).clean()

    assert pandas.api.types.is_string_dtype(result["genome_id"])  # tipo str en pandas


def test_label_cleaner_filters_low_frequency_antibiotics(tmp_path):
    csv = tmp_path / "labels.csv"
    _write_labels(csv, [
        ("1.1", "rare_antibiotic", "Resistant"),
        ("1.2", "rare_antibiotic", "Resistant"),
        ("2.1", "common_antibiotic", "Resistant"),
        ("2.2", "common_antibiotic", "Resistant"),
        ("2.3", "common_antibiotic", "Resistant"),
    ])

    # Caso 1: Umbral de 3 (debería eliminar rare_antibiotic)
    result = LabelCleaner(csv, min_records_per_antibiotic=3).clean()
    assert len(result) == 3
    assert all(result["antibiotic"] == "common_antibiotic")

    # Caso 2: Umbral de 5 (debería eliminar ambos)
    result = LabelCleaner(csv, min_records_per_antibiotic=5).clean()
    assert len(result) == 0


# ── GenomeFilter ───────────────────────────────────────────────────────────────


def test_genome_filter_excludes_missing_fasta(tmp_path):
    fasta_dir = tmp_path / "fastas"
    fasta_dir.mkdir()
    # No se crean archivos FASTA

    valid = GenomeFilter(fasta_dir).filter(["genome_1"])

    assert "genome_1" not in valid


def test_genome_filter_excludes_short_genomes(tmp_path):
    fasta_dir = tmp_path / "fastas"
    fasta_dir.mkdir()
    _write_fasta(fasta_dir / "short.fna", sequence="ACGT" * 10)  # 40 pb — demasiado corto

    valid = GenomeFilter(fasta_dir, min_length=MIN_GENOME_LENGTH).filter(["short"])

    assert "short" not in valid


def test_genome_filter_includes_valid_genomes(tmp_path):
    fasta_dir = tmp_path / "fastas"
    fasta_dir.mkdir()
    _write_fasta(fasta_dir / "valid.fna")  # 500 000 pb — supera el umbral

    valid = GenomeFilter(fasta_dir).filter(["valid"])

    assert "valid" in valid


def test_genome_filter_handles_mixed_batch(tmp_path):
    fasta_dir = tmp_path / "fastas"
    fasta_dir.mkdir()
    _write_fasta(fasta_dir / "good.fna")
    _write_fasta(fasta_dir / "short.fna", sequence="ACGT")

    valid = GenomeFilter(fasta_dir, min_length=MIN_GENOME_LENGTH).filter(["good", "short", "missing"])

    assert valid == {"good"}
