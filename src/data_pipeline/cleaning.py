import logging
from pathlib import Path

import pandas
from Bio import SeqIO

from .constants import MIN_GENOME_LENGTH

logger = logging.getLogger(__name__)


class LabelCleaner:
    """Removes contradictory pairs and consistent duplicates from AMR labels."""

    def __init__(self, labels_path: Path):
        self._labels_path = Path(labels_path)
        self._dataframe: pandas.DataFrame | None = None
        self._n_initial = 0
        self._n_contradictory_pairs = 0
        self._n_contradictory_rows = 0
        self._n_duplicates_removed = 0

    def clean(self) -> pandas.DataFrame:
        self._load()
        self._remove_contradictory_pairs()
        self._deduplicate()
        logger.info(
            f"Cleaning complete: {self._n_initial} → {len(self._dataframe)} records "
            f"({self._n_contradictory_pairs} contradictory pairs / "
            f"{self._n_contradictory_rows} rows removed, "
            f"{self._n_duplicates_removed} consistent duplicates removed)"
        )
        return self._dataframe

    def _load(self) -> None:
        self._dataframe = pandas.read_csv(
            self._labels_path, dtype={"genome_id": str}
        )
        self._n_initial = len(self._dataframe)
        logger.info(f"Labels loaded: {self._n_initial} records")

    def _remove_contradictory_pairs(self) -> None:
        phenotype_counts = (
            self._dataframe.groupby(["genome_id", "antibiotic"])[
                "resistant_phenotype"
            ].nunique()
        )
        contradictory_indices = phenotype_counts[phenotype_counts > 1].index
        self._n_contradictory_pairs = len(contradictory_indices)
        mask = (
            self._dataframe.set_index(["genome_id", "antibiotic"])
            .index.isin(contradictory_indices)
        )
        self._n_contradictory_rows = int(mask.sum())
        self._dataframe = self._dataframe[~mask].reset_index(drop=True)

    def _deduplicate(self) -> None:
        before = len(self._dataframe)
        self._dataframe = self._dataframe.drop_duplicates(
            subset=["genome_id", "antibiotic"], keep="first"
        ).reset_index(drop=True)
        self._n_duplicates_removed = before - len(self._dataframe)


class GenomeFilter:
    """Filters genomes by FASTA availability and minimum sequence length."""

    def __init__(self, fasta_dir: Path, min_length: int = MIN_GENOME_LENGTH):
        self._fasta_dir = Path(fasta_dir)
        self._min_length = min_length
        self._valid: set[str] = set()
        self._short: list[str] = []
        self._missing: list[str] = []

    def filter(self, genome_ids: list[str]) -> set[str]:
        for genome_id in genome_ids:
            fasta_path = self._fasta_dir / f"{genome_id}.fna"
            if not fasta_path.exists():
                self._missing.append(genome_id)
                continue
            length = self._compute_length(fasta_path)
            if length < self._min_length:
                self._short.append(genome_id)
                logger.info(
                    f"Genome {genome_id} discarded: "
                    f"{length:,} bp < {self._min_length:,} bp"
                )
            else:
                self._valid.add(genome_id)
        logger.info(
            f"Genomic filter: {len(self._valid)} valid, "
            f"{len(self._short)} short discarded, "
            f"{len(self._missing)} missing FASTA"
        )
        return self._valid

    @staticmethod
    def _compute_length(fasta_path: Path) -> int:
        return sum(len(record.seq) for record in SeqIO.parse(fasta_path, "fasta"))
