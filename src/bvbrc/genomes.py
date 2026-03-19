"""
genomes.py

Cliente para el endpoint genome_sequence de BV-BRC.
Descarga genomas en formato FASTA dado una lista de genome_id.

Referencia de la API: docs/reference/bvbrc_api.md
"""

import logging
import time
from pathlib import Path

from ._http import (
    BVBRC_API_BASE_URL,
    SLEEP_BETWEEN_REQUESTS,
    make_api_request_with_retries,
)


logger = logging.getLogger(__name__)


class GenomeFetcher:
    """Encapsula la descarga FASTA de un único genoma desde BV-BRC."""

    _ENDPOINT = f"{BVBRC_API_BASE_URL}/genome_sequence/"
    _HEADERS = {"Accept": "application/dna+fasta"}

    def __init__(self, genome_id: str, output_directory: Path):
        self._genome_id = genome_id
        self._output_directory = Path(output_directory)
        self._output_path = self._output_directory / f"{genome_id}.fna"
        self._url = (
            f"{self._ENDPOINT}"
            f"?eq(genome_id,{genome_id})"
            f"&limit(10000,0)"  # 10000 contigs es un techo razonable para bacterias
        )

    def fetch(self) -> Path:
        """
        Descarga el genoma FASTA y lo guarda en disco.

        Si el archivo ya existe en output_directory, la descarga se omite
        para no repetir trabajo en caso de interrupción.

        Returns:
            Path al archivo FASTA guardado.

        Raises:
            RuntimeError: Si la respuesta FASTA está vacía (genome_id inválido
                          o genoma sin secuencias en BV-BRC).
        """
        if self._output_path.exists():
            logger.debug(f"Genoma {self._genome_id} ya descargado, omitiendo.")
            return self._output_path

        self._output_directory.mkdir(parents=True, exist_ok=True)
        content = self._download()
        self._output_path.write_text(content, encoding="utf-8")
        logger.debug(f"Genoma {self._genome_id} guardado en: {self._output_path}")
        return self._output_path

    def _download(self) -> str:
        response = make_api_request_with_retries(self._url, self._HEADERS)
        content = response.text
        if not content.strip():
            raise RuntimeError(
                f"El genoma {self._genome_id} no devolvió secuencias FASTA. "
                f"Verificar que el genome_id sea válido en BV-BRC."
            )
        return content


class GenomeBatchFetcher:
    """Descarga FASTA de una lista de genomas, tolerando fallos individuales."""

    def __init__(self, genome_ids: list[str], output_directory: Path):
        self._genome_ids = genome_ids
        self._output_directory = Path(output_directory)
        self._successful: dict[str, Path] = {}
        self._failed: list[str] = []

    def fetch(self) -> dict[str, Path]:
        """
        Descarga cada genoma de la lista, uno por uno.

        Omite los genomas cuyo archivo .fna ya exista. Los genome_id que
        fallen se registran como advertencia sin abortar el resto.

        Returns:
            Diccionario {genome_id: path_al_archivo} con las descargas exitosas.
        """
        total = len(self._genome_ids)
        logger.info(f"Iniciando descarga de {total} genomas FASTA...")

        for index, genome_id in enumerate(self._genome_ids):
            logger.info(f"[{index + 1}/{total}] Descargando genoma {genome_id}")
            self._fetch_one(genome_id)
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        self._log_summary()
        return self._successful

    def _fetch_one(self, genome_id: str) -> None:
        try:
            path = GenomeFetcher(genome_id, self._output_directory).fetch()
            self._successful[genome_id] = path
        except (RuntimeError, OSError) as exception:
            logger.error(f"Error descargando genoma {genome_id}: {exception}")
            self._failed.append(genome_id)

    def _log_summary(self) -> None:
        total = len(self._genome_ids)
        logger.info(
            f"Descarga finalizada. "
            f"Exitosos: {len(self._successful)}/{total}. "
            f"Fallidos: {len(self._failed)}."
        )
        if self._failed:
            logger.warning(
                f"Los siguientes genome_id fallaron y deben reintentarse manualmente: "
                f"{self._failed}"
            )


def download_genome_fasta(genome_id: str, output_directory: Path) -> Path:
    """Descarga el FASTA de un genoma. Ver GenomeFetcher.fetch para detalles."""
    return GenomeFetcher(genome_id, output_directory).fetch()


def download_multiple_genomes_fasta(
    genome_ids: list[str],
    output_directory: Path,
) -> dict[str, Path]:
    """Descarga FASTA de una lista de genomas. Ver GenomeBatchFetcher.fetch para detalles."""
    return GenomeBatchFetcher(genome_ids, output_directory).fetch()

