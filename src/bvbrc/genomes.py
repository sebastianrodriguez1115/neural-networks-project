"""
genomes.py

Cliente para el endpoint genome_sequence de BV-BRC.
Descarga genomas en formato FASTA dado una lista de genome_id.

Referencia de la API: docs/implementation/bvbrc_api.md
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


def download_genome_fasta(genome_id: str, output_directory: Path) -> Path:
    """
    Descarga el genoma FASTA de un genome_id dado desde BV-BRC.

    Consulta el endpoint genome_sequence y solicita la respuesta en
    formato FASTA mediante el header Accept: application/dna+fasta.
    Todos los contigs del genoma quedan en un único archivo .fna.

    Si el archivo ya existe en output_directory, la descarga se omite
    para no repetir trabajo en caso de interrupción.

    Args:
        genome_id:        Identificador del genoma en BV-BRC (e.g. '1280.12345').
        output_directory: Directorio donde guardar el archivo .fna.

    Returns:
        Path al archivo FASTA guardado.

    Raises:
        RuntimeError: Si la respuesta FASTA está vacía (genome_id inválido
                      o genoma sin secuencias en BV-BRC).
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    output_file_path = output_directory / f"{genome_id}.fna"

    # Si el archivo ya existe, no volver a descargarlo
    if output_file_path.exists():
        logger.debug(f"Genoma {genome_id} ya descargado, omitiendo.")
        return output_file_path

    endpoint_url = f"{BVBRC_API_BASE_URL}/genome_sequence/"

    # El header Accept controla el formato de la respuesta.
    # 'application/dna+fasta' indica que queremos las secuencias en formato FASTA.
    request_headers = {"Accept": "application/dna+fasta"}

    request_url = (
        f"{endpoint_url}"
        f"?eq(genome_id,{genome_id})"
        f"&limit(10000,0)"  # 10000 contigs es un techo razonable para bacterias
    )

    response = make_api_request_with_retries(request_url, request_headers)
    fasta_content = response.text

    if not fasta_content.strip():
        raise RuntimeError(
            f"El genoma {genome_id} no devolvió secuencias FASTA. "
            f"Verificar que el genome_id sea válido en BV-BRC."
        )

    output_file_path.write_text(fasta_content, encoding="utf-8")
    logger.debug(f"Genoma {genome_id} guardado en: {output_file_path}")

    return output_file_path


def download_multiple_genomes_fasta(
    genome_ids: list[str],
    output_directory: Path,
) -> dict[str, Path]:
    """
    Descarga el FASTA de una lista de genomas, uno por uno.

    Omite los genomas que ya estén en output_directory. Los genome_id que
    fallen se registran como advertencia para que puedan reintentarse.

    Args:
        genome_ids:       Lista de genome_id a descargar.
        output_directory: Directorio donde guardar los archivos .fna.

    Returns:
        Diccionario {genome_id: path_al_archivo} con las descargas exitosas.
        Los genome_id que fallaron no aparecen en el diccionario.
    """
    output_directory = Path(output_directory)

    successful_downloads: dict[str, Path] = {}
    failed_genome_ids: list[str] = []

    total_genomes = len(genome_ids)
    logger.info(f"Iniciando descarga de {total_genomes} genomas FASTA...")

    for index, genome_id in enumerate(genome_ids):
        logger.info(f"[{index + 1}/{total_genomes}] Descargando genoma {genome_id}")

        try:
            fasta_file_path = download_genome_fasta(genome_id, output_directory)
            successful_downloads[genome_id] = fasta_file_path

        except Exception as exception:
            logger.error(f"Error descargando genoma {genome_id}: {exception}")
            failed_genome_ids.append(genome_id)

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    logger.info(
        f"Descarga finalizada. "
        f"Exitosos: {len(successful_downloads)}/{total_genomes}. "
        f"Fallidos: {len(failed_genome_ids)}."
    )

    if failed_genome_ids:
        logger.warning(
            f"Los siguientes genome_id fallaron y deben reintentarse manualmente: "
            f"{failed_genome_ids}"
        )

    return successful_downloads
