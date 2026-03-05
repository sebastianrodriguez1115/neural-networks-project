"""
amr.py

Cliente para el endpoint genome_amr de BV-BRC.
Descarga etiquetas de resistencia antimicrobiana (AMR) para organismos ESKAPE.

Referencia de la API: docs/implementation/bvbrc_api.md
"""

import logging
import time
from pathlib import Path

import pandas

from ._http import (
    BVBRC_API_BASE_URL,
    MAX_RETRIES,
    PAGE_SIZE,
    SLEEP_BETWEEN_REQUESTS,
    make_api_request_with_retries,
    parse_total_records_from_content_range,
)


logger = logging.getLogger(__name__)

# Taxon IDs de NCBI para los organismos del grupo ESKAPE.
# Para Enterobacter se usa el ID del género (547) porque agrupa
# E. cloacae, E. aerogenes y otras especies clínicamente relevantes.
ESKAPE_TAXON_IDS = {
    "Enterococcus faecium":    1352,
    "Staphylococcus aureus":   1280,
    "Klebsiella pneumoniae":    573,
    "Acinetobacter baumannii":  470,
    "Pseudomonas aeruginosa":   287,
    "Enterobacter spp.":        547,
}

# Campos que se solicitan al endpoint genome_amr.
# Se pide solo lo necesario para construir los triples (genome_id, antibiotic, label)
# y para poder auditar la calidad de las etiquetas.
AMR_FIELDS_TO_SELECT = [
    "genome_id",
    "taxon_id",
    "antibiotic",
    "resistant_phenotype",
    "laboratory_typing_method",
    "testing_standard",
]


def _build_amr_rql_query(taxon_ids: list[int]) -> str:
    """
    Construye el query en Resource Query Language (RQL) para el endpoint genome_amr.

    Filtros aplicados:
        - taxon_id pertenece a la lista dada (organismos ESKAPE)
        - evidence = Laboratory (excluye predicciones computacionales)
        - resistant_phenotype es Resistant o Susceptible (excluye Intermediate
          y Non-susceptible, que son etiquetas ambiguas para clasificación binaria)
    """
    taxon_ids_joined = ",".join(str(taxon_id) for taxon_id in taxon_ids)

    rql_query = (
        f"and("
        f"in(taxon_id,({taxon_ids_joined})),"
        f"eq(evidence,Laboratory),"
        f"in(resistant_phenotype,(Resistant,Susceptible))"
        f")"
    )
    return rql_query


def fetch_amr_labels(
    taxon_ids: list[int] | None = None,
    output_path: Path | None = None,
) -> pandas.DataFrame:
    """
    Descarga las etiquetas AMR de BV-BRC para los taxones especificados.

    Consulta el endpoint genome_amr con paginación automática. Filtra por
    evidencia de laboratorio y fenotipos binarios (Resistant / Susceptible).

    Args:
        taxon_ids:   Lista de taxon IDs a consultar.
                     Por defecto usa todos los ESKAPE (ESKAPE_TAXON_IDS).
        output_path: Si se provee, guarda el DataFrame resultante como CSV.

    Returns:
        DataFrame con columnas definidas en AMR_FIELDS_TO_SELECT.
    """
    if taxon_ids is None:
        taxon_ids = list(ESKAPE_TAXON_IDS.values())

    rql_query = _build_amr_rql_query(taxon_ids)
    fields_joined = ",".join(AMR_FIELDS_TO_SELECT)
    endpoint_url = f"{BVBRC_API_BASE_URL}/genome_amr/"

    # Solicitamos JSON para poder construir el DataFrame directamente
    request_headers = {"Accept": "application/json"}

    all_records = []
    current_offset = 0
    total_records = None  # Se llena al leer el primer Content-Range

    logger.info("Iniciando descarga de etiquetas AMR desde BV-BRC...")

    while True:
        # Construir URL con los parámetros de paginación para este bloque
        paginated_url = (
            f"{endpoint_url}"
            f"?{rql_query}"
            f"&select({fields_joined})"
            f"&limit({PAGE_SIZE},{current_offset})"
            f"&sort(+genome_id)"  # sort obligatorio para paginación consistente
        )

        logger.info(
            f"Descargando registros {current_offset}–{current_offset + PAGE_SIZE - 1}"
            + (f" de {total_records}" if total_records is not None else "")
        )

        response = make_api_request_with_retries(paginated_url, request_headers)
        batch_records = response.json()
        all_records.extend(batch_records)

        # El header Content-Range contiene el total de registros disponibles.
        # Lo leemos solo en el primer request para evitar trabajo redundante.
        if total_records is None and "Content-Range" in response.headers:
            total_records = parse_total_records_from_content_range(
                response.headers["Content-Range"]
            )
            logger.info(f"Total de registros AMR a descargar: {total_records}")

        current_offset += PAGE_SIZE

        # Condición de corte: batch incompleto (último bloque) o llegamos al total
        if len(batch_records) < PAGE_SIZE:
            break
        if total_records is not None and current_offset >= total_records:
            break

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    logger.info(f"Descarga completada. Total de registros obtenidos: {len(all_records)}")

    amr_dataframe = pandas.DataFrame(all_records, columns=AMR_FIELDS_TO_SELECT)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        amr_dataframe.to_csv(output_path, index=False)
        logger.info(f"Etiquetas AMR guardadas en: {output_path}")

    return amr_dataframe
