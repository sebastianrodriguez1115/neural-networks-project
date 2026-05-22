"""
amr.py

Cliente para el endpoint genome_amr de BV-BRC.
Descarga etiquetas de resistencia antimicrobiana (AMR) desde BV-BRC.

Referencia de la API: docs/reference/bvbrc_api.md
"""

import logging
import time
from pathlib import Path
from urllib.parse import quote

import pandas

from ._http import (
    BVBRC_API_BASE_URL,
    PAGE_SIZE,
    SLEEP_BETWEEN_REQUESTS,
    make_api_request_with_retries,
    parse_total_records_from_content_range,
)


logger = logging.getLogger(__name__)

# Taxon IDs de NCBI para los organismos del grupo ESKAPE.
# NOTA: Enterobacter spp. (547) fue excluido. Aunque es parte de ESKAPE, el endpoint
# /genome_amr/ no soporta filtrado por taxon_lineage_ids, lo que obligaría a realizar
# una doble consulta previa por /genome/ para obtener los genome_ids, complicando el pipeline.
ESKAPE_TAXON_IDS = {
    "Enterococcus faecium":    1352,
    "Staphylococcus aureus":   1280,
    "Klebsiella pneumoniae":   573,
    "Acinetobacter baumannii": 470,
    "Pseudomonas aeruginosa":  287,
    # "Enterobacter spp.":       547, # Excluido por complejidad en BV-BRC API
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


class AMRFetcher:
    """Encapsula una operación de descarga paginada del endpoint genome_amr."""

    _ENDPOINT = f"{BVBRC_API_BASE_URL}/genome_amr/"
    _HEADERS = {"Accept": "application/json"}
    _FIELDS = ",".join(AMR_FIELDS_TO_SELECT)

    def __init__(self, taxon_ids: list[int] | None, typing_method: str | None = None):
        self._query = self._build_query(taxon_ids, typing_method=typing_method)
        self._records: list[dict] = []
        self._offset = 0
        self._total: int | None = None

    def fetch(self, output_path: Path | None = None) -> pandas.DataFrame:
        """
        Ejecuta la descarga paginada y retorna un DataFrame con los registros AMR.

        Args:
            output_path: Si se provee, guarda el DataFrame resultante como CSV.

        Returns:
            DataFrame con columnas definidas en AMR_FIELDS_TO_SELECT.
        """
        logger.info("Iniciando descarga de etiquetas AMR desde BV-BRC...")
        while True:
            batch_size = self._fetch_next_page()
            if self._is_complete(batch_size):
                break
            time.sleep(SLEEP_BETWEEN_REQUESTS)
        logger.info(f"Descarga completada. Total de registros obtenidos: {len(self._records)}")

        df = pandas.DataFrame(self._records, columns=AMR_FIELDS_TO_SELECT)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Etiquetas AMR guardadas en: {output_path}")

        return df

    def _fetch_next_page(self) -> int:
        logger.info(
            f"Descargando registros {self._offset}–{self._offset + PAGE_SIZE - 1}"
            + (f" de {self._total}" if self._total is not None else "")
        )
        response = make_api_request_with_retries(self._page_url(), self._HEADERS)
        batch = response.json()
        self._records.extend(batch)
        self._update_total(response)
        self._offset += PAGE_SIZE
        return len(batch)

    def _update_total(self, response) -> None:
        if self._total is None and "Content-Range" in response.headers:
            self._total = parse_total_records_from_content_range(response.headers["Content-Range"])
            logger.info(f"Total de registros AMR a descargar: {self._total}")

    def _is_complete(self, batch_size: int) -> bool:
        return batch_size < PAGE_SIZE or (self._total is not None and self._offset >= self._total)

    def _page_url(self) -> str:
        return (
            f"{self._ENDPOINT}"
            f"?{self._query}"
            f"&select({self._FIELDS})"
            f"&limit({PAGE_SIZE},{self._offset})"
            f"&sort(+genome_id)"
        )

    @staticmethod
    def _build_query(
        taxon_ids: list[int] | None,
        typing_method: str | None = None,
    ) -> str:
        """
        Construye la consulta en Resource Query Language (RQL) para el endpoint genome_amr.

        Filtros aplicados:
            - taxon_id pertenece a la lista dada, si se provee
            - evidence = Laboratory (excluye predicciones computacionales)
            - resistant_phenotype es Resistant o Susceptible (excluye Intermediate
              y Non-susceptible, que son etiquetas ambiguas para clasificación binaria)
            - laboratory_typing_method, si se provee
        """
        filters = []
        if taxon_ids is not None:
            if not taxon_ids:
                raise ValueError("taxon_ids no puede ser una lista vacía")
            taxon_ids_joined = ",".join(str(taxon_id) for taxon_id in taxon_ids)
            filters.append(f"in(taxon_id,({taxon_ids_joined}))")

        filters.extend([
            "eq(evidence,Laboratory)",
            "in(resistant_phenotype,(Resistant,Susceptible))",
        ])

        if typing_method is not None:
            typing_method = typing_method.strip()
            if not typing_method:
                raise ValueError("typing_method no puede estar vacío")
            filters.append(f"eq(laboratory_typing_method,{quote(typing_method, safe='')})")

        return f"and({','.join(filters)})"


def fetch_amr_labels(
    taxon_ids: list[int] | None = None,
    output_path: Path | None = None,
    *,
    all_taxa: bool = False,
    exclude_taxon_ids: list[int] | None = None,
    typing_method: str | None = None,
) -> pandas.DataFrame:
    """Descarga etiquetas AMR de BV-BRC. Ver AMRFetcher.fetch para detalles."""
    if all_taxa and taxon_ids is not None:
        raise ValueError("No se puede combinar all_taxa=True con taxon_ids explícitos")

    if not all_taxa and taxon_ids is None:
        taxon_ids = list(ESKAPE_TAXON_IDS.values())

    dataframe = AMRFetcher(taxon_ids, typing_method=typing_method).fetch()

    if typing_method is not None:
        typing_method = typing_method.strip()
        dataframe = dataframe[
            dataframe["laboratory_typing_method"] == typing_method
        ].reset_index(drop=True)

    if exclude_taxon_ids:
        dataframe = dataframe[~dataframe["taxon_id"].isin(exclude_taxon_ids)].reset_index(drop=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)
        logger.info(f"Etiquetas AMR guardadas en: {output_path}")

    return dataframe
