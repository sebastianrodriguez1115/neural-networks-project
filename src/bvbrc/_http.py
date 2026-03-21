"""
_http.py

Utilidades HTTP compartidas entre los clientes de BV-BRC.
Este módulo es interno al paquete bvbrc — no importar desde fuera.
"""

import logging
import time

import requests


logger = logging.getLogger(__name__)

# URL base de la API REST de BV-BRC
BVBRC_API_BASE_URL = "https://www.bv-brc.org/api"

# Tiempo de espera entre requests consecutivos (segundos).
# No hay límite de tasa documentado, pero se agrega como cortesía con el servidor.
SLEEP_BETWEEN_REQUESTS = 0.1

# Número máximo de registros por request (límite de la API de BV-BRC).
PAGE_SIZE = 25000

# Número de reintentos ante errores HTTP transitorios (timeouts, 5xx).
MAX_RETRIES = 3

# Timeout en segundos para cada request HTTP individual.
REQUEST_TIMEOUT = 120


def make_api_request_with_retries(url: str, headers: dict) -> requests.Response:
    """
    Realiza un GET HTTP a la URL dada con reintentos ante errores transitorios.

    Usa espera exponencial entre reintentos: 1s, 2s, 4s, ...
    Lanza RuntimeError si todos los intentos fallan.
    """
    last_exception = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as exception:
            last_exception = exception
            wait_seconds = 2 ** attempt  # 1s, 2s, 4s
            logger.warning(
                f"Error en intento {attempt + 1}/{MAX_RETRIES}: {exception}. "
                f"Reintentando en {wait_seconds}s..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Fallaron {MAX_RETRIES} intentos para la URL: {url}"
    ) from last_exception


def parse_total_records_from_content_range(content_range_header: str) -> int:
    """
    Extrae el total de registros del header HTTP Content-Range.

    La API de BV-BRC devuelve este header en cada respuesta paginada
    con el formato: 'items START-END/TOTAL'

    Ejemplo: 'items 0-24999/153422' → retorna 153422
    """
    total_records = int(content_range_header.split("/")[-1])
    return total_records
