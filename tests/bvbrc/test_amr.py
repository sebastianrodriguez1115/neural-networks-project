"""
test_amr.py

Tests de bvbrc/amr.py:
    - _build_amr_rql_query (función pura)
    - fetch_amr_labels (make_api_request_with_retries mockeado)
"""

from unittest.mock import MagicMock, patch

import pandas
import pytest

from bvbrc.amr import (
    AMR_FIELDS_TO_SELECT,
    AMRFetcher,
    ESKAPE_TAXON_IDS,
    fetch_amr_labels,
)
from bvbrc._http import PAGE_SIZE


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_mock_amr_response(records: list[dict], content_range: str | None = None) -> MagicMock:
    """Crea una respuesta HTTP mockeada con los registros AMR dados."""
    mock_response = MagicMock()
    mock_response.json.return_value = records
    mock_response.headers = {"Content-Range": content_range} if content_range else {}
    return mock_response


def _make_fake_amr_record(genome_id: str = "1280.1", phenotype: str = "Resistant") -> dict:
    """Crea un registro AMR mínimo válido para usar en tests."""
    return {
        "genome_id": genome_id,
        "taxon_id": 1280,
        "antibiotic": "penicillin",
        "resistant_phenotype": phenotype,
        "laboratory_typing_method": "MIC",
        "testing_standard": "CLSI",
    }


# ── AMRFetcher._build_query ─────────────────────────────────────────────────────

def test_query_contains_taxon_id_for_single_taxon():
    query = AMRFetcher._build_query([1280])
    assert "in(taxon_id,(1280))" in query


def test_query_contains_all_taxon_ids():
    query = AMRFetcher._build_query([1280, 573, 470])
    assert "in(taxon_id,(1280,573,470))" in query


def test_query_filters_by_laboratory_evidence():
    query = AMRFetcher._build_query([1280])
    assert "eq(evidence,Laboratory)" in query


def test_query_includes_only_binary_phenotypes():
    query = AMRFetcher._build_query([1280])
    assert "in(resistant_phenotype,(Resistant,Susceptible))" in query


def test_query_implicitly_excludes_intermediate():
    """Intermediate no debe aparecer como valor permitido en la consulta."""
    query = AMRFetcher._build_query([1280])
    assert "Intermediate" not in query


# ── fetch_amr_labels ───────────────────────────────────────────────────────────

def test_returns_dataframe_with_correct_columns():
    mock_response = _make_mock_amr_response([_make_fake_amr_record()])

    with patch("bvbrc.amr.make_api_request_with_retries", return_value=mock_response):
        result = fetch_amr_labels(taxon_ids=[1280])

    assert isinstance(result, pandas.DataFrame)
    assert list(result.columns) == AMR_FIELDS_TO_SELECT


def test_returns_dataframe_with_downloaded_records():
    records = [_make_fake_amr_record("1280.1"), _make_fake_amr_record("1280.2")]
    mock_response = _make_mock_amr_response(records)

    with patch("bvbrc.amr.make_api_request_with_retries", return_value=mock_response):
        result = fetch_amr_labels(taxon_ids=[1280])

    assert len(result) == 2
    assert list(result["genome_id"]) == ["1280.1", "1280.2"]


def test_returns_empty_dataframe_when_no_records():
    mock_response = _make_mock_amr_response([])

    with patch("bvbrc.amr.make_api_request_with_retries", return_value=mock_response):
        result = fetch_amr_labels(taxon_ids=[1280])

    assert isinstance(result, pandas.DataFrame)
    assert len(result) == 0


def test_paginates_until_batch_is_smaller_than_page_size():
    """
    Simula dos páginas: la primera devuelve PAGE_SIZE registros (hay más disponibles),
    la segunda devuelve menos (última página). El total debe ser PAGE_SIZE + 1.
    """
    first_batch = [_make_fake_amr_record(f"1280.{i}") for i in range(PAGE_SIZE)]
    second_batch = [_make_fake_amr_record("1280.99999")]

    first_response = _make_mock_amr_response(
        first_batch,
        content_range=f"items 0-{PAGE_SIZE - 1}/{PAGE_SIZE + 1}",
    )
    second_response = _make_mock_amr_response(second_batch)

    with patch(
        "bvbrc.amr.make_api_request_with_retries",
        side_effect=[first_response, second_response],
    ):
        result = fetch_amr_labels(taxon_ids=[1280])

    assert len(result) == PAGE_SIZE + 1


def test_uses_eskape_taxons_by_default():
    """Sin taxon_ids, la URL del request debe contener todos los taxon IDs ESKAPE."""
    mock_response = _make_mock_amr_response([])

    with patch(
        "bvbrc.amr.make_api_request_with_retries", return_value=mock_response
    ) as mock_request:
        fetch_amr_labels()

    called_url = mock_request.call_args[0][0]
    for taxon_id in ESKAPE_TAXON_IDS.values():
        assert str(taxon_id) in called_url


def test_saves_csv_when_output_path_is_given(tmp_path):
    records = [_make_fake_amr_record()]
    mock_response = _make_mock_amr_response(records)
    output_file = tmp_path / "labels.csv"

    with patch("bvbrc.amr.make_api_request_with_retries", return_value=mock_response):
        fetch_amr_labels(taxon_ids=[1280], output_path=output_file)

    assert output_file.exists()
    saved = pandas.read_csv(output_file)
    assert len(saved) == 1
    assert list(saved.columns) == AMR_FIELDS_TO_SELECT


def test_creates_parent_directory_for_csv_if_missing(tmp_path):
    mock_response = _make_mock_amr_response([_make_fake_amr_record()])
    output_file = tmp_path / "subdir" / "labels.csv"

    with patch("bvbrc.amr.make_api_request_with_retries", return_value=mock_response):
        fetch_amr_labels(taxon_ids=[1280], output_path=output_file)

    assert output_file.exists()


def test_does_not_save_csv_when_no_output_path_given(tmp_path):
    mock_response = _make_mock_amr_response([_make_fake_amr_record()])

    with patch("bvbrc.amr.make_api_request_with_retries", return_value=mock_response):
        fetch_amr_labels(taxon_ids=[1280])

    assert list(tmp_path.iterdir()) == []
