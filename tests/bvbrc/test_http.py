"""
test_http.py

Tests de bvbrc/_http.py:
    - parse_total_records_from_content_range (función pura)
    - make_api_request_with_retries (requests.get mockeado)
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from bvbrc._http import (
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    make_api_request_with_retries,
    parse_total_records_from_content_range,
)


# ── parse_total_records_from_content_range ─────────────────────────────────────

def test_parses_total_from_typical_header():
    result = parse_total_records_from_content_range("items 0-24999/153422")
    assert result == 153422


def test_parses_total_from_single_page():
    result = parse_total_records_from_content_range("items 0-99/100")
    assert result == 100


def test_parses_total_when_there_is_one_record():
    result = parse_total_records_from_content_range("items 0-0/1")
    assert result == 1


def test_parses_total_from_large_dataset():
    result = parse_total_records_from_content_range("items 25000-49999/1000000")
    assert result == 1000000


# ── make_api_request_with_retries ──────────────────────────────────────────────

def test_returns_response_on_first_successful_attempt():
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None

    with patch("bvbrc._http.requests.get", return_value=mock_response) as mock_get:
        result = make_api_request_with_retries(
            url="http://example.com",
            headers={"Accept": "application/json"},
        )

    assert result is mock_response
    mock_get.assert_called_once()


def test_passes_url_and_headers_to_requests_get():
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    test_url = "http://example.com/api/genome_amr/"
    test_headers = {"Accept": "application/json"}

    with patch("bvbrc._http.requests.get", return_value=mock_response) as mock_get:
        make_api_request_with_retries(url=test_url, headers=test_headers)

    mock_get.assert_called_once_with(test_url, headers=test_headers, timeout=REQUEST_TIMEOUT)


@patch("bvbrc._http.time.sleep")
def test_retries_on_transient_error_and_succeeds(_mock_sleep):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None

    with patch("bvbrc._http.requests.get") as mock_get:
        # Los primeros dos intentos fallan, el tercero tiene éxito
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("timeout"),
            requests.exceptions.ConnectionError("timeout"),
            mock_response,
        ]
        result = make_api_request_with_retries("http://example.com", {})

    assert result is mock_response
    assert mock_get.call_count == 3


@patch("bvbrc._http.time.sleep")
def test_raises_runtime_error_when_all_retries_fail(_mock_sleep):
    with patch("bvbrc._http.requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.ConnectionError("timeout")

        with pytest.raises(RuntimeError, match="Fallaron"):
            make_api_request_with_retries("http://example.com", {})

    assert mock_get.call_count == MAX_RETRIES


@patch("bvbrc._http.time.sleep")
def test_original_exception_is_chained_in_runtime_error(_mock_sleep):
    original_error = requests.exceptions.ConnectionError("timeout")

    with patch("bvbrc._http.requests.get", side_effect=original_error):
        with pytest.raises(RuntimeError) as exc_info:
            make_api_request_with_retries("http://example.com", {})

    assert exc_info.value.__cause__ is original_error
