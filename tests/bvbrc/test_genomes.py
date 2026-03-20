"""
test_genomes.py

Tests de bvbrc/genomes.py:
    - download_genome_fasta (make_api_request_with_retries mockeado)
    - download_multiple_genomes_fasta (make_api_request_with_retries mockeado)
"""

from unittest.mock import MagicMock, patch

import pytest

from bvbrc.genomes import download_genome_fasta, download_multiple_genomes_fasta


# ── Helpers ────────────────────────────────────────────────────────────────────

FAKE_FASTA_CONTENT = ">contig1\nATGCATGCATGC\n>contig2\nGGGCCCAAATTT\n"


def _make_mock_fasta_response(content: str = FAKE_FASTA_CONTENT) -> MagicMock:
    """Crea una respuesta HTTP mockeada con el contenido FASTA dado."""
    mock_response = MagicMock()
    mock_response.text = content
    return mock_response


# ── download_genome_fasta ──────────────────────────────────────────────────────

def test_downloads_fasta_and_saves_file(tmp_path):
    mock_response = _make_mock_fasta_response()

    with patch("bvbrc.genomes.make_api_request_with_retries", return_value=mock_response):
        result = download_genome_fasta("1280.12345", tmp_path)

    expected_file = tmp_path / "1280.12345.fna"
    assert result == expected_file
    assert expected_file.exists()
    assert expected_file.read_text() == FAKE_FASTA_CONTENT


def test_output_filename_uses_genome_id(tmp_path):
    mock_response = _make_mock_fasta_response()

    with patch("bvbrc.genomes.make_api_request_with_retries", return_value=mock_response):
        result = download_genome_fasta("573.99999", tmp_path)

    assert result.name == "573.99999.fna"


def test_skips_download_if_file_already_exists(tmp_path):
    """Si el archivo .fna ya existe en disco, no debe realizarse ningún request HTTP."""
    existing_file = tmp_path / "1280.12345.fna"
    existing_file.write_text(FAKE_FASTA_CONTENT)

    with patch("bvbrc.genomes.make_api_request_with_retries") as mock_request:
        result = download_genome_fasta("1280.12345", tmp_path)

    mock_request.assert_not_called()
    assert result == existing_file


def test_creates_output_directory_if_missing(tmp_path):
    mock_response = _make_mock_fasta_response()
    output_dir = tmp_path / "subdir" / "fasta"

    with patch("bvbrc.genomes.make_api_request_with_retries", return_value=mock_response):
        download_genome_fasta("1280.12345", output_dir)

    assert output_dir.exists()


def test_raises_error_on_empty_fasta_response(tmp_path):
    """Una respuesta vacía indica un genome_id inválido o sin secuencias en BV-BRC."""
    mock_response = _make_mock_fasta_response(content="   ")

    with patch("bvbrc.genomes.make_api_request_with_retries", return_value=mock_response):
        with pytest.raises(RuntimeError, match="no devolvió secuencias FASTA"):
            download_genome_fasta("1280.99999", tmp_path)


def test_does_not_create_file_on_invalid_response(tmp_path):
    """Si la descarga falla, no debe quedar ningún archivo en disco."""
    mock_response = _make_mock_fasta_response(content="")

    with patch("bvbrc.genomes.make_api_request_with_retries", return_value=mock_response):
        with pytest.raises(RuntimeError):
            download_genome_fasta("1280.99999", tmp_path)

    assert not (tmp_path / "1280.99999.fna").exists()


# ── download_multiple_genomes_fasta ────────────────────────────────────────────

def test_downloads_multiple_genomes_successfully(tmp_path):
    mock_response = _make_mock_fasta_response()

    with patch("bvbrc.genomes.make_api_request_with_retries", return_value=mock_response):
        result = download_multiple_genomes_fasta(["1280.1", "1280.2"], tmp_path)

    assert "1280.1" in result
    assert "1280.2" in result
    assert len(result) == 2


def test_continues_downloading_if_one_genome_fails(tmp_path):
    """Un fallo individual no debe detener el resto de las descargas."""
    good_response = _make_mock_fasta_response()
    bad_response = _make_mock_fasta_response(content="")  # vacío → RuntimeError

    with patch(
        "bvbrc.genomes.make_api_request_with_retries",
        side_effect=[good_response, bad_response, good_response],
    ):
        result = download_multiple_genomes_fasta(["1280.1", "1280.2", "1280.3"], tmp_path)

    assert "1280.1" in result
    assert "1280.2" not in result  # este falló
    assert "1280.3" in result


def test_returns_empty_dict_if_all_downloads_fail(tmp_path):
    bad_response = _make_mock_fasta_response(content="")

    with patch("bvbrc.genomes.make_api_request_with_retries", return_value=bad_response):
        result = download_multiple_genomes_fasta(["1280.1", "1280.2"], tmp_path)

    assert result == {}


def test_returns_empty_dict_for_empty_genome_id_list(tmp_path):
    with patch("bvbrc.genomes.make_api_request_with_retries") as mock_request:
        result = download_multiple_genomes_fasta([], tmp_path)

    mock_request.assert_not_called()
    assert result == {}
