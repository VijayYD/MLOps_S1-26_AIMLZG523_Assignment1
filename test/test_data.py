"""Tests for src/heart_mlops/data/{clean,download}.py.

No real network calls -- download.py's HTTP layer is mocked throughout.
"""

from __future__ import annotations

import hashlib
import io
import urllib.error
import zipfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from heart_mlops.config import RAW_COLUMNS
from heart_mlops.data import download as dl
from heart_mlops.data.clean import clean, data_quality_summary, load_combined_raw

# --- clean.py ------------------------------------------------------------


def _raw_df(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    df["site"] = "cleveland"
    return df


def test_question_mark_tokens_become_nan():
    df = _raw_df([[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, "?", 6, 0]])
    cleaned = clean(df)
    assert pd.isna(cleaned.loc[0, "ca"])


def test_implausible_zero_chol_and_trestbps_become_nan():
    df = _raw_df(
        [
            [63, 1, 1, 0, 0, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],  # trestbps=0, chol=0
            [50, 0, 2, 120, 200, 0, 0, 160, 0, 0.0, 1, 1, 3, 0],
        ]
    )
    cleaned = clean(df)
    assert pd.isna(cleaned.loc[0, "trestbps"])
    assert pd.isna(cleaned.loc[0, "chol"])
    # a genuine ca == 0 (not chol/trestbps) must NOT be touched
    assert cleaned.loc[1, "ca"] == 1


def test_target_binarisation():
    df = _raw_df(
        [
            [50, 1, 1, 120, 200, 0, 0, 160, 0, 0.0, 1, 0, 3, 0],
            [60, 1, 1, 120, 200, 0, 0, 160, 0, 0.0, 1, 0, 3, 1],
            [70, 1, 1, 120, 200, 0, 0, 160, 0, 0.0, 1, 0, 3, 4],
        ]
    )
    cleaned = clean(df)
    # Deliberately wrong assertion -- CI red/green demo on a throwaway
    # branch, per the project's Definition of Done. Never merged to main.
    assert cleaned["target"].tolist() == [1, 1, 1]
    assert cleaned["num"].tolist() == [0, 1, 4]


def test_exact_duplicate_rows_dropped():
    row = [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0]
    df = _raw_df([row, row, [50, 0, 2, 120, 200, 0, 0, 160, 0, 0.0, 1, 1, 3, 0]])
    cleaned = clean(df)
    assert len(cleaned) == 2


def test_site_column_preserved_and_dtypes_correct():
    df = _raw_df([[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0]])
    cleaned = clean(df)
    assert cleaned.loc[0, "site"] == "cleveland"
    assert cleaned["age"].dtype == np.float64
    assert str(cleaned["ca"].dtype) == "Int64"


def test_data_quality_summary_has_one_row_per_site_plus_all():
    df = _raw_df([[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0]])
    cleaned = clean(df)
    summary = data_quality_summary(cleaned)
    assert set(summary["site"]) == {"cleveland", "ALL"}


def test_load_combined_raw_reads_all_four_sites(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "DATA_RAW_DIR", tmp_path)
    import heart_mlops.data.clean as clean_module

    monkeypatch.setattr(clean_module, "DATA_RAW_DIR", tmp_path)

    row = "63,1,1,145,233,1,2,150,0,2.3,3,0,6,0\n"
    for site in ["cleveland", "hungarian", "switzerland", "va"]:
        (tmp_path / f"processed.{site}.data").write_text(row)

    combined = load_combined_raw()
    assert len(combined) == 4
    assert set(combined["site"]) == {"cleveland", "hungarian", "switzerland", "va"}


# --- download.py -----------------------------------------------------------


def _fake_response(content: bytes):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return content

    return _Resp()


def test_download_site_success(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "DATA_RAW_DIR", tmp_path)
    content = b"63,1,1,145,233,1,2,150,0,2.3,3,0,6,0\n"
    checksum = hashlib.sha256(content).hexdigest()
    monkeypatch.setitem(dl.RAW_FILE_SHA256, "cleveland", checksum)

    with patch("urllib.request.urlopen", return_value=_fake_response(content)):
        path = dl.download_site("cleveland")

    assert path.read_bytes() == content


def test_download_site_idempotent_skips_valid_cached_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "DATA_RAW_DIR", tmp_path)
    content = b"63,1,1,145,233,1,2,150,0,2.3,3,0,6,0\n"
    checksum = hashlib.sha256(content).hexdigest()
    monkeypatch.setitem(dl.RAW_FILE_SHA256, "cleveland", checksum)
    (tmp_path / "processed.cleveland.data").write_bytes(content)

    with patch("urllib.request.urlopen") as mock_urlopen:
        dl.download_site("cleveland")
    mock_urlopen.assert_not_called()


def test_download_site_falls_back_to_zip_on_primary_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "DATA_RAW_DIR", tmp_path)
    monkeypatch.setattr(dl, "MAX_RETRIES", 1)
    monkeypatch.setattr(dl, "RETRY_BACKOFF_SECONDS", 0)
    content = b"63,1,1,145,233,1,2,150,0,2.3,3,0,6,0\n"
    checksum = hashlib.sha256(content).hexdigest()
    monkeypatch.setitem(dl.RAW_FILE_SHA256, "cleveland", checksum)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("processed.cleveland.data", content)
    zip_bytes = zip_buffer.getvalue()

    def fake_urlopen(url, timeout=None):
        if url.endswith(".zip"):
            return _fake_response(zip_bytes)
        raise urllib.error.URLError("primary source down")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        path = dl.download_site("cleveland")

    assert path.read_bytes() == content


def test_download_site_raises_when_both_sources_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "DATA_RAW_DIR", tmp_path)
    monkeypatch.setattr(dl, "MAX_RETRIES", 1)

    with (
        patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")),
        pytest.raises(RuntimeError, match="both the primary URL"),
    ):
        dl.download_site("cleveland")


def test_download_site_rejects_checksum_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "DATA_RAW_DIR", tmp_path)
    monkeypatch.setitem(dl.RAW_FILE_SHA256, "cleveland", "0" * 64)
    content = b"not the expected content"

    with (
        pytest.raises(RuntimeError, match="checksum mismatch"),
        patch("urllib.request.urlopen", return_value=_fake_response(content)),
    ):
        dl.download_site("cleveland")


def test_main_exits_nonzero_when_download_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dl, "DATA_RAW_DIR", tmp_path)
    monkeypatch.setattr(dl, "MAX_RETRIES", 1)
    monkeypatch.setattr("sys.argv", ["download.py"])

    with (
        patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")),
        pytest.raises(SystemExit) as exc_info,
    ):
        dl.main()

    assert exc_info.value.code == 1
