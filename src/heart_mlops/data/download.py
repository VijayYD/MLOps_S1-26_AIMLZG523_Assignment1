"""Download the four UCI Heart Disease source files (Cleveland, Hungarian,
Switzerland, VA Long Beach).

Usage:
    python -m heart_mlops.data.download [--force]

Primary source: the UCI machine-learning-databases/heart-disease/ directory,
one file per site. Fallback (used only if a primary URL fails): the UCI
static zip for dataset id 45, which bundles byte-identical copies of the
same four files. If both sources fail for a site, this exits non-zero with
a clear message -- it never fabricates data.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from heart_mlops.config import (
    DATA_RAW_DIR,
    RAW_FILE_SHA256,
    SITES,
    UCI_BASE_URL,
    UCI_FALLBACK_ZIP_URL,
)

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 20
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_with_retries(url: str, dest: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
                dest.write_bytes(response.read())
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, MAX_RETRIES, url, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"failed to download {url} after {MAX_RETRIES} attempts") from last_error


def _download_from_fallback_zip(site: str, dest: Path) -> None:
    tmp_zip = DATA_RAW_DIR / "_fallback_heart_disease.zip"
    if not tmp_zip.exists():
        _download_with_retries(UCI_FALLBACK_ZIP_URL, tmp_zip)
    member = f"processed.{site}.data"
    with zipfile.ZipFile(tmp_zip) as zf, zf.open(member) as src, dest.open("wb") as out:
        shutil.copyfileobj(src, out)


def download_site(site: str, force: bool = False) -> Path:
    dest = DATA_RAW_DIR / f"processed.{site}.data"
    expected_checksum = RAW_FILE_SHA256[site]

    if dest.exists() and not force:
        if _sha256(dest) == expected_checksum:
            logger.info("%s already present and checksum-valid, skipping", dest.name)
            return dest
        logger.warning("%s present but checksum mismatch, re-downloading", dest.name)

    primary_url = f"{UCI_BASE_URL}/processed.{site}.data"
    try:
        logger.info("Downloading %s from %s", site, primary_url)
        _download_with_retries(primary_url, dest)
    except RuntimeError as primary_error:
        logger.warning(
            "Primary source failed for %s (%s); trying fallback zip", site, primary_error
        )
        try:
            _download_from_fallback_zip(site, dest)
        except Exception as fallback_error:
            raise RuntimeError(
                f"both the primary URL ({primary_url}) and the fallback zip "
                f"({UCI_FALLBACK_ZIP_URL}) failed for site '{site}'. "
                f"Primary error: {primary_error}. Fallback error: {fallback_error}"
            ) from fallback_error

    actual_checksum = _sha256(dest)
    if actual_checksum != expected_checksum:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"checksum mismatch for site '{site}': expected {expected_checksum}, "
            f"got {actual_checksum}. The UCI source file may have changed; "
            "refusing to proceed with unexpected data."
        )
    logger.info("%s downloaded and checksum-verified", dest.name)
    return dest


def download_all(force: bool = False) -> list[Path]:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = [download_site(site, force=force) for site in SITES]
    (DATA_RAW_DIR / "_fallback_heart_disease.zip").unlink(missing_ok=True)
    return paths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if a valid cached copy already exists",
    )
    args = parser.parse_args()

    try:
        paths = download_all(force=args.force)
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)

    for path in paths:
        logger.info("OK: %s", path)


if __name__ == "__main__":
    main()
