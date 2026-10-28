"""Safely download and extract NAS-Bench-301 XGBoost surrogate models.

The archive SHA-256 digest MUST be obtained and verified independently from a
trusted source. We deliberately do not hard-code an unverified checksum.
Usage:
    python examples/download_nb301_models.py --sha256 VERIFIED_64_HEX_DIGEST
    python examples/download_nb301_models.py --archive /path/to/file.zip \\
        --sha256 VERIFIED_64_HEX_DIGEST --dest /path/to/models

Nothing is downloaded or extracted when this module is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import stat
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_URL = "https://ndownloader.figshare.com/files/24992018"
DEFAULT_DESTINATION = Path("~/nb301_models").expanduser()
MAX_ARCHIVE_BYTES = 3 * 1024**3
MAX_UNCOMPRESSED_BYTES = 12 * 1024**3
MAX_MEMBER_BYTES = 4 * 1024**3
MAX_SELECTED_FILES = 5000
MAX_ARCHIVE_ENTRIES = 10000
MAX_COMPRESSION_RATIO = 1000.0
BUFFER_SIZE = 1024 * 1024


def _validate_digest(expected_sha256: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("sha256 must be a 64-character hexadecimal digest from a trusted source")
    return expected_sha256.lower()


def _sha256_file(path: Path, *, max_bytes: int) -> str:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        while chunk := stream.read(BUFFER_SIZE):
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("Archive exceeds maximum allowed download size")
            digest.update(chunk)
    return digest.hexdigest()


class _HTTPSOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse any redirect to an unencrypted transport before following it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise ValueError("Model download redirect must use HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_archive(
    url: str, destination: Path, *, expected_sha256: str,
    max_bytes: int = MAX_ARCHIVE_BYTES,
) -> None:
    """Download to a temporary location; reject HTTP and SHA mismatches."""
    expected = _validate_digest(expected_sha256)
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError("Model downloads must use HTTPS")

    digest = hashlib.sha256()
    total = 0
    created_output = False
    try:
        with urllib.request.build_opener(_HTTPSOnlyRedirectHandler()).open(url, timeout=60) as response:
            if urllib.parse.urlsplit(response.geturl()).scheme != "https":
                raise ValueError("Model download redirected to a non-HTTPS URL")
            with destination.open("xb") as output:
                created_output = True
                while chunk := response.read(BUFFER_SIZE):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("Archive exceeds maximum allowed download size")
                    digest.update(chunk)
                    output.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("Model archive SHA-256 mismatch; refusing to extract")
    except Exception:
        # Never delete a pre-existing file if exclusive creation failed.
        if created_output:
            destination.unlink(missing_ok=True)
        raise


def _safe_relative_path(info: zipfile.ZipInfo) -> Path:
    """Reject path traversal, Windows paths, links, and special entries."""
    raw = info.filename
    if not raw or chr(92) in raw or chr(0) in raw or raw.startswith("/"):
        raise ValueError(f"Unsafe archive member path: {raw!r}")

    parts = raw.rstrip("/").split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Unsafe archive member path: {raw!r}")
    if re.match(r"^[A-Za-z]:", parts[0]):
        raise ValueError(f"Unsafe archive member path: {raw!r}")

    file_type = stat.S_IFMT(info.external_attr >> 16)
    if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise ValueError(f"Link or unsupported archive entry: {raw!r}")
    return Path(*parts)


def _is_selected(relpath: Path) -> bool:
    # Retain the same XGBoost-family selection as the historical downloader,
    # plus the two supported model configuration filenames.
    return (
        "xgb" in relpath.as_posix().lower()
        or relpath.name in {"data_config.json", "model_config.json"}
    )


def extract_models(
    archive_path: Path, destination: Path, *,
    max_total_bytes: int = MAX_UNCOMPRESSED_BYTES,
    max_member_bytes: int = MAX_MEMBER_BYTES,
    max_files: int = MAX_SELECTED_FILES,
    max_archive_entries: int = MAX_ARCHIVE_ENTRIES,
    max_compression_ratio: float = MAX_COMPRESSION_RATIO,
) -> int:
    """Extract selected regular files into a NEW staging directory.

    Resource limits use both ZIP header sizes and actual decompressed bytes.
    Do not supply an existing destination with important files.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    with zipfile.ZipFile(archive_path) as archive:
        if len(archive.infolist()) > max_archive_entries:
            raise ValueError("Archive contains too many entries")
        chosen: list[tuple[zipfile.ZipInfo, Path]] = []
        seen: set[Path] = set()
        claimed_bytes = 0

        for info in archive.infolist():
            relpath = _safe_relative_path(info)
            if info.is_dir() or not _is_selected(relpath):
                continue
            if relpath in seen:
                raise ValueError(f"Duplicate archive entry: {relpath}")
            seen.add(relpath)
            if len(chosen) >= max_files:
                raise ValueError("Archive exceeds maximum selected file count")
            if info.file_size > max_member_bytes:
                raise ValueError(f"Archive member exceeds maximum size: {relpath}")
            claimed_bytes += info.file_size
            if claimed_bytes > max_total_bytes:
                raise ValueError("Archive exceeds total uncompressed size limit")
            if info.file_size / max(info.compress_size, 1) > max_compression_ratio:
                raise ValueError(f"Suspicious archive compression ratio: {relpath}")
            chosen.append((info, relpath))

        if not chosen:
            raise ValueError("Archive contains no selected XGBoost models or configs")

        destination.mkdir(parents=True, exist_ok=False)
        try:
            extracted_total = 0
            for info, relpath in chosen:
                target = destination / relpath
                target.parent.mkdir(parents=True, exist_ok=True)
                actual_size = 0
                with archive.open(info) as src, target.open("xb") as dst:
                    while chunk := src.read(BUFFER_SIZE):
                        actual_size += len(chunk)
                        extracted_total += len(chunk)
                        if actual_size > max_member_bytes or extracted_total > max_total_bytes:
                            raise ValueError("Decompressed file exceeds extraction limits")
                        dst.write(chunk)
            return len(chosen)
        except Exception:
            shutil.rmtree(destination)
            raise


def install_from_archive(
    archive_path: Path, destination: Path, *,
    expected_sha256: str,
    max_archive_bytes: int = MAX_ARCHIVE_BYTES,
    **extract_limits: int | float,
) -> int:
    """Verify checksum, safely extract, then atomically move staged models."""
    archive_path = Path(archive_path)
    destination = Path(destination).expanduser()
    expected = _validate_digest(expected_sha256)
    if _sha256_file(archive_path, max_bytes=max_archive_bytes) != expected:
        raise ValueError("Model archive SHA-256 mismatch; refusing to extract")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".nb301-staging-", dir=parent) as tmp:
        stage = Path(tmp) / "models"
        extracted = extract_models(archive_path, stage, **extract_limits)
        if destination.exists():
            raise FileExistsError(f"Destination already exists: {destination}")
        os.rename(stage, destination)
    return extracted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sha256", required=True,
        help="Verified 64-character SHA-256 from a trusted source (required)",
    )
    parser.add_argument("--archive", type=Path, help="Use an already-downloaded ZIP; skip network")
    parser.add_argument("--url", default=DEFAULT_URL, help="HTTPS download URL")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args(argv)

    try:
        expected = _validate_digest(args.sha256)
        if args.archive is not None:
            count = install_from_archive(
                args.archive, args.dest, expected_sha256=expected,
            )
        else:
            with tempfile.TemporaryDirectory(prefix="nb301-download-") as tmp:
                archive_path = Path(tmp) / "models.zip"
                download_archive(args.url, archive_path, expected_sha256=expected)
                count = install_from_archive(
                    archive_path, args.dest, expected_sha256=expected,
                )
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Model installation failed: {exc}\\n")

    print(f"Verified and extracted {count} model/config files to {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
