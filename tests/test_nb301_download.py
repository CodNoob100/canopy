"""Offline security regressions for the NAS-Bench-301 model downloader."""

import hashlib
import io
import stat
import zipfile
from pathlib import Path

import pytest

from examples.download_nb301_models import (
    _safe_relative_path,
    download_archive,
    extract_models,
    install_from_archive,
)


def make_zip(path: Path, members):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, contents in members:
            archive.writestr(name, contents)
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_valid_archive_extracts_only_expected_models(tmp_path):
    archive = make_zip(tmp_path / "models.zip", [
        ("xgb_v1/surrogate.model", b"model"),
        ("xgb_v1/data_config.json", b'{"ok": true}'),
        ("other/unrelated.bin", b"ignored"),
    ])
    target = tmp_path / "installed"
    count = install_from_archive(archive, target, expected_sha256=sha256(archive))

    assert count == 2
    assert (target / "xgb_v1" / "surrogate.model").read_bytes() == b"model"
    assert not (target / "other" / "unrelated.bin").exists()


def test_bad_digest_prevents_extraction(tmp_path):
    archive = make_zip(tmp_path / "models.zip", [("xgb/model.bin", b"m")])
    target = tmp_path / "installed"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        install_from_archive(archive, target, expected_sha256="0" * 64)
    assert not target.exists()


@pytest.mark.parametrize("unsafe", [
    "../xgb/escape", "/xgb/absolute", "C:/xgb/windows",
    "xgb/../escape", "xgb\\\\escape", "./xgb/escape",
])
def test_unsafe_zip_paths_are_rejected(tmp_path, unsafe):
    archive = make_zip(tmp_path / "unsafe.zip", [(unsafe, b"attack")])
    target = tmp_path / "installed"
    with pytest.raises(ValueError, match="Unsafe archive member"):
        install_from_archive(archive, target, expected_sha256=sha256(archive))
    assert not target.exists()


def test_zip_symlinks_are_rejected(tmp_path):
    info = zipfile.ZipInfo("xgb/model_link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = tmp_path / "symlink.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(info, "target")
    with pytest.raises(ValueError, match="Link or unsupported"):
        install_from_archive(
            archive, tmp_path / "installed", expected_sha256=sha256(archive)
        )


def test_duplicate_zip_names_are_rejected(tmp_path):
    archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("xgb/model", b"1")
        with pytest.warns(UserWarning):
            out.writestr("xgb/model", b"2")
    with pytest.raises(ValueError, match="Duplicate"):
        install_from_archive(
            archive, tmp_path / "installed", expected_sha256=sha256(archive)
        )


def test_selected_member_size_limit(tmp_path):
    archive = make_zip(tmp_path / "large.zip", [("xgb/model", b"A" * 50)])
    with pytest.raises(ValueError, match="maximum size"):
        install_from_archive(
            archive, tmp_path / "installed",
            expected_sha256=sha256(archive), max_member_bytes=10
        )


def test_total_size_limit(tmp_path):
    archive = make_zip(tmp_path / "many.zip", [
        ("xgb/a", b"12345"), ("xgb/b", b"67890"),
    ])
    with pytest.raises(ValueError, match="total uncompressed"):
        install_from_archive(
            archive, tmp_path / "installed",
            expected_sha256=sha256(archive), max_total_bytes=7,
        )


def test_file_count_limit(tmp_path):
    archive = make_zip(tmp_path / "many.zip", [
        ("xgb/a", b"1"), ("xgb/b", b"2"),
    ])
    with pytest.raises(ValueError, match="maximum selected"):
        install_from_archive(
            archive, tmp_path / "installed",
            expected_sha256=sha256(archive), max_files=1,
        )


def test_existing_destination_is_not_modified(tmp_path):
    archive = make_zip(tmp_path / "models.zip", [("xgb/model", b"new")])
    target = tmp_path / "installed"
    target.mkdir()
    (target / "important.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        install_from_archive(archive, target, expected_sha256=sha256(archive))
    assert (target / "important.txt").read_text() == "keep"


def test_unverified_digest_format_is_rejected(tmp_path):
    archive = make_zip(tmp_path / "models.zip", [("xgb/model", b"data")])
    with pytest.raises(ValueError, match="64-character"):
        install_from_archive(archive, tmp_path / "installed", expected_sha256="unknown")


def test_plain_http_download_is_rejected_before_any_network_access(tmp_path):
    with pytest.raises(ValueError, match="HTTPS"):
        download_archive(
            "http://example.test/archive.zip", tmp_path / "models.zip",
            expected_sha256="0" * 64,
        )
    assert not (tmp_path / "models.zip").exists()


def test_malformed_archive_does_not_create_destination(tmp_path):
    archive = tmp_path / "invalid.zip"
    archive.write_bytes(b"not a zip")
    with pytest.raises(zipfile.BadZipFile):
        install_from_archive(
            archive, tmp_path / "installed", expected_sha256=sha256(archive),
        )
    assert not (tmp_path / "installed").exists()
