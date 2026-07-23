from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from deepbedmap.assets import Asset, download_asset, verify_asset


def _asset_for(path: Path) -> Asset:
    content = path.read_bytes()
    return Asset(
        name="fixture.bin",
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        md5=hashlib.md5(content).hexdigest(),  # noqa: S324 - release compatibility metadata
        urls=(path.as_uri(),),
        description="test fixture",
    )


def test_download_and_verify_local_asset(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"verified DeepBedMap fixture")
    asset = _asset_for(source)
    destination = tmp_path / "cache" / asset.name
    assert download_asset(asset, destination, retries=0) == destination.resolve()
    assert verify_asset(destination, asset) == (True, "verified")
    assert download_asset(asset, destination, offline=True) == destination.resolve()


def test_offline_missing_asset_is_clear(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"fixture")
    asset = _asset_for(source)
    with pytest.raises(FileNotFoundError, match="offline mode"):
        download_asset(asset, tmp_path / "missing.bin", offline=True)


def test_complete_partial_is_verified_and_promoted_without_network(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"completed before the previous process exited")
    asset = _asset_for(source)
    destination = tmp_path / "cache" / asset.name
    destination.parent.mkdir()
    partial = destination.with_name(f"{destination.name}.part")
    partial.write_bytes(source.read_bytes())
    source.unlink()

    assert download_asset(asset, destination, retries=0) == destination.resolve()
    assert verify_asset(destination, asset) == (True, "verified")
    assert not partial.exists()


def test_invalid_full_partial_is_discarded_before_retry(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"known good payload")
    asset = _asset_for(source)
    destination = tmp_path / "cache" / asset.name
    destination.parent.mkdir()
    partial = destination.with_name(f"{destination.name}.part")
    partial.write_bytes(b"x" * asset.size)

    assert download_asset(asset, destination, retries=0) == destination.resolve()
    assert destination.read_bytes() == source.read_bytes()


def test_oversized_response_is_stopped_before_writing_payload(tmp_path: Path) -> None:
    source = tmp_path / "oversized.bin"
    source.write_bytes(b"x" * (1024 * 1024))
    expected = b"tiny"
    asset = Asset(
        name="bounded.bin",
        size=len(expected),
        sha256=hashlib.sha256(expected).hexdigest(),
        md5=hashlib.md5(expected).hexdigest(),  # noqa: S324 - release metadata shape
        urls=(source.as_uri(),),
        description="bounded fixture",
    )
    destination = tmp_path / asset.name

    with pytest.raises(OSError, match="exceeded the declared asset size"):
        download_asset(asset, destination, retries=0)
    partial = destination.with_name(f"{destination.name}.part")
    assert partial.stat().st_size <= asset.size
