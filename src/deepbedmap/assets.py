"""Verified, resumable access to the public DeepBedMap release assets."""

from __future__ import annotations

import hashlib
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Asset:
    name: str
    size: int
    sha256: str | None
    md5: str
    urls: tuple[str, ...]
    description: str


WEIGHTS = Asset(
    name="srgan_generator_model_weights.npz",
    size=33_225_666,
    sha256="59c604294be8fd4931cf9f224ffad9b0e6416f533b380f712f974d73a0931726",
    md5="8464a631e7cea64b1ae02fb84bf60bb6",
    urls=(
        "https://github.com/weiji14/deepbedmap/releases/download/v1.1.0/"
        "srgan_generator_model_weights.npz",
        "https://zenodo.org/api/records/4054246/files/srgan_generator_model_weights.npz/content",
    ),
    description="Published DeepBedMap v1.1 Chainer generator weights",
)

DEM = Asset(
    name="deepbedmap_dem.tif",
    size=495_739_549,
    sha256=None,
    md5="c295e1e54daa78fef46361188b988619",
    urls=(
        "https://github.com/weiji14/deepbedmap/releases/download/v1.1.0/deepbedmap_dem.tif",
        "https://zenodo.org/api/records/4054246/files/deepbedmap_dem.tif/content",
    ),
    description="Published 250 m Antarctic DeepBedMap GeoTIFF",
)

ASSETS = {"weights": WEIGHTS, "dem": DEM}


def cache_dir() -> Path:
    configured = os.environ.get("DEEPBEDMAP_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache" / "deepbedmap").resolve()


def cached_path(asset: Asset) -> Path:
    return cache_dir() / asset.name


def file_digest(path: str | Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset(path: str | Path, asset: Asset) -> tuple[bool, str]:
    path = Path(path)
    if not path.is_file():
        return False, "file does not exist"
    size = path.stat().st_size
    if size != asset.size:
        return False, f"size is {size:,} bytes; expected {asset.size:,}"
    if asset.sha256:
        actual = file_digest(path, "sha256")
        if actual.lower() != asset.sha256.lower():
            return False, f"SHA256 is {actual}; expected {asset.sha256}"
    else:
        actual = file_digest(path, "md5")
        if actual.lower() != asset.md5.lower():
            return False, f"MD5 is {actual}; expected {asset.md5}"
    return True, "verified"


def _download_once(url: str, partial: Path, expected_size: int, timeout: float) -> None:
    current_size = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "DeepBedMap/2.0 (+https://github.com/weiji14/deepbedmap)"}
    if current_size:
        headers["Range"] = f"bytes={current_size}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        resume = current_size > 0 and getattr(response, "status", 200) == 206
        mode = "ab" if resume else "wb"
        received_size = current_size if resume else 0
        with partial.open(mode) as output:
            while chunk := response.read(1024 * 1024):
                received_size += len(chunk)
                if received_size > expected_size:
                    raise OSError(
                        "Download exceeded the declared asset size "
                        f"({received_size:,} > {expected_size:,} bytes)"
                    )
                output.write(chunk)
    if partial.stat().st_size > expected_size:
        raise OSError(
            f"Downloaded file is larger than expected ({partial.stat().st_size:,} > "
            f"{expected_size:,})"
        )


def _promote_complete_partial(
    partial: Path,
    destination: Path,
    asset: Asset,
    *,
    discard_invalid: bool,
) -> tuple[bool, str | None]:
    """Atomically promote a complete verified partial download, if present."""

    if not partial.is_file() or partial.stat().st_size < asset.size:
        return False, None
    valid, reason = verify_asset(partial, asset)
    if valid:
        os.replace(partial, destination)
        return True, None
    if discard_invalid:
        partial.unlink()
    return False, reason


def download_asset(
    asset: Asset,
    destination: str | Path | None = None,
    *,
    retries: int = 4,
    timeout: float = 60.0,
    force: bool = False,
    offline: bool = False,
) -> Path:
    """Download and verify an asset, retaining ``.part`` files for resume."""

    destination = Path(destination) if destination is not None else cached_path(asset)
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    if destination.exists() and not force:
        valid, reason = verify_asset(destination, asset)
        if valid:
            return destination
        if offline:
            raise OSError(f"Cached asset is invalid: {reason}")
        corrupt = destination.with_name(f"{destination.name}.corrupt")
        os.replace(destination, corrupt)
    elif destination.exists():
        destination.unlink()
    if force and partial.exists():
        partial.unlink()

    promoted, partial_error = _promote_complete_partial(
        partial,
        destination,
        asset,
        discard_invalid=not offline,
    )
    if promoted:
        return destination
    if offline:
        if partial_error:
            raise OSError(f"Cached partial asset is invalid: {partial_error}")
        raise FileNotFoundError(f"Asset is not cached and offline mode is enabled: {destination}")

    errors: list[str] = []
    for attempt in range(retries + 1):
        url = asset.urls[attempt % len(asset.urls)]
        try:
            _download_once(url, partial, asset.size, timeout)
            promoted, partial_error = _promote_complete_partial(
                partial,
                destination,
                asset,
                discard_invalid=True,
            )
            if promoted:
                return destination
            if partial_error:
                raise OSError(f"downloaded file failed verification: {partial_error}")
            current_size = partial.stat().st_size if partial.exists() else 0
            raise OSError(f"incomplete download ({current_size:,}/{asset.size:,} bytes)")
        except (OSError, urllib.error.URLError) as error:
            errors.append(f"{url}: {error}")
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise OSError("Unable to download a verified asset:\n" + "\n".join(errors))
