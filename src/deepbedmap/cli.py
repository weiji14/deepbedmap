"""Command-line interface for the modern DeepBedMap runtime."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from . import __version__
from .assets import (
    ASSETS,
    DEM,
    WEIGHTS,
    cache_dir,
    cached_path,
    download_asset,
    file_digest,
    verify_asset,
)

NATIVE_WEIGHTS_NAME = "deepbedmap_generator_v1.1.0.pt"


def native_weights_path() -> Path:
    return cache_dir() / NATIVE_WEIGHTS_NAME


def _load_model(path: Path):
    from .weights import load_legacy_weights, load_native_checkpoint

    if path.suffix.lower() == ".npz":
        source_sha256 = file_digest(path, "sha256")
        return load_legacy_weights(path), {
            "format": "legacy-chainer-npz",
            "model": "DeepBedMap Generator v1.1.0",
            "residual_scaling": 0.2,
            "source_sha256": source_sha256,
            "weights_release": "v1.1.0" if source_sha256 == WEIGHTS.sha256 else "custom",
        }
    model, metadata = load_native_checkpoint(path)
    return model, metadata


def _find_weights(explicit: str | None, *, required: bool) -> Path | None:
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Explicit model weights do not exist: {candidate}")

    candidates = (native_weights_path(), cached_path(WEIGHTS))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if required:
        raise FileNotFoundError(
            "No model weights found. Run '.\\run.ps1 fetch-weights' or pass --weights."
        )
    return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "deepbedmap": __version__,
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "rasterio": rasterio.__version__,
        "gdal": rasterio.__gdal_version__,
        "cache": str(cache_dir()),
        "legacy_weights": str(cached_path(WEIGHTS)),
        "legacy_weights_present": cached_path(WEIGHTS).is_file(),
        "native_weights": str(native_weights_path()),
        "native_weights_present": native_weights_path().is_file(),
    }
    try:
        import torch
        import torchvision

        report.update(
            {
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            }
        )
    except (ImportError, OSError, RuntimeError) as error:
        report.update({"torch": None, "cuda_available": False, "torch_error": str(error)})
    return report


def _print_report(report: dict[str, Any]) -> None:
    width = max(len(key) for key in report)
    for key, value in report.items():
        print(f"{key.replace('_', ' ').title():<{width + 2}} {value}")


def command_doctor(args: argparse.Namespace) -> int:
    report = runtime_report()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_report(report)
        if not report["native_weights_present"] and not report["legacy_weights_present"]:
            print("\nWeights are optional for the synthetic smoke test; fetch them for prediction.")
    return 0 if report.get("torch") else 1


def command_list_assets(args: argparse.Namespace) -> int:
    rows = []
    for name, asset in ASSETS.items():
        path = cached_path(asset)
        valid, reason = verify_asset(path, asset)
        rows.append(
            {
                "asset": name,
                "size_bytes": asset.size,
                "path": str(path),
                "valid": valid,
                "status": reason,
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        for row in rows:
            print(
                f"{row['asset']:<8} {row['size_bytes'] / 1024**2:8.1f} MiB  "
                f"{'ready' if row['valid'] else row['status']}\n         {row['path']}"
            )
    return 0


def command_fetch_weights(args: argparse.Namespace) -> int:
    destination = Path(args.output).expanduser() if args.output else cached_path(WEIGHTS)
    print(f"Fetching {WEIGHTS.description} ...")
    path = download_asset(
        WEIGHTS,
        destination,
        force=args.force,
        offline=args.offline,
        retries=args.retries,
    )
    print(f"Verified: {path}")
    if not args.no_convert:
        from .weights import convert_legacy_checkpoint

        native = (
            Path(args.native_output).expanduser() if args.native_output else native_weights_path()
        )
        print("Converting Chainer arrays to a metadata-rich PyTorch checkpoint ...")
        convert_legacy_checkpoint(path, native)
        print(f"Ready: {native.resolve()}")
    return 0


def command_fetch_dem(args: argparse.Namespace) -> int:
    destination = Path(args.output).expanduser() if args.output else cached_path(DEM)
    print(f"Fetching {DEM.description} ({DEM.size / 1024**2:.1f} MiB) ...")
    path = download_asset(
        DEM,
        destination,
        force=args.force,
        offline=args.offline,
        retries=args.retries,
    )
    print(f"Verified: {path}")
    return 0


def command_convert_weights(args: argparse.Namespace) -> int:
    from .weights import convert_legacy_checkpoint, inspect_legacy_checkpoint

    source = Path(args.source).expanduser() if args.source else cached_path(WEIGHTS)
    destination = Path(args.output).expanduser() if args.output else native_weights_path()
    info = inspect_legacy_checkpoint(source)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    path = convert_legacy_checkpoint(source, destination)
    print(f"Native checkpoint: {path.resolve()}")
    return 0


def _synthetic_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {
        "x": rng.normal(size=(1, 1, 11, 11)).astype(np.float32),
        "w1": rng.normal(size=(1, 1, 110, 110)).astype(np.float32),
        "w2": rng.normal(size=(1, 2, 22, 22)).astype(np.float32),
        "w3": rng.normal(size=(1, 1, 11, 11)).astype(np.float32),
    }


def command_smoke(args: argparse.Namespace) -> int:
    from .inference import predict_arrays
    from .model import GeneratorModel, parameter_count

    weights = None if args.random else _find_weights(args.weights, required=False)
    if weights:
        model, metadata = _load_model(weights)
        source = str(weights)
    else:
        model = GeneratorModel()
        metadata = {"format": "random-initialization", "residual_scaling": 0.2}
        source = "random initialization (offline architecture smoke)"
    started = time.perf_counter()
    output, device = predict_arrays(model, _synthetic_inputs(), device=args.device)
    elapsed = time.perf_counter() - started
    result = {
        "status": "ok",
        "device": str(device),
        "weights": source,
        "checkpoint_format": metadata.get("format"),
        "checkpoint_sha256": file_digest(weights, "sha256") if weights else None,
        "parameters": parameter_count(model),
        "output_shape": list(output.shape),
        "output_min": float(output.min()),
        "output_max": float(output.max()),
        "output_mean": float(output.mean()),
        "all_finite": bool(np.isfinite(output).all()),
        "elapsed_seconds": round(elapsed, 3),
    }
    if result["output_shape"] != [1, 1, 36, 36] or not result["all_finite"]:
        raise RuntimeError(f"Smoke-test invariant failed: {result}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def command_predict(args: argparse.Namespace) -> int:
    from .inference import predict_arrays
    from .io import ModelInputPaths, prepare_model_inputs, write_geotiff

    weights = _find_weights(args.weights, required=True)
    assert weights is not None
    model, metadata = _load_model(weights)
    bounds = tuple(float(value) for value in args.bounds)
    paths = ModelInputPaths(
        bed=args.bed,
        surface=args.surface,
        velocity_x=args.velocity_x,
        velocity_y=args.velocity_y,
        accumulation=args.accumulation,
    )
    inputs = prepare_model_inputs(
        paths,
        bounds,  # type: ignore[arg-type]
        bed_fill=args.bed_fill,
        surface_fill=args.surface_fill,
        velocity_fill=args.velocity_fill,
        accumulation_fill=args.accumulation_fill,
    )
    output, device = predict_arrays(model, inputs, device=args.device)
    checkpoint_sha256 = file_digest(weights, "sha256")
    destination = write_geotiff(  # type: ignore[arg-type]
        args.output,
        output,
        bounds,
        tags={
            "CHECKPOINT_FORMAT": str(metadata.get("format", "unknown")),
            "CHECKPOINT_SHA256": checkpoint_sha256,
            "WEIGHTS_RELEASE": str(metadata.get("weights_release", "custom")),
        },
    )
    result = {
        "status": "ok",
        "output": str(destination),
        "shape": list(output.shape),
        "bounds": list(bounds),
        "crs": "EPSG:3031",
        "resolution_metres": 250,
        "device": str(device),
        "weights": str(weights),
        "checkpoint_format": metadata.get("format"),
        "checkpoint_sha256": checkpoint_sha256,
        "weights_release": metadata.get("weights_release", "custom"),
        "residual_scaling": metadata.get("residual_scaling", 0.2),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _download_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", help="Override the project cache destination")
    parser.add_argument("--force", action="store_true", help="Replace an existing cached file")
    parser.add_argument("--offline", action="store_true", help="Verify cache without networking")
    parser.add_argument("--retries", type=int, default=4, help="Retry count (default: 4)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepbedmap",
        description="Modern Windows-native DeepBedMap inference runtime",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Inspect the local runtime and GPU")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(handler=command_doctor)

    assets = subparsers.add_parser("list-assets", help="Show cached public assets")
    assets.add_argument("--json", action="store_true")
    assets.set_defaults(handler=command_list_assets)

    fetch_weights = subparsers.add_parser(
        "fetch-weights", help="Download, verify, and convert the public v1.1 weights"
    )
    _download_arguments(fetch_weights)
    fetch_weights.add_argument("--no-convert", action="store_true")
    fetch_weights.add_argument("--native-output")
    fetch_weights.set_defaults(handler=command_fetch_weights)

    fetch_dem = subparsers.add_parser("fetch-dem", help="Download the published 250 m DEM")
    _download_arguments(fetch_dem)
    fetch_dem.set_defaults(handler=command_fetch_dem)

    convert = subparsers.add_parser(
        "convert-weights", help="Convert a legacy Chainer NPZ to native PyTorch"
    )
    convert.add_argument("source", nargs="?")
    convert.add_argument("--output")
    convert.set_defaults(handler=command_convert_weights)

    smoke = subparsers.add_parser("smoke", help="Run an offline synthetic model forward pass")
    smoke.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    smoke.add_argument("--weights")
    smoke.add_argument(
        "--random", action="store_true", help="Ignore cached weights and test architecture only"
    )
    smoke.set_defaults(handler=command_smoke)

    predict = subparsers.add_parser(
        "predict", help="Predict one EPSG:3031 region from local rasters"
    )
    predict.add_argument("--bed", required=True, help="BEDMAP2 bed raster (1 km)")
    predict.add_argument("--surface", required=True, help="REMA surface raster (100 m)")
    predict.add_argument("--velocity-x", required=True, help="Signed X velocity raster")
    predict.add_argument("--velocity-y", required=True, help="Signed Y velocity raster")
    predict.add_argument("--accumulation", required=True, help="Accumulation raster (1 km)")
    predict.add_argument(
        "--bounds",
        required=True,
        nargs=4,
        type=float,
        metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
        help="Output bounds in EPSG:3031; width and height must be multiples of 1 km",
    )
    predict.add_argument("--output", required=True, help="Output GeoTIFF path")
    predict.add_argument("--weights", help="Native .pt or legacy .npz checkpoint")
    predict.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    predict.add_argument("--bed-fill", type=float, default=-5_000.0)
    predict.add_argument("--surface-fill", type=float)
    predict.add_argument("--velocity-fill", type=float, default=0.0)
    predict.add_argument("--accumulation-fill", type=float, default=0.0)
    predict.set_defaults(handler=command_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        if os.environ.get("DEEPBEDMAP_DEBUG"):
            raise
        print(f"error: {error}", file=sys.stderr)
        return 2
