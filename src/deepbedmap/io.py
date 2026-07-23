"""Cross-platform raster preparation and GeoTIFF output."""

from __future__ import annotations

import math
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject

Bounds: TypeAlias = tuple[float, float, float, float]
ANTARCTIC_CRS = CRS.from_epsg(3031)
MODEL_PADDING_METRES = 1_000.0
OUTPUT_RESOLUTION_METRES = 250.0


@dataclass(frozen=True)
class ModelInputPaths:
    bed: str | Path
    surface: str | Path
    velocity_x: str | Path
    velocity_y: str | Path
    accumulation: str | Path


def validate_bounds(bounds: Bounds) -> Bounds:
    if len(bounds) != 4 or not all(math.isfinite(value) for value in bounds):
        raise ValueError("Bounds must contain four finite EPSG:3031 coordinates")
    left, bottom, right, top = (float(value) for value in bounds)
    if not left < right or not bottom < top:
        raise ValueError("Bounds must satisfy left < right and bottom < top")
    return left, bottom, right, top


def expand_bounds(bounds: Bounds, padding: float = MODEL_PADDING_METRES) -> Bounds:
    left, bottom, right, top = validate_bounds(bounds)
    return left - padding, bottom - padding, right + padding, top + padding


def _grid_size(bounds: Bounds, resolution: float) -> tuple[int, int]:
    left, bottom, right, top = validate_bounds(bounds)
    if resolution <= 0 or not math.isfinite(resolution):
        raise ValueError("Resolution must be a positive finite number")
    width_float = (right - left) / resolution
    height_float = (top - bottom) / resolution
    width, height = round(width_float), round(height_float)
    if not math.isclose(width_float, width, abs_tol=1e-6) or not math.isclose(
        height_float, height, abs_tol=1e-6
    ):
        raise ValueError(f"Bounds are not an integer multiple of {resolution:g} metres")
    if width < 1 or height < 1:
        raise ValueError("Requested grid is empty")
    return height, width


def read_raster_tile(
    path: str | Path,
    bounds: Bounds,
    resolution: float,
    *,
    band: int = 1,
    fill_value: float | None = None,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Read one band onto an exact EPSG:3031 pixel-centred grid."""

    bounds = validate_bounds(bounds)
    height, width = _grid_size(bounds, resolution)
    transform = from_bounds(*bounds, width=width, height=height)
    destination = np.full((height, width), np.nan, dtype=np.float32)
    with rasterio.open(path) as source:
        if source.crs is None or source.crs != ANTARCTIC_CRS:
            raise ValueError(f"Raster CRS must be EPSG:3031: {path} reports {source.crs}")
        if band < 1 or band > source.count:
            raise ValueError(f"Band {band} is not available in {path}")
        if not math.isclose(abs(source.transform.a), abs(source.transform.e), rel_tol=1e-6):
            raise ValueError(f"Raster must use square source pixels: {path}")
        reproject(
            source=rasterio.band(source, band),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=transform,
            dst_crs=ANTARCTIC_CRS,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    missing = ~np.isfinite(destination)
    if missing.any():
        if fill_value is None:
            raise ValueError(
                f"Raster {path} leaves {int(missing.sum()):,} missing pixels in requested bounds"
            )
        destination[missing] = np.float32(fill_value)
    return destination[np.newaxis, np.newaxis, :, :]


def prepare_model_inputs(
    paths: ModelInputPaths,
    output_bounds: Bounds,
    *,
    bed_fill: float = -5_000.0,
    surface_fill: float | None = None,
    velocity_fill: float = 0.0,
    accumulation_fill: float = 0.0,
) -> dict[str, np.ndarray]:
    """Prepare the four signed, float32 NCHW tensors used by the generator."""

    output_bounds = validate_bounds(output_bounds)
    _grid_size(output_bounds, 1_000.0)
    padded = expand_bounds(output_bounds)
    x = read_raster_tile(paths.bed, padded, 1_000.0, fill_value=bed_fill)
    w1 = read_raster_tile(paths.surface, padded, 100.0, fill_value=surface_fill)
    vx = read_raster_tile(paths.velocity_x, padded, 500.0, fill_value=velocity_fill)
    vy = read_raster_tile(paths.velocity_y, padded, 500.0, fill_value=velocity_fill)
    w2 = np.concatenate((vx, vy), axis=1)
    w3 = read_raster_tile(paths.accumulation, padded, 1_000.0, fill_value=accumulation_fill)
    return {"x": x, "w1": w1, "w2": w2, "w3": w3}


def _prediction_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 4:
        if array.shape[:2] != (1, 1):
            raise ValueError("A prediction batch must have shape (1, 1, height, width)")
        array = array[0, 0]
    elif array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError("A channel-first prediction must have one band")
        array = array[0]
    elif array.ndim != 2:
        raise ValueError("Prediction must be 2D, CHW, or NCHW")
    if not np.isfinite(array).all():
        raise ValueError("Prediction contains NaN or infinite values")
    return array


def write_geotiff(
    path: str | Path,
    array: np.ndarray,
    bounds: Bounds,
    *,
    compress: str = "deflate",
    tags: Mapping[str, str] | None = None,
) -> Path:
    """Write a float32, mask-safe EPSG:3031 prediction without a numeric sentinel."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bounds = validate_bounds(bounds)
    array = _prediction_array(array)
    height, width = array.shape
    x_resolution = (bounds[2] - bounds[0]) / width
    y_resolution = (bounds[3] - bounds[1]) / height
    if not math.isclose(x_resolution, y_resolution, rel_tol=1e-7):
        raise ValueError("Output bounds and shape must produce square pixels")
    if not math.isclose(x_resolution, OUTPUT_RESOLUTION_METRES, abs_tol=1e-6):
        raise ValueError(
            f"DeepBedMap output must be 250 m; bounds and shape imply {x_resolution:g} m"
        )
    transform = from_bounds(*bounds, width=width, height=height)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.tif")
    try:
        with rasterio.open(
            temporary,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs=ANTARCTIC_CRS,
            transform=transform,
            nodata=None,
            compress=compress,
            predictor=3,
            tiled=height >= 16 and width >= 16,
            bigtiff="IF_SAFER",
        ) as destination:
            destination.write(array, 1)
            output_tags = {
                "AREA_OR_POINT": "Area",
                "MODEL": "DeepBedMap Generator v1.1.0",
                "RESOLUTION_METRES": str(int(OUTPUT_RESOLUTION_METRES)),
            }
            if tags:
                output_tags.update({str(key): str(value) for key, value in tags.items()})
            destination.update_tags(**output_tags)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path.resolve()
