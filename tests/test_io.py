from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from deepbedmap.io import (
    ANTARCTIC_CRS,
    ModelInputPaths,
    prepare_model_inputs,
    read_raster_tile,
    write_geotiff,
)


def _write_raster(
    path: Path,
    data: np.ndarray,
    bounds: tuple[float, float, float, float],
    *,
    crs: CRS = ANTARCTIC_CRS,
) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_bounds(*bounds, width=data.shape[1], height=data.shape[0]),
        nodata=np.nan,
    ) as dataset:
        dataset.write(data.astype(np.float32), 1)
    return path


def test_prepare_model_inputs_preserves_shapes_and_signed_velocity(tmp_path: Path) -> None:
    padded = (-1_000.0, -1_000.0, 10_000.0, 10_000.0)
    bed = _write_raster(tmp_path / "bed.tif", np.full((11, 11), -100.0), padded)
    surface = _write_raster(tmp_path / "surface.tif", np.full((110, 110), 500.0), padded)
    velocity_x = _write_raster(tmp_path / "vx.tif", np.full((22, 22), -7.0), padded)
    velocity_y = _write_raster(tmp_path / "vy.tif", np.full((22, 22), 9.0), padded)
    accumulation = _write_raster(tmp_path / "acc.tif", np.full((11, 11), 0.25), padded)

    inputs = prepare_model_inputs(
        ModelInputPaths(bed, surface, velocity_x, velocity_y, accumulation),
        (0.0, 0.0, 9_000.0, 9_000.0),
    )
    assert inputs["x"].shape == (1, 1, 11, 11)
    assert inputs["w1"].shape == (1, 1, 110, 110)
    assert inputs["w2"].shape == (1, 2, 22, 22)
    assert inputs["w3"].shape == (1, 1, 11, 11)
    assert np.all(inputs["w2"][:, 0] == -7.0)
    assert np.all(inputs["w2"][:, 1] == 9.0)


def test_read_raster_rejects_wrong_crs(tmp_path: Path) -> None:
    path = _write_raster(
        tmp_path / "wrong-crs.tif",
        np.zeros((2, 2), dtype=np.float32),
        (0.0, 0.0, 2_000.0, 2_000.0),
        crs=CRS.from_epsg(4326),
    )
    with pytest.raises(ValueError, match="EPSG:3031"):
        read_raster_tile(path, (0.0, 0.0, 2_000.0, 2_000.0), 1_000.0)


def test_write_geotiff_round_trip_without_numeric_nodata(tmp_path: Path) -> None:
    prediction = np.arange(36 * 36, dtype=np.float32).reshape(1, 1, 36, 36)
    path = write_geotiff(
        tmp_path / "含 空格 prediction.tif",
        prediction,
        (0.0, 0.0, 9_000.0, 9_000.0),
    )
    with rasterio.open(path) as dataset:
        assert dataset.crs == ANTARCTIC_CRS
        assert dataset.shape == (36, 36)
        assert dataset.res == (250.0, 250.0)
        assert dataset.nodata is None
        np.testing.assert_array_equal(dataset.read(indexes=[1])[0], prediction[0, 0])
    assert not list(tmp_path.glob(".*.tmp.tif"))


def test_write_geotiff_rejects_non_square_pixels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="square pixels"):
        write_geotiff(
            tmp_path / "bad.tif",
            np.zeros((36, 36), dtype=np.float32),
            (0.0, 0.0, 9_000.0, 8_000.0),
        )
