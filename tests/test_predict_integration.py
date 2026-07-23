from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from deepbedmap.assets import WEIGHTS, cached_path, file_digest
from deepbedmap.cli import main, native_weights_path
from deepbedmap.io import ANTARCTIC_CRS

OUTPUT_BOUNDS = (0.0, 0.0, 9_000.0, 9_000.0)
PADDED_BOUNDS = (-1_000.0, -1_000.0, 10_000.0, 10_000.0)


def _cached_checkpoint() -> Path | None:
    repository_cache = Path(__file__).resolve().parents[1] / ".cache" / "deepbedmap"
    candidates = (
        repository_cache / "deepbedmap_generator_v1.1.0.pt",
        native_weights_path(),
        cached_path(WEIGHTS),
        repository_cache / WEIGHTS.name,
    )
    return next((path for path in candidates if path.is_file()), None)


def _write_input(path: Path, data: np.ndarray) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=ANTARCTIC_CRS,
        transform=from_bounds(
            *PADDED_BOUNDS,
            width=data.shape[1],
            height=data.shape[0],
        ),
        nodata=None,
    ) as dataset:
        dataset.write(data.astype(np.float32), 1)
    return path


@pytest.mark.integration
def test_cli_predict_writes_real_250m_geotiff(tmp_path: Path, capsys) -> None:
    weights = _cached_checkpoint()
    if weights is None:
        pytest.skip("published weights are not cached; run '.\\run.ps1 fetch-weights'")

    # The five input grids cover the model's required 1 km context around a
    # 9 km square output region, at each input's native model resolution.
    bed = _write_input(tmp_path / "bed.tif", np.full((11, 11), -350.0))
    surface = _write_input(
        tmp_path / "surface.tif",
        np.linspace(400.0, 700.0, 110 * 110, dtype=np.float32).reshape(110, 110),
    )
    velocity_x = _write_input(tmp_path / "velocity-x.tif", np.full((22, 22), -12.0))
    velocity_y = _write_input(tmp_path / "velocity-y.tif", np.full((22, 22), 8.0))
    accumulation = _write_input(tmp_path / "accumulation.tif", np.full((11, 11), 0.22))
    output = tmp_path / "prediction.tif"

    status = main(
        [
            "predict",
            "--bed",
            str(bed),
            "--surface",
            str(surface),
            "--velocity-x",
            str(velocity_x),
            "--velocity-y",
            str(velocity_y),
            "--accumulation",
            str(accumulation),
            "--bounds",
            *(str(value) for value in OUTPUT_BOUNDS),
            "--output",
            str(output),
            "--weights",
            str(weights),
            "--device",
            "cpu",
        ]
    )

    captured = capsys.readouterr()
    assert status == 0, captured.err
    result = json.loads(captured.out)
    assert result["status"] == "ok"
    assert result["shape"] == [1, 1, 36, 36]
    assert result["resolution_metres"] == 250
    assert Path(result["weights"]) == weights.resolve()
    assert result["residual_scaling"] == 0.2
    assert result["checkpoint_sha256"] == file_digest(weights)
    assert result["weights_release"] == "v1.1.0"

    with rasterio.open(output) as dataset:
        prediction = dataset.read(indexes=[1])[0]
        assert dataset.crs == ANTARCTIC_CRS
        assert dataset.shape == (36, 36)
        assert dataset.res == (250.0, 250.0)
        assert tuple(dataset.bounds) == OUTPUT_BOUNDS
        assert dataset.nodata is None
        assert dataset.read_masks(indexes=[1]).min() == 255
        assert np.isfinite(prediction).all()
        tags = dataset.tags()
        assert tags["MODEL"] == "DeepBedMap Generator v1.1.0"
        assert tags["CHECKPOINT_SHA256"] == file_digest(weights)
        assert tags["WEIGHTS_RELEASE"] == "v1.1.0"
