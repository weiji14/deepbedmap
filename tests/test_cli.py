from __future__ import annotations

import json
import sys
from pathlib import Path

from deepbedmap.cli import main


def test_doctor_json(capsys) -> None:
    assert main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["deepbedmap"] == "2.0.0"
    assert (3, 11) <= sys.version_info[:2] < (3, 14)
    assert report["rasterio"]
    assert report["torch"]


def test_explicit_missing_weights_do_not_fall_back_to_cache(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "does-not-exist.pt"
    assert main(["smoke", "--weights", str(missing), "--device", "cpu"]) == 2
    assert "Explicit model weights do not exist" in capsys.readouterr().err
