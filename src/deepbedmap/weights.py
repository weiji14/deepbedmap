"""Conversion and loading of the public Chainer generator checkpoint."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import (
    PUBLISHED_RESIDUAL_BLOCKS,
    PUBLISHED_RESIDUAL_SCALING,
    GeneratorModel,
    parameter_count,
)

CHECKPOINT_FORMAT = "deepbedmap-pytorch-v3"
PUBLISHED_PARAMETER_COUNT = 8_907_749
PUBLISHED_MODEL_NAME = "DeepBedMap Generator v1.1.0"
OFFSET_LAYOUT = "chainer-xy-split; permuted to torchvision-yx-interleaved at runtime"


def legacy_key_to_torch(key: str, *, num_residual_blocks: int = PUBLISHED_RESIDUAL_BLOCKS) -> str:
    """Map one Chainer serializer key to the matching PyTorch state key.

    Chainer's ``Sequential`` serializer names the first repeated link ``0`` and
    the remaining links in reverse execution order. The published 12-block NPZ
    therefore executes archive blocks ``0, 11, 10, ..., 1``. PyTorch's
    ``ModuleList`` uses ordinary execution order, so those indices must also be
    translated while converting the checkpoint.
    """

    match = re.match(r"residual_network/(\d+)/(.*)", key)
    if match:
        archive_index = int(match.group(1))
        if not 0 <= archive_index < num_residual_blocks:
            raise ValueError(f"Residual block index is out of range: {key}")
        execution_index = 0 if archive_index == 0 else num_residual_blocks - archive_index
        key = f"residual_network/{execution_index}/{match.group(2)}"

    if key.endswith("/W"):
        key = f"{key[:-2]}/weight"
    elif key.endswith("/b"):
        key = f"{key[:-2]}/bias"
    else:
        raise ValueError(f"Unsupported Chainer parameter key: {key}")
    return key.replace("/", ".")


def inspect_legacy_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        residual_indices = {
            int(match.group(1))
            for key in keys
            if (match := re.match(r"residual_network/(\d+)/", key))
        }
        total_parameters = sum(int(archive[key].size) for key in keys)
        all_finite = all(bool(np.isfinite(archive[key]).all()) for key in keys)
    blocks = max(residual_indices) + 1 if residual_indices else 0
    return {
        "path": str(path.resolve()),
        "arrays": len(keys),
        "num_residual_blocks": blocks,
        "parameters": total_parameters,
        "all_finite": all_finite,
    }


def load_legacy_weights(
    path: str | Path,
    *,
    residual_scaling: float = PUBLISHED_RESIDUAL_SCALING,
    device: str | torch.device = "cpu",
) -> GeneratorModel:
    """Load the public Chainer ``.npz`` without importing Chainer or CuPy."""

    info = inspect_legacy_checkpoint(path)
    if info["num_residual_blocks"] != PUBLISHED_RESIDUAL_BLOCKS:
        raise ValueError(
            "The checkpoint does not match the published 12-block generator: "
            f"found {info['num_residual_blocks']} blocks"
        )
    if not info["all_finite"]:
        raise ValueError("The legacy checkpoint contains NaN or infinite values")

    model = GeneratorModel(
        num_residual_blocks=info["num_residual_blocks"],
        residual_scaling=residual_scaling,
    )
    expected = model.state_dict()
    converted: dict[str, torch.Tensor] = {}
    with np.load(path, allow_pickle=False) as archive:
        for legacy_key in archive.files:
            torch_key = legacy_key_to_torch(
                legacy_key, num_residual_blocks=info["num_residual_blocks"]
            )
            converted[torch_key] = torch.from_numpy(np.array(archive[legacy_key], copy=True))

    missing = sorted(set(expected) - set(converted))
    unexpected = sorted(set(converted) - set(expected))
    if missing or unexpected:
        raise ValueError(f"Checkpoint key mismatch; missing={missing}, unexpected={unexpected}")
    shape_errors = {
        key: (tuple(converted[key].shape), tuple(expected[key].shape))
        for key in expected
        if converted[key].shape != expected[key].shape
    }
    if shape_errors:
        raise ValueError(f"Checkpoint tensor shape mismatch: {shape_errors}")

    model.load_state_dict(converted, strict=True)
    model.eval()
    return model.to(device)


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash a state dict independent of the surrounding Torch archive."""

    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(value) for value in tensor.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def checkpoint_metadata(
    model: GeneratorModel,
    state_dict: Mapping[str, torch.Tensor],
    *,
    source: str | None = None,
    source_sha256: str | None = None,
    weights_release: str = "custom",
) -> dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "model": PUBLISHED_MODEL_NAME,
        "num_residual_blocks": model.num_residual_blocks,
        "residual_scaling": model.residual_scaling,
        "parameter_count": parameter_count(model),
        "source": source,
        "source_sha256": source_sha256,
        "state_sha256": state_dict_sha256(state_dict),
        "weights_release": weights_release,
        "offset_layout": OFFSET_LAYOUT,
        "crs": "EPSG:3031",
        "output_resolution_metres": 250,
    }


def save_native_checkpoint(
    model: GeneratorModel,
    destination: str | Path,
    *,
    source: str | None = None,
    source_sha256: str | None = None,
    weights_release: str = "custom",
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    payload = {
        "format": CHECKPOINT_FORMAT,
        "metadata": checkpoint_metadata(
            model,
            state_dict,
            source=source,
            source_sha256=source_sha256,
            weights_release=weights_release,
        ),
        "state_dict": state_dict,
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def convert_legacy_checkpoint(source: str | Path, destination: str | Path) -> Path:
    from .assets import WEIGHTS, file_digest

    source = Path(source)
    model = load_legacy_weights(source)
    if parameter_count(model) != PUBLISHED_PARAMETER_COUNT:
        raise ValueError(
            f"Published model should contain {PUBLISHED_PARAMETER_COUNT:,} parameters, "
            f"found {parameter_count(model):,}"
        )
    source_sha256 = file_digest(source, "sha256")
    release = "v1.1.0" if source_sha256 == WEIGHTS.sha256 else "custom"
    return save_native_checkpoint(
        model,
        destination,
        source=source.name,
        source_sha256=source_sha256,
        weights_release=release,
    )


def load_native_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[GeneratorModel, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported native checkpoint format in {path}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Native checkpoint metadata is missing or invalid")
    expected_metadata = {
        "format": CHECKPOINT_FORMAT,
        "model": PUBLISHED_MODEL_NAME,
        "num_residual_blocks": PUBLISHED_RESIDUAL_BLOCKS,
        "residual_scaling": PUBLISHED_RESIDUAL_SCALING,
        "parameter_count": PUBLISHED_PARAMETER_COUNT,
        "offset_layout": OFFSET_LAYOUT,
        "crs": "EPSG:3031",
        "output_resolution_metres": 250,
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Native checkpoint metadata does not match DeepBedMap v1.1: {mismatches}")
    if metadata.get("weights_release") not in {"v1.1.0", "custom"}:
        raise ValueError("Native checkpoint weights_release must be 'v1.1.0' or 'custom'")
    source_sha256 = metadata.get("source_sha256")
    if source_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", str(source_sha256)):
        raise ValueError("Native checkpoint source_sha256 is invalid")

    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Native checkpoint state_dict is missing or invalid")
    invalid_tensors = [
        key for key, value in state_dict.items() if not isinstance(value, torch.Tensor)
    ]
    if invalid_tensors:
        raise ValueError(f"Native checkpoint contains non-tensor values: {invalid_tensors}")
    non_finite = [
        key
        for key, value in state_dict.items()
        if value.is_floating_point() and not bool(torch.isfinite(value).all())
    ]
    if non_finite:
        raise ValueError(f"Native checkpoint contains NaN or infinite tensors: {non_finite}")
    actual_state_sha256 = state_dict_sha256(state_dict)
    expected_state_sha256 = metadata.get("state_sha256")
    if not isinstance(expected_state_sha256, str) or not hmac.compare_digest(
        actual_state_sha256, expected_state_sha256
    ):
        raise ValueError("Native checkpoint state digest does not match its metadata")

    model = GeneratorModel(
        num_residual_blocks=PUBLISHED_RESIDUAL_BLOCKS,
        residual_scaling=PUBLISHED_RESIDUAL_SCALING,
    )
    model.load_state_dict(state_dict, strict=True)
    if parameter_count(model) != PUBLISHED_PARAMETER_COUNT:
        raise ValueError("Native checkpoint model has an unexpected parameter count")
    model.eval()
    return model.to(device), metadata
