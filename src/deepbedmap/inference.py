"""Device selection and deterministic DeepBedMap inference."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from .model import GeneratorModel


def select_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be one of: auto, cpu, cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch runtime cannot use it")
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        try:
            probe = torch.zeros(1, device="cuda")
            del probe
            return torch.device("cuda")
        except (RuntimeError, OSError):
            if requested == "cuda":
                raise
    return torch.device("cpu")


def predict_arrays(
    model: GeneratorModel,
    inputs: Mapping[str, np.ndarray],
    *,
    device: str | torch.device = "auto",
) -> tuple[np.ndarray, torch.device]:
    selected = select_device(device) if isinstance(device, str) else device
    model = model.to(selected).eval()
    required = {"x", "w1", "w2", "w3"}
    if set(inputs) != required:
        raise ValueError(f"Inputs must contain exactly {sorted(required)}")
    tensors = {
        name: torch.as_tensor(np.asarray(value, dtype=np.float32), device=selected)
        for name, value in inputs.items()
    }
    with torch.inference_mode():
        output = model(**tensors)
    result = output.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise RuntimeError("Model produced NaN or infinite values")
    return result, selected
