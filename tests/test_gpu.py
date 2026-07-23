from __future__ import annotations

import numpy as np
import pytest
import torch

from deepbedmap.assets import WEIGHTS, cached_path
from deepbedmap.inference import predict_arrays
from deepbedmap.weights import load_legacy_weights

RELEASE_WEIGHTS = cached_path(WEIGHTS)


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not RELEASE_WEIGHTS.exists(), reason="public checkpoint is not cached")
def test_published_checkpoint_cpu_cuda_parity() -> None:
    rng = np.random.default_rng(42)
    inputs = {
        "x": rng.normal(size=(1, 1, 11, 11)).astype(np.float32),
        "w1": rng.normal(size=(1, 1, 110, 110)).astype(np.float32),
        "w2": rng.normal(size=(1, 2, 22, 22)).astype(np.float32),
        "w3": rng.normal(size=(1, 1, 11, 11)).astype(np.float32),
    }
    model = load_legacy_weights(RELEASE_WEIGHTS)

    cpu, _ = predict_arrays(model, inputs, device="cpu")
    cuda, _ = predict_arrays(model, inputs, device="cuda")

    np.testing.assert_allclose(cuda, cpu, rtol=1e-4, atol=0.01)
