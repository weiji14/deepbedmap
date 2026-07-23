from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from deepbedmap.model import (
    PUBLISHED_RESIDUAL_SCALING,
    ChainerCompatibleDeformConv2d,
    GeneratorModel,
    parameter_count,
)


def test_published_parameter_count_and_metadata() -> None:
    model = GeneratorModel()
    assert parameter_count(model) == 8_907_749
    assert model.num_residual_blocks == 12
    assert model.residual_scaling == PUBLISHED_RESIDUAL_SCALING == 0.2


def test_generator_shape_with_small_network() -> None:
    model = GeneratorModel(num_residual_blocks=1).eval()
    with torch.inference_mode():
        result = model(
            x=torch.zeros(2, 1, 11, 12),
            w1=torch.zeros(2, 1, 110, 120),
            w2=torch.zeros(2, 2, 22, 24),
            w3=torch.zeros(2, 1, 11, 12),
        )
    assert result.shape == (2, 1, 36, 40)
    assert torch.isfinite(result).all()


def test_generator_rejects_misaligned_inputs() -> None:
    model = GeneratorModel(num_residual_blocks=1)
    with pytest.raises(ValueError, match="resolutions"):
        model(
            x=torch.zeros(1, 1, 11, 11),
            w1=torch.zeros(1, 1, 109, 110),
            w2=torch.zeros(1, 2, 22, 22),
            w3=torch.zeros(1, 1, 11, 11),
        )


def test_zero_offset_deformable_convolution_matches_conv2d() -> None:
    layer = ChainerCompatibleDeformConv2d(1, 1)
    with torch.no_grad():
        layer.offset_conv.weight.zero_()
        layer.offset_conv.bias.zero_()
        layer.deform_conv.weight.copy_(torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3))
        layer.deform_conv.bias.fill_(0.75)
    image = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
    actual = layer(image)
    expected = F.conv2d(image, layer.deform_conv.weight, layer.deform_conv.bias, padding=1)
    torch.testing.assert_close(actual, expected)


def test_chainer_xy_offsets_are_permuted_for_torchvision() -> None:
    layer = ChainerCompatibleDeformConv2d(1, 1)
    with torch.no_grad():
        layer.offset_conv.weight.zero_()
        layer.offset_conv.bias.zero_()
        # Chainer channel 4 is the centre kernel point's x offset.
        layer.offset_conv.bias[4] = 1.0
        layer.deform_conv.weight.zero_()
        layer.deform_conv.weight[0, 0, 1, 1] = 1.0
        layer.deform_conv.bias.zero_()
    image = torch.arange(5, dtype=torch.float32).repeat(5, 1).reshape(1, 1, 5, 5)
    actual = layer(image)[0, 0]
    expected = torch.tensor(
        [[1, 2, 3, 4, 0]] * 5,
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected)
