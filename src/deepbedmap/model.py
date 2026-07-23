"""PyTorch implementation of the published DeepBedMap generator."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import DeformConv2d

PUBLISHED_RESIDUAL_BLOCKS = 12
PUBLISHED_RESIDUAL_SCALING = 0.2


def _init_chainer_he(module: nn.Module) -> None:
    """Match Chainer's ``HeNormal(scale=0.1, fan_option='fan_in')``."""

    if isinstance(module, (nn.Conv2d, DeformConv2d)):
        nn.init.kaiming_normal_(module.weight, a=0.0, mode="fan_in", nonlinearity="relu")
        with torch.no_grad():
            module.weight.mul_(0.1)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ChainerCompatibleDeformConv2d(nn.Module):
    """Chainer's two-stage deformable convolution implemented with Torchvision.

    Chainer stores offsets as all x channels followed by all y channels. Torchvision
    consumes interleaved y/x channels. Keeping the original layout inside
    ``offset_conv`` makes legacy weight loading lossless; the small permutation in
    ``forward`` bridges the two conventions.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        offset_channels = 2 * kernel_size * kernel_size
        padding = kernel_size // 2
        self.offset_conv = nn.Conv2d(
            in_channels, offset_channels, kernel_size, padding=padding, bias=True
        )
        self.deform_conv = DeformConv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=True
        )

    def forward(self, x: Tensor) -> Tensor:
        chainer_offsets = self.offset_conv(x)
        kernel_points = self.kernel_size * self.kernel_size
        x_offsets = chainer_offsets[:, :kernel_points]
        y_offsets = chainer_offsets[:, kernel_points:]
        torchvision_offsets = torch.stack((y_offsets, x_offsets), dim=2).flatten(1, 2)
        return self.deform_conv(x, torchvision_offsets)


class DeepbedmapInputBlock(nn.Module):
    """Bring the four physical inputs onto one 1 km feature grid."""

    def __init__(self, out_channels: int = 32) -> None:
        super().__init__()
        self.conv_on_X = nn.Conv2d(1, out_channels, 3, stride=1, padding=0)
        self.conv_on_W1 = nn.Conv2d(1, out_channels, 30, stride=10, padding=0)
        self.conv_on_W2 = nn.Conv2d(2, out_channels, 6, stride=2, padding=0)
        self.conv_on_W3 = nn.Conv2d(1, out_channels, 3, stride=1, padding=0)

    def forward(self, x: Tensor, w1: Tensor, w2: Tensor, w3: Tensor) -> Tensor:
        features = (
            self.conv_on_X(x),
            self.conv_on_W1(w1),
            self.conv_on_W2(w2),
            self.conv_on_W3(w3),
        )
        spatial_shapes = {tuple(feature.shape[-2:]) for feature in features}
        if len(spatial_shapes) != 1:
            raise ValueError(f"Input rasters do not align after convolution: {spatial_shapes}")
        return torch.cat(features, dim=1)


class ResidualDenseBlock(nn.Module):
    """Five densely connected convolution layers with residual scaling."""

    def __init__(
        self,
        in_out_channels: int = 64,
        inter_channels: int = 32,
        residual_scaling: float = PUBLISHED_RESIDUAL_SCALING,
    ) -> None:
        super().__init__()
        self.residual_scaling = residual_scaling
        self.conv_layer1 = nn.Conv2d(in_out_channels, inter_channels, 3, padding=1)
        self.conv_layer2 = nn.Conv2d(in_out_channels + inter_channels, inter_channels, 3, padding=1)
        self.conv_layer3 = nn.Conv2d(
            in_out_channels + 2 * inter_channels, inter_channels, 3, padding=1
        )
        self.conv_layer4 = nn.Conv2d(
            in_out_channels + 3 * inter_channels, inter_channels, 3, padding=1
        )
        self.conv_layer5 = nn.Conv2d(
            in_out_channels + 4 * inter_channels, in_out_channels, 3, padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        a1 = F.leaky_relu(self.conv_layer1(x), negative_slope=0.2)
        a2 = F.leaky_relu(self.conv_layer2(torch.cat((x, a1), dim=1)), negative_slope=0.2)
        a3 = F.leaky_relu(self.conv_layer3(torch.cat((x, a1, a2), dim=1)), negative_slope=0.2)
        a4 = F.leaky_relu(self.conv_layer4(torch.cat((x, a1, a2, a3), dim=1)), negative_slope=0.2)
        a5 = self.conv_layer5(torch.cat((x, a1, a2, a3, a4), dim=1))
        return x + a5 * self.residual_scaling


class ResInResDenseBlock(nn.Module):
    """Three residual dense blocks wrapped in an outer residual connection."""

    def __init__(self, residual_scaling: float = PUBLISHED_RESIDUAL_SCALING) -> None:
        super().__init__()
        self.residual_scaling = residual_scaling
        self.residual_dense_block1 = ResidualDenseBlock(residual_scaling=residual_scaling)
        self.residual_dense_block2 = ResidualDenseBlock(residual_scaling=residual_scaling)
        self.residual_dense_block3 = ResidualDenseBlock(residual_scaling=residual_scaling)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.residual_dense_block1(x)
        residual = self.residual_dense_block2(residual)
        residual = self.residual_dense_block3(residual)
        return x + residual * self.residual_scaling


class GeneratorModel(nn.Module):
    """Published DeepBedMap v1.1 generator, migrated from Chainer to PyTorch."""

    def __init__(
        self,
        num_residual_blocks: int = PUBLISHED_RESIDUAL_BLOCKS,
        residual_scaling: float = PUBLISHED_RESIDUAL_SCALING,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        if num_residual_blocks < 1:
            raise ValueError("num_residual_blocks must be at least 1")
        if not 0 < residual_scaling <= 1:
            raise ValueError("residual_scaling must be in (0, 1]")
        self.num_residual_blocks = num_residual_blocks
        self.residual_scaling = residual_scaling
        self.input_block = DeepbedmapInputBlock()
        self.pre_residual_conv_layer = nn.Conv2d(128, 64, 3, padding=1)
        self.residual_network = nn.ModuleList(
            [ResInResDenseBlock(residual_scaling) for _ in range(num_residual_blocks)]
        )
        self.post_residual_conv_layer = nn.Conv2d(64, 64, 3, padding=1)
        self.post_upsample_conv_layer_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.post_upsample_conv_layer_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.final_conv_layer1 = ChainerCompatibleDeformConv2d(64, 64)
        self.final_conv_layer2 = ChainerCompatibleDeformConv2d(64, out_channels)
        self.apply(_init_chainer_he)

    @staticmethod
    def _validate_inputs(inputs: Sequence[Tensor]) -> None:
        x, w1, w2, w3 = inputs
        if any(tensor.ndim != 4 for tensor in inputs):
            raise ValueError("Every model input must be NCHW (four dimensions)")
        batches = {tensor.shape[0] for tensor in inputs}
        if len(batches) != 1:
            raise ValueError("All model inputs must have the same batch size")
        channels = (x.shape[1], w1.shape[1], w2.shape[1], w3.shape[1])
        if channels != (1, 1, 2, 1):
            raise ValueError(f"Expected input channels (1, 1, 2, 1), got {channels}")
        x_height, x_width = x.shape[-2:]
        expected_shapes = (
            (x_height * 10, x_width * 10),
            (x_height * 2, x_width * 2),
            (x_height, x_width),
        )
        actual_shapes = (tuple(w1.shape[-2:]), tuple(w2.shape[-2:]), tuple(w3.shape[-2:]))
        if actual_shapes != expected_shapes:
            raise ValueError(
                "Input resolutions must be X:W1:W2:W3 = 1:10:2:1; "
                f"expected {expected_shapes}, got {actual_shapes}"
            )
        if min(x_height, x_width) < 3:
            raise ValueError("The BEDMAP2 input must be at least 3x3 pixels")

    def forward(self, x: Tensor, w1: Tensor, w2: Tensor, w3: Tensor) -> Tensor:
        self._validate_inputs((x, w1, w2, w3))
        features = self.input_block(x, w1, w2, w3)
        residual_input = F.leaky_relu(self.pre_residual_conv_layer(features), negative_slope=0.2)
        residual = residual_input
        for block in self.residual_network:
            residual = block(residual)
        residual = residual_input + self.post_residual_conv_layer(residual)
        residual = F.interpolate(residual, scale_factor=2, mode="nearest")
        residual = F.leaky_relu(self.post_upsample_conv_layer_1(residual), negative_slope=0.2)
        residual = F.interpolate(residual, scale_factor=2, mode="nearest")
        residual = F.leaky_relu(self.post_upsample_conv_layer_2(residual), negative_slope=0.2)
        residual = F.leaky_relu(self.final_conv_layer1(residual), negative_slope=0.2)
        return self.final_conv_layer2(residual)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
