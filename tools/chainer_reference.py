"""Generate a one-time Chainer reference output for migration validation.

This script is deliberately outside the supported runtime. Run it only inside an
isolated Python 3.8 environment containing Chainer 7.0 and NumPy 1.23.
"""

from __future__ import annotations

import argparse
import ast
import types
from pathlib import Path

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

CLASS_NAMES = {
    "DeepbedmapInputBlock",
    "ResidualDenseBlock",
    "ResInResDenseBlock",
    "GeneratorModel",
}


def load_generator_class(source: Path):
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    classes = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in CLASS_NAMES
    ]
    namespace = {
        "chainer": chainer,
        "F": F,
        "L": L,
        "np": np,
        "cupy": types.SimpleNamespace(ndarray=np.ndarray),
    }
    exec(compile(ast.Module(classes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["GeneratorModel"]


def fixed_inputs() -> dict[str, np.ndarray]:
    def values(shape: tuple[int, ...], modulus: int, scale: float, offset: float) -> np.ndarray:
        data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        return (data % np.float32(modulus)) * np.float32(scale) + np.float32(offset)

    return {
        "x": values((1, 1, 11, 11), 121, 20.0, -2_000.0),
        "w1": values((1, 1, 110, 110), 251, 10.0, 500.0),
        "w2": values((1, 2, 22, 22), 97, 15.0, -720.0),
        "w3": values((1, 1, 11, 11), 31, 0.01, 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("srgan_train.py"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    generator_class = load_generator_class(args.source)
    model = generator_class(num_residual_blocks=12, residual_scaling=0.2)
    chainer.serializers.load_npz(str(args.weights), model)
    with chainer.using_config("train", False), chainer.using_config("enable_backprop", False):
        inputs = fixed_inputs()
        input_block = model.input_block(**inputs)
        pre_residual = F.leaky_relu(model.pre_residual_conv_layer(input_block), slope=0.2)
        residual_steps = {}
        residual_network = pre_residual
        for index, block in enumerate(model.residual_network):
            residual_network = block(residual_network)
            residual_steps[f"residual_{index}"] = residual_network.array
        post_residual = pre_residual + model.post_residual_conv_layer(residual_network)
        upsample1_raw = F.resize_images(
            post_residual,
            output_shape=(2 * post_residual.shape[-2], 2 * post_residual.shape[-1]),
            mode="nearest",
        )
        upsample1 = F.leaky_relu(model.post_upsample_conv_layer_1(upsample1_raw), slope=0.2)
        upsample2_raw = F.resize_images(
            upsample1,
            output_shape=(2 * upsample1.shape[-2], 2 * upsample1.shape[-1]),
            mode="nearest",
        )
        upsample2 = F.leaky_relu(model.post_upsample_conv_layer_2(upsample2_raw), slope=0.2)
        offset1 = model.final_conv_layer1.offset_conv(upsample2)
        deform1 = F.leaky_relu(model.final_conv_layer1.deform_conv(upsample2, offset1), slope=0.2)
        offset2 = model.final_conv_layer2.offset_conv(deform1)
        output = model.final_conv_layer2.deform_conv(deform1, offset2).array.astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        input_block=input_block.array,
        pre_residual=pre_residual.array,
        residual_network=residual_network.array,
        post_residual=post_residual.array,
        upsample1_raw=upsample1_raw.array,
        upsample1=upsample1.array,
        upsample2_raw=upsample2_raw.array,
        upsample2=upsample2.array,
        offset1=offset1.array,
        deform1=deform1.array,
        offset2=offset2.array,
        output=output,
        **residual_steps,
    )
    print(
        f"shape={output.shape} min={output.min():.9g} max={output.max():.9g} "
        f"mean={output.mean():.9g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
