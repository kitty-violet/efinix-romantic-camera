"""Executable A/B architecture specification for the 720p DS-ResStyle network.

This file intentionally uses only the Python standard library.  It freezes the
operator order, tensor shapes, convolution MAC counts and raw weight counts for
the A/B comparison without prematurely choosing padding, activation, BN or
quantization details that are still open design questions.

MAC accounting follows the existing project convention:

* one multiply-accumulate is counted as one MAC;
* nearest-neighbour copies and residual additions are not counted as MACs;
* bias, scale tables, padding bubbles, DMA and DDR stalls are excluded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Callable, Iterable


WIDTH = 1280
HEIGHT = 720
FPS = 30


@dataclass(frozen=True)
class TensorShape:
    width: int
    height: int
    channels: int

    @property
    def elements(self) -> int:
        return self.width * self.height * self.channels


@dataclass(frozen=True)
class LayerSpec:
    name: str
    operator: str
    input: TensorShape
    output: TensorShape
    macs: int
    weights: int
    note: str = ""


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    description: str
    layers: tuple[LayerSpec, ...]

    @property
    def macs_per_frame(self) -> int:
        return sum(layer.macs for layer in self.layers)

    @property
    def macs_per_second(self) -> int:
        return self.macs_per_frame * FPS

    @property
    def weight_count(self) -> int:
        return sum(layer.weights for layer in self.layers)


def conv3(name: str, shape: TensorShape, out_channels: int, note: str = "") -> LayerSpec:
    output = TensorShape(shape.width, shape.height, out_channels)
    weights = 9 * shape.channels * out_channels
    return LayerSpec(name, "conv3x3", shape, output, output.width * output.height * weights, weights, note)


def dw3(name: str, shape: TensorShape, note: str = "") -> LayerSpec:
    weights = 9 * shape.channels
    return LayerSpec(name, "dw3x3", shape, shape, shape.width * shape.height * weights, weights, note)


def pw1(name: str, shape: TensorShape, out_channels: int, note: str = "") -> LayerSpec:
    output = TensorShape(shape.width, shape.height, out_channels)
    weights = shape.channels * out_channels
    return LayerSpec(name, "pw1x1", shape, output, output.width * output.height * weights, weights, note)


def dsconv(
    name: str,
    shape: TensorShape,
    out_channels: int,
    *,
    stride: int = 1,
    note: str = "",
) -> LayerSpec:
    if stride not in (1, 2):
        raise ValueError(f"unsupported stride: {stride}")
    output = TensorShape(shape.width // stride, shape.height // stride, out_channels)
    weights = 9 * shape.channels + shape.channels * out_channels
    macs = output.width * output.height * weights
    return LayerSpec(name, f"dw3x3_s{stride}+pw1x1", shape, output, macs, weights, note)


def upsample2(name: str, shape: TensorShape, note: str = "") -> LayerSpec:
    output = TensorShape(shape.width * 2, shape.height * 2, shape.channels)
    return LayerSpec(name, "nearest_x2", shape, output, 0, 0, note)


def residual_add(name: str, shape: TensorShape, note: str = "") -> LayerSpec:
    return LayerSpec(name, "residual_add", shape, shape, 0, 0, note)


def shared_encoder_and_body() -> list[LayerSpec]:
    layers: list[LayerSpec] = []
    shape = TensorShape(WIDTH, HEIGHT, 3)

    layer = conv3("stem", shape, 8, "A/B共用普通3x3 RGB入口")
    layers.append(layer)
    shape = layer.output

    layer = dsconv("down1", shape, 16, stride=2)
    layers.append(layer)
    shape = layer.output

    layer = dsconv("down2", shape, 32, stride=2)
    layers.append(layer)
    shape = layer.output

    for block in range(1, 4):
        skip_shape = shape
        for conv in range(1, 3):
            layer = dsconv(f"res{block}_conv{conv}", shape, 32)
            layers.append(layer)
            shape = layer.output
        if shape != skip_shape:
            raise AssertionError(f"residual block {block} changed shape")
        layer = residual_add(f"res{block}_add", shape)
        layers.append(layer)
        shape = layer.output

    return layers


def build_model_a() -> ModelSpec:
    """A: current 8/16/32 baseline with upsample -> DW/PW decoder."""

    layers = shared_encoder_and_body()
    shape = layers[-1].output

    layer = upsample2("up1_nn", shape)
    layers.append(layer)
    shape = layer.output
    layer = dsconv("up1_dsconv", shape, 16)
    layers.append(layer)
    shape = layer.output

    layer = upsample2("up2_nn", shape)
    layers.append(layer)
    shape = layer.output
    layer = dsconv("up2_dsconv", shape, 8)
    layers.append(layer)
    shape = layer.output

    layer = conv3("rgb_head", shape, 3, "A版普通3x3输出头")
    layers.append(layer)
    return ModelSpec(
        "A_BASELINE_8_16_32_R3",
        "8/16/32、3残差块；NN放大后执行DW+PW；最终3x3输出",
        tuple(layers),
    )


def build_model_b() -> ModelSpec:
    """B: balanced low-resolution decoder plus high-resolution DW refinement."""

    layers = shared_encoder_and_body()
    shape = layers[-1].output

    layer = dw3("up1_low_dw", shape, "低分辨率空间处理")
    layers.append(layer)
    shape = layer.output
    layer = pw1("up1_low_pw", shape, 16, "低分辨率降通道")
    layers.append(layer)
    shape = layer.output
    layer = upsample2("up1_nn", shape)
    layers.append(layer)
    shape = layer.output
    layer = dw3("up1_high_dw", shape, "放大后逐通道边缘细化")
    layers.append(layer)
    shape = layer.output

    layer = dw3("up2_low_dw", shape, "低分辨率空间处理")
    layers.append(layer)
    shape = layer.output
    layer = pw1("up2_low_pw", shape, 8, "低分辨率降通道")
    layers.append(layer)
    shape = layer.output
    layer = upsample2("up2_nn", shape)
    layers.append(layer)
    shape = layer.output
    layer = dw3("up2_high_dw", shape, "720p逐通道边缘细化")
    layers.append(layer)
    shape = layer.output

    layer = pw1("rgb_head", shape, 3, "B版1x1 RGB输出头")
    layers.append(layer)
    return ModelSpec(
        "B_BALANCED_DECODER_8_16_32_R3",
        "8/16/32、3残差块；低分辨率DW+PW、NN放大、高分辨率DW细化；最终1x1输出",
        tuple(layers),
    )


MODELS: dict[str, Callable[[], ModelSpec]] = {
    "A": build_model_a,
    "B": build_model_b,
}


EXPECTED = {
    "A": {"macs_per_frame": 1_370_419_200, "weight_count": 10_232},
    "B": {"macs_per_frame": 966_758_400, "weight_count": 10_256},
}


def validate_model(label: str, model: ModelSpec) -> None:
    expected_input = TensorShape(WIDTH, HEIGHT, 3)
    previous = expected_input
    for layer in model.layers:
        if layer.input != previous:
            raise AssertionError(f"{model.model_id}: {layer.name} input {layer.input} != {previous}")
        previous = layer.output
    if previous != expected_input:
        raise AssertionError(f"{model.model_id}: unexpected final shape {previous}")
    if model.macs_per_frame != EXPECTED[label]["macs_per_frame"]:
        raise AssertionError(f"{model.model_id}: MAC mismatch {model.macs_per_frame}")
    if model.weight_count != EXPECTED[label]["weight_count"]:
        raise AssertionError(f"{model.model_id}: weight mismatch {model.weight_count}")


def model_to_dict(model: ModelSpec) -> dict[str, object]:
    return {
        "model_id": model.model_id,
        "description": model.description,
        "input_resolution": [WIDTH, HEIGHT],
        "fps_target": FPS,
        "macs_per_frame": model.macs_per_frame,
        "gmac_per_frame": model.macs_per_frame / 1e9,
        "gmac_per_second_at_30fps": model.macs_per_second / 1e9,
        "weight_count_without_bias_or_scales": model.weight_count,
        "layers": [asdict(layer) for layer in model.layers],
    }


def print_summary(models: Iterable[tuple[str, ModelSpec]]) -> None:
    print("version\tmodel_id\tGMAC/frame\tGMAC/s@30\tweights")
    for label, model in models:
        print(
            f"{label}\t{model.model_id}\t{model.macs_per_frame / 1e9:.9f}\t"
            f"{model.macs_per_second / 1e9:.6f}\t{model.weight_count}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the DS-ResStyle A/B architecture specs")
    parser.add_argument("--model", choices=("A", "B", "all"), default="all")
    parser.add_argument("--json", action="store_true", help="emit complete machine-readable layer specs")
    args = parser.parse_args()

    labels = tuple(MODELS) if args.model == "all" else (args.model,)
    models = [(label, MODELS[label]()) for label in labels]
    for label, model in models:
        validate_model(label, model)

    if args.json:
        print(json.dumps({label: model_to_dict(model) for label, model in models}, ensure_ascii=False, indent=2))
    else:
        print_summary(models)


if __name__ == "__main__":
    main()
