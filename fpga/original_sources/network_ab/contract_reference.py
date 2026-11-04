"""Stdlib numerical oracle for the shared Model B mathematical contract.

Small-tensor verification, not a fast 720p engine or a training implementation.
Residual alignment uses exact rational scales and unbounded reference integers;
it does NOT assert that an arbitrary multiplier/shift circuit exists in RTL.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

if __package__:
    from .model_contract import ContractError, QUANTIZED, compile_contract
else:
    from model_contract import ContractError, QUANTIZED, compile_contract

I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1


def round_half_away(value: Fraction) -> int:
    value = Fraction(value)
    magnitude = abs(value.numerator)
    rounded = (2 * magnitude + value.denominator) // (2 * value.denominator)
    return -rounded if value < 0 else rounded


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def positive_scale(value: Any, label: str = "scale") -> Fraction:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(v) is not int for v in value) or value[0] <= 0 or value[1] <= 0):
        raise ContractError(f"{label}: requires explicit positive numerator/denominator")
    return Fraction(*value)


def u8_to_i8(value: int) -> int:
    if type(value) is not int or not 0 <= value <= 255:
        raise ContractError("input channel must be uint8")
    return value - 128


def u8_to_real(value: int) -> Fraction:
    return Fraction(u8_to_i8(value), 128)


def i8_to_u8(value: int) -> int:
    if type(value) is not int:
        raise ContractError("output channel must be an integer")
    return clamp(value, -128, 127) + 128


def real_to_i8(value: Fraction | int | float) -> int:
    return clamp(round_half_away(Fraction(value) * 128), -128, 127)


def real_to_vgg_rgb01(value: Fraction | int | float) -> Fraction:
    bounded = max(Fraction(-1), min(Fraction(127, 128), Fraction(value)))
    return (128 * bounded + 128) / 255


def q17_multiplier(input_scale: Fraction, weight_scale: Fraction,
                   output_scale: Fraction) -> int:
    raw = round_half_away(input_scale * weight_scale / output_scale * (1 << 17))
    if not -(1 << 17) <= raw < (1 << 17):
        raise ContractError(f"Q17 multiplier {raw} is not signed18; no silent clipping")
    return raw


def requantize(dot: int, bias: int, multiplier: int, *, relu: bool = False) -> tuple[int, bool]:
    if type(bias) is not int or not I32_MIN <= bias <= I32_MAX:
        raise ContractError("bias must be signed INT32")
    if type(multiplier) is not int or not -(1 << 17) <= multiplier < (1 << 17):
        raise ContractError("multiplier must be a sign-extended signed18 Q17 value")
    biased = dot + bias
    overflow = not I32_MIN <= biased <= I32_MAX
    integer = round_half_away(Fraction(clamp(biased, I32_MIN, I32_MAX) * multiplier, 1 << 17))
    integer = clamp(integer, -128, 127)
    return (max(0, integer) if relu else integer), overflow


def replicate_sample(tensor: list, y: int, x: int, channel: int) -> int:
    return tensor[clamp(y, 0, len(tensor)-1)][clamp(x, 0, len(tensor[0])-1)][channel]


def nearest2(tensor: list) -> list:
    return [[list(tensor[y//2][x//2]) for x in range(2*len(tensor[0]))]
            for y in range(2*len(tensor))]


def tensor_shape(tensor: list) -> list[int]:
    if (not isinstance(tensor, list) or not tensor or not isinstance(tensor[0], list)
            or not tensor[0] or not isinstance(tensor[0][0], list) or not tensor[0][0]):
        raise ContractError("expected nonempty HWC list tensor")
    height, width, channels = len(tensor), len(tensor[0]), len(tensor[0][0])
    for row in tensor:
        if len(row) != width:
            raise ContractError("ragged tensor width")
        for pixel in row:
            if len(pixel) != channels or any(type(v) is not int or not -128 <= v <= 127 for v in pixel):
                raise ContractError("ragged channels or non-INT8 tensor value")
    return [height, width, channels]


def convolution(tensor: list, op: dict, weights: list[list[int]], bias: list[int],
                multipliers: list[int]) -> tuple[list, list]:
    height, width, cin = tensor_shape(tensor)
    cout, kernel, stride = op["channels"], op["kernel"], op["stride"]
    terms = kernel * kernel * (1 if op["kind"] == "dw3x3" else cin)
    if (len(weights) != cout or any(len(row) != terms for row in weights)
            or len(bias) != cout or len(multipliers) != cout
            or any(type(w) is not int or not -128 <= w <= 127 for row in weights for w in row)):
        raise ContractError("convolution coefficients do not match original OC/ky/kx/IC tensor")
    result, overflow = [], []
    pad = kernel // 2
    for y in range((height + stride - 1)//stride):
        output_row, overflow_row = [], []
        for x in range((width + stride - 1)//stride):
            pixel, flags = [], []
            for oc in range(cout):
                dot, term = 0, 0
                for ky in range(kernel):
                    for kx in range(kernel):
                        channels = (oc,) if op["kind"] == "dw3x3" else range(cin)
                        for ic in channels:
                            dot += replicate_sample(tensor, y*stride+ky-pad, x*stride+kx-pad, ic) * weights[oc][term]
                            term += 1
                value, flag = requantize(dot, bias[oc], multipliers[oc], relu=op["activation"] == "relu")
                pixel.append(value)
                flags.append(flag)
            output_row.append(pixel)
            overflow_row.append(flags)
        result.append(output_row)
        overflow.append(overflow_row)
    return result, overflow


def residual_value(main: int, skip: int, parameters: dict) -> int:
    if any(type(v) is not int or not -128 <= v <= 127 for v in (main, skip)):
        raise ContractError("residual inputs must be INT8")
    required = {"main_scale", "skip_scale", "common_scale", "output_scale"}
    if not isinstance(parameters, dict) or set(parameters) != required:
        raise ContractError("explicit residual runtime scales required; no unit-scale default")
    scales = {key: positive_scale(parameters[key], key) for key in required}
    # Deliberately generic oracle: production alignment encoding is still OPEN.
    a = clamp(round_half_away(main * scales["main_scale"] / scales["common_scale"]), I32_MIN, I32_MAX)
    b = clamp(round_half_away(skip * scales["skip_scale"] / scales["common_scale"]), I32_MIN, I32_MAX)
    summed = clamp(a + b, I32_MIN, I32_MAX)
    output_multiplier = q17_multiplier(scales["common_scale"], Fraction(1), scales["output_scale"])
    return requantize(summed, 0, output_multiplier)[0]


def fold_batch_norm(weights: list[list[float]], conv_bias: list[float] | None,
                    gamma: list[float], beta: list[float], running_mean: list[float],
                    running_variance: list[float], epsilon: float) -> tuple[list, list]:
    channels = len(weights)
    arrays = (gamma, beta, running_mean, running_variance)
    if (not channels or any(len(a) != channels for a in arrays)
            or not math.isfinite(epsilon) or epsilon <= 0):
        raise ContractError("BN field dimensions/epsilon")
    conv_bias = [0.0]*channels if conv_bias is None else conv_bias
    if len(conv_bias) != channels or any(v < 0 for v in running_variance):
        raise ContractError("BN bias/variance")
    if any(not math.isfinite(v) for a in (*arrays, conv_bias, *weights) for v in a):
        raise ContractError("nonfinite BN/weight value")
    factors = [gamma[c] / math.sqrt(running_variance[c] + epsilon) for c in range(channels)]
    return ([[w*factors[c] for w in weights[c]] for c in range(channels)],
            [(conv_bias[c]-running_mean[c])*factors[c]+beta[c] for c in range(channels)])


def execute_integer(contract: dict, tensor: list, parameters: dict,
                    residual_parameters: dict) -> dict:
    """Execute a small full graph only when every real-scale field is supplied.

    Can validate synthetic diagnostics; never grants production deployment.
    Caller supplies already BN-folded integer weights/bias and actual scales.
    """
    plan = compile_contract(contract, input_shape=tensor_shape(tensor))
    expected = {row["id"] for row in plan["nodes"] if row["kind"] in QUANTIZED}
    residual_names = {row["id"] for row in plan["nodes"] if row["kind"] == "residual_add"}
    if not isinstance(parameters, dict) or set(parameters) != expected:
        raise ContractError("all and only 24 quantized parameter records required")
    if not isinstance(residual_parameters, dict) or set(residual_parameters) != residual_names:
        raise ContractError("all three explicit residual scale records required")
    tensors = {"input": tensor}
    scales = {"input": Fraction(1, 128)}
    overflows = {}
    for op in plan["nodes"]:
        name, source = op["id"], op["inputs"][0]
        if op["kind"] in QUANTIZED:
            p = parameters[name]
            fields = {"weights", "bias", "multipliers", "input_scale", "output_scale", "weight_scales"}
            if not isinstance(p, dict) or set(p) != fields:
                raise ContractError(f"{name}: incomplete coefficient/scale record")
            si, so = positive_scale(p["input_scale"]), positive_scale(p["output_scale"])
            if si != scales[source] or (name == contract["output"]["node"] and so != Fraction(1, 128)):
                raise ContractError(f"{name}: input/final-output scale mismatch")
            if not isinstance(p["weight_scales"], list) or len(p["weight_scales"]) != op["channels"]:
                raise ContractError(f"{name}: per-output-channel scales required")
            computed = [q17_multiplier(si, positive_scale(v), so) for v in p["weight_scales"]]
            if computed != p["multipliers"]:
                raise ContractError(f"{name}: multiplier differs from supplied scales")
            tensors[name], overflows[name] = convolution(tensors[source], op, p["weights"], p["bias"], p["multipliers"])
            scales[name] = so
        elif op["kind"] == "nearest_x2":
            tensors[name], scales[name] = nearest2(tensors[source]), scales[source]
        else:
            skip = op["inputs"][1]
            p = residual_parameters[name]
            if (positive_scale(p.get("main_scale")) != scales[source]
                    or positive_scale(p.get("skip_scale")) != scales[skip]):
                raise ContractError(f"{name}: residual parent scales mismatch")
            main_tensor, skip_tensor = tensors[source], tensors[skip]
            tensors[name] = [[[residual_value(a, b, p) for a, b in zip(mp, sp)]
                              for mp, sp in zip(mr, sr)] for mr, sr in zip(main_tensor, skip_tensor)]
            scales[name] = positive_scale(p["output_scale"])
    return {"output": tensors[contract["output"]["node"]], "tensors": tensors,
            "bias_overflow": overflows, "production_start_allowed": False,
            "residual_alignment_encoding": "REFERENCE_ONLY_HARDWARE_OPEN"}
