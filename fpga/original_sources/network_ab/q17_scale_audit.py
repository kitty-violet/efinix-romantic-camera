"""Audit whether exported INT8 requantization scales fit fixed signed Q17.

The FPGA candidate computes, per output channel::

    q_out = saturate_i8(round_half_away(acc_i32 * multiplier_q17 / 2**17))

where ``multiplier_q17`` must fit signed 18 bits.  For symmetric INT8
quantization the ideal real multiplier is::

    input_scale * weight_scale[channel] / output_scale

This tool deliberately uses only the Python standard library.  It audits the
exported scales and, when calibration accumulators are supplied, compares the
ideal real-scale result with the exact fixed-Q17 integer datapath.  It does not
replace FP32/INT8 image-quality evaluation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_HALF_UP, localcontext
from pathlib import Path
from typing import Any

from model_ab_spec import build_model_a, build_model_b


SCHEMA = "q17-scale-audit-v1"
Q17_FRACTION_BITS = 17
Q17_DENOMINATOR = 1 << Q17_FRACTION_BITS
Q17_MIN = -(1 << 17)
Q17_MAX = (1 << 17) - 1


class AuditInputError(ValueError):
    """Raised when the export audit input is incomplete or inconsistent."""


@dataclass(frozen=True)
class ChannelAudit:
    channel: int
    ideal_multiplier: str
    q17_raw_integer: int
    q17_fits_signed_i18: bool
    q17_multiplier: str | None
    absolute_multiplier_error: str | None
    relative_multiplier_error: str | None
    accumulator_abs_max: int | None
    conservative_output_delta_bound: int | None
    sample_count: int
    integer_mismatch_count: int | None
    max_abs_output_delta: int | None
    ideal_saturation_count: int | None
    q17_saturation_count: int | None


def decimal_value(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise AuditInputError(f"{field} must be a positive finite number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise AuditInputError(f"{field} is not a decimal number: {value!r}") from exc
    if not result.is_finite() or result <= 0:
        raise AuditInputError(f"{field} must be a positive finite number")
    return result


def round_half_away_decimal(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_HALF_UP))


def round_half_away_ratio(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    magnitude = abs(numerator)
    rounded = (magnitude + denominator // 2) // denominator
    return -rounded if numerator < 0 else rounded


def clamp_i8(value: int, relu: bool) -> int:
    if relu and value < 0:
        value = 0
    return max(-128, min(127, value))


def quantize_ideal(accumulator: int, multiplier: Decimal, relu: bool) -> int:
    with localcontext() as context:
        context.prec = 50
        rounded = round_half_away_decimal(Decimal(accumulator) * multiplier)
    return clamp_i8(rounded, relu)


def quantize_q17(accumulator: int, multiplier_q17: int, relu: bool) -> int:
    rounded = round_half_away_ratio(accumulator * multiplier_q17, Q17_DENOMINATOR)
    return clamp_i8(rounded, relu)


def decimal_text(value: Decimal) -> str:
    if value.is_zero():
        return "0"
    return format(value.normalize(), "f")


def parse_optional_per_channel_ints(
    layer: dict[str, Any], field: str, channel_count: int
) -> list[int | None]:
    values = layer.get(field)
    if values is None:
        return [None] * channel_count
    if not isinstance(values, list) or len(values) != channel_count:
        raise AuditInputError(
            f"{layer.get('name', '<unnamed>')}.{field} must contain {channel_count} entries"
        )
    parsed: list[int | None] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AuditInputError(
                f"{layer.get('name', '<unnamed>')}.{field}[{index}] must be a non-negative integer"
            )
        parsed.append(value)
    return parsed


def parse_accumulator_samples(
    layer: dict[str, Any], channel_count: int
) -> list[list[int] | None]:
    samples = layer.get("accumulator_samples")
    if samples is None:
        return [None] * channel_count
    if not isinstance(samples, list) or len(samples) != channel_count:
        raise AuditInputError(
            f"{layer.get('name', '<unnamed>')}.accumulator_samples must contain "
            f"{channel_count} channel lists"
        )
    parsed: list[list[int] | None] = []
    for channel, values in enumerate(samples):
        if not isinstance(values, list):
            raise AuditInputError(
                f"{layer.get('name', '<unnamed>')}.accumulator_samples[{channel}] must be a list"
            )
        channel_values: list[int] = []
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, int):
                raise AuditInputError(
                    f"{layer.get('name', '<unnamed>')}.accumulator_samples"
                    f"[{channel}][{index}] must be an integer"
                )
            if value < -(1 << 31) or value > (1 << 31) - 1:
                raise AuditInputError(
                    f"{layer.get('name', '<unnamed>')}.accumulator_samples"
                    f"[{channel}][{index}] is outside signed INT32"
                )
            channel_values.append(value)
        parsed.append(channel_values)
    return parsed


def require_zero_point_contract(layer: dict[str, Any], channel_count: int) -> None:
    name = layer["name"]
    for field in ("input_zero_point", "output_zero_point"):
        value = layer.get(field)
        if isinstance(value, bool) or value != 0:
            raise AuditInputError(f"{name}.{field} must be explicitly set to integer 0")
    values = layer.get("weight_zero_points")
    if not isinstance(values, list) or len(values) != channel_count:
        raise AuditInputError(
            f"{name}.weight_zero_points must contain {channel_count} integer-zero entries"
        )
    for index, value in enumerate(values):
        if isinstance(value, bool) or value != 0:
            raise AuditInputError(f"{name}.weight_zero_points[{index}] must be integer 0")


def known_operation_contract(model_id: str) -> list[tuple[str, int]] | None:
    builders = {
        "A_BASELINE_8_16_32_R3": build_model_a,
        "B_BALANCED_DECODER_8_16_32_R3": build_model_b,
    }
    builder = builders.get(model_id)
    if builder is None:
        return None
    contract: list[tuple[str, int]] = []
    for layer in builder().layers:
        if layer.operator in ("conv3x3", "dw3x3", "pw1x1"):
            contract.append((layer.name, layer.output.channels))
        elif layer.operator.startswith("dw3x3_s"):
            contract.append((f"{layer.name}.dw", layer.input.channels))
            contract.append((f"{layer.name}.pw", layer.output.channels))
    return contract


def audit_layer(layer: dict[str, Any]) -> dict[str, Any]:
    name = layer.get("name")
    if not isinstance(name, str) or not name.strip():
        raise AuditInputError("every layer needs a non-empty name")
    input_scale = decimal_value(layer.get("input_scale"), f"{name}.input_scale")
    output_scale = decimal_value(layer.get("output_scale"), f"{name}.output_scale")
    raw_weight_scales = layer.get("weight_scales")
    if not isinstance(raw_weight_scales, list) or not raw_weight_scales:
        raise AuditInputError(f"{name}.weight_scales must be a non-empty list")
    weight_scales = [
        decimal_value(value, f"{name}.weight_scales[{index}]")
        for index, value in enumerate(raw_weight_scales)
    ]
    require_zero_point_contract(layer, len(weight_scales))
    relu = layer.get("relu", False)
    if not isinstance(relu, bool):
        raise AuditInputError(f"{name}.relu must be true or false")

    channel_count = len(weight_scales)
    supplied_abs_max = parse_optional_per_channel_ints(layer, "accumulator_abs_max", channel_count)
    samples = parse_accumulator_samples(layer, channel_count)
    channels: list[ChannelAudit] = []

    for channel, weight_scale in enumerate(weight_scales):
        with localcontext() as context:
            context.prec = 50
            ideal = input_scale * weight_scale / output_scale
            q17_raw = round_half_away_decimal(ideal * Q17_DENOMINATOR)
            fits = Q17_MIN <= q17_raw <= Q17_MAX
            represented = Decimal(q17_raw) / Decimal(Q17_DENOMINATOR) if fits else None
            absolute_error = abs(represented - ideal) if represented is not None else None
            relative_error = absolute_error / abs(ideal) if absolute_error is not None else None

        channel_samples = samples[channel]
        observed_abs_max = (
            max((abs(value) for value in channel_samples), default=0)
            if channel_samples is not None
            else None
        )
        declared_abs_max = supplied_abs_max[channel]
        accumulator_abs_max = declared_abs_max
        if observed_abs_max is not None:
            accumulator_abs_max = max(declared_abs_max or 0, observed_abs_max)

        delta_bound = None
        if fits and accumulator_abs_max is not None and absolute_error is not None:
            with localcontext() as context:
                context.prec = 50
                # The extra one covers a possible difference at an integer
                # half-way rounding boundary.
                error_product = Decimal(accumulator_abs_max) * absolute_error
                delta_bound = int(error_product.to_integral_value(rounding=ROUND_CEILING)) + 1

        mismatch_count = None
        max_abs_output_delta = None
        ideal_saturation_count = None
        q17_saturation_count = None
        if channel_samples is not None and fits:
            mismatch_count = 0
            max_abs_output_delta = 0
            ideal_saturation_count = 0
            q17_saturation_count = 0
            for accumulator in channel_samples:
                ideal_output = quantize_ideal(accumulator, ideal, relu)
                q17_output = quantize_q17(accumulator, q17_raw, relu)
                mismatch_count += int(ideal_output != q17_output)
                max_abs_output_delta = max(max_abs_output_delta, abs(ideal_output - q17_output))
                ideal_saturation_count += int(ideal_output in (-128, 127))
                q17_saturation_count += int(q17_output in (-128, 127))

        channels.append(
            ChannelAudit(
                channel=channel,
                ideal_multiplier=decimal_text(ideal),
                q17_raw_integer=q17_raw,
                q17_fits_signed_i18=fits,
                q17_multiplier=decimal_text(represented) if represented is not None else None,
                absolute_multiplier_error=(
                    decimal_text(absolute_error) if absolute_error is not None else None
                ),
                relative_multiplier_error=(
                    decimal_text(relative_error) if relative_error is not None else None
                ),
                accumulator_abs_max=accumulator_abs_max,
                conservative_output_delta_bound=delta_bound,
                sample_count=len(channel_samples) if channel_samples is not None else 0,
                integer_mismatch_count=mismatch_count,
                max_abs_output_delta=max_abs_output_delta,
                ideal_saturation_count=ideal_saturation_count,
                q17_saturation_count=q17_saturation_count,
            )
        )

    fit_count = sum(channel.q17_fits_signed_i18 for channel in channels)
    sample_count = sum(channel.sample_count for channel in channels)
    sampled_channels = sum(channel.sample_count > 0 for channel in channels)
    mismatch_count = sum(channel.integer_mismatch_count or 0 for channel in channels)
    return {
        "name": name,
        "relu": relu,
        "channel_count": channel_count,
        "q17_fit_count": fit_count,
        "q17_unfit_count": channel_count - fit_count,
        "q17_range_pass": fit_count == channel_count,
        "samples_evaluated": sample_count,
        "sampled_channel_count": sampled_channels,
        "integer_mismatch_count": mismatch_count if sample_count else None,
        "sample_equivalence_pass": mismatch_count == 0 if sample_count else None,
        "channels": [asdict(channel) for channel in channels],
    }


def audit_document(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("schema") != SCHEMA:
        raise AuditInputError(f"schema must be {SCHEMA!r}")
    model_id = document.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise AuditInputError("model_id must be a non-empty string")
    layers = document.get("layers")
    if not isinstance(layers, list) or not layers:
        raise AuditInputError("layers must be a non-empty list")
    contract = known_operation_contract(model_id)
    if contract is not None:
        actual_names = [layer.get("name") if isinstance(layer, dict) else None for layer in layers]
        expected_names = [name for name, _channels in contract]
        if actual_names != expected_names:
            raise AuditInputError(
                f"{model_id} layer list/order mismatch; expected {expected_names}, got {actual_names}"
            )
        for layer, (name, expected_channels) in zip(layers, contract):
            weight_scales = layer.get("weight_scales")
            actual_channels = len(weight_scales) if isinstance(weight_scales, list) else 0
            if actual_channels != expected_channels:
                raise AuditInputError(
                    f"{name}.weight_scales must contain {expected_channels} output channels, "
                    f"got {actual_channels}"
                )
    layer_results = [audit_layer(layer) for layer in layers]
    total_channels = sum(layer["channel_count"] for layer in layer_results)
    unfit_channels = sum(layer["q17_unfit_count"] for layer in layer_results)
    total_samples = sum(layer["samples_evaluated"] for layer in layer_results)
    total_mismatches = sum(layer["integer_mismatch_count"] or 0 for layer in layer_results)
    return {
        "schema": SCHEMA,
        "model_id": model_id,
        "q17_fraction_bits": Q17_FRACTION_BITS,
        "q17_signed_integer_range": [Q17_MIN, Q17_MAX],
        "layer_count": len(layer_results),
        "channel_count": total_channels,
        "unfit_channel_count": unfit_channels,
        "hardware_range_pass": unfit_channels == 0,
        "samples_evaluated": total_samples,
        "integer_mismatch_count": total_mismatches if total_samples else None,
        "sample_equivalence_pass": total_mismatches == 0 if total_samples else None,
        "quality_gate": "UNMEASURED",
        "quality_gate_note": (
            "This scale audit cannot replace fixed-image/video FP32-versus-INT8 quality evaluation."
        ),
        "layers": layer_results,
    }


def print_summary(result: dict[str, Any]) -> None:
    print(
        f"MODEL {result['model_id']} layers={result['layer_count']} "
        f"channels={result['channel_count']} unfit={result['unfit_channel_count']} "
        f"range={'PASS' if result['hardware_range_pass'] else 'FAIL'}"
    )
    for layer in result["layers"]:
        sample_text = (
            f"samples={layer['samples_evaluated']} mismatches={layer['integer_mismatch_count']}"
            if layer["samples_evaluated"]
            else "samples=UNMEASURED"
        )
        print(
            f"{layer['name']}\tchannels={layer['channel_count']}\t"
            f"unfit={layer['q17_unfit_count']}\t{sample_text}"
        )
    print("QUALITY UNMEASURED (requires FP32/INT8 fixed-image and video evaluation)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit fixed signed-Q17 requantization scales")
    parser.add_argument("input", type=Path, help="q17-scale-audit-v1 JSON exported by QAT")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--fail-on-unfit", action="store_true")
    args = parser.parse_args()

    try:
        document = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise AuditInputError("top-level JSON value must be an object")
        result = audit_document(document)
    except (OSError, json.JSONDecodeError, AuditInputError) as exc:
        parser.error(str(exc))

    print_summary(result)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if args.fail_on_unfit and not result["hardware_range_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
