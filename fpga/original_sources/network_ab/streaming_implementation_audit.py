"""Expose implementation gaps behind the historical ideal tiled budget.

This is sensitivity analysis, not a measured frame-time or a replacement
network specification. It reuses the actual model B and halo counts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import hardware_schedule_model as schedule


def audit() -> dict:
    encoder, residual, decoder = schedule.build_groups(320)
    overrides = dict(encoder.operation_output_values)
    overrides.update(schedule.expand_residual_overrides(residual.operation_output_values))
    overrides.update(decoder.operation_output_values)
    engine = next(candidate for candidate in schedule.throughput.CANDIDATES
                  if candidate.name == "MUL18_Q17_112MAC_24RQ")
    model = schedule.throughput.model_spec.build_model_b()
    ideal = schedule.throughput.evaluate(model, engine, output_value_overrides=overrides)
    scenarios = []
    for name, affected in (("historical_zero_inter_dot_bubbles", set()),
                           ("measured_dw_II11_sensitivity_only", {"dw3x3"}),
                           ("two_bubbles_all_dot_kinds_sensitivity", {"dw3x3", "pw1x1", "conv3x3"})):
        layers = []
        for layer in ideal["layers"]:
            bubble_cycles = layer["mac_batches"] * (2 if layer["kind"] in affected else 0)
            mac_cycles = layer["mac_cycles"] + bubble_cycles
            layers.append({"name": layer["name"], "kind": layer["kind"],
                           "output_values": layer["output_values"],
                           "mac_batches": layer["mac_batches"],
                           "inter_dot_bubble_cycles": bubble_cycles,
                           "mac_cycles": mac_cycles,
                           "requant_cycles": layer["requant_cycles"],
                           "cycles": max(mac_cycles, layer["requant_cycles"])})
        cycles = sum(layer["cycles"] for layer in layers)
        scenarios.append({"name": name, "cycles": cycles, "ms_at_150mhz": cycles / 150000,
                          "early_30ms_gate": cycles <= 4500000,
                          "frame_30fps_gate": cycles <= 5000000, "layers": layers})
    # Each residual context is exercised three times. A shared issue unit has
    # to serve their sum, whereas independent stage ports may overlap.
    port_words = sum(int(line["row_read_words_64bit"]) * repetitions
                     for group, repetitions in ((encoder, 1), (residual, 3), (decoder, 1))
                     for line in group.line_buffers)
    dw_values = sum(layer["output_values"] for layer in ideal["layers"] if layer["kind"] == "dw3x3")
    # Actual one-dot wrapper measurements, not an extrapolation from bare MAC
    # II. RGB Cout3 still takes one padded OC8 command for every pixel.
    pw_specs = {}
    for layer in model.layers:
        if layer.operator == "pw1x1":
            pw_specs[layer.name] = layer
        elif layer.operator.startswith("dw3x3_s"):
            pw_specs[layer.name + ".pw"] = layer
    measured_completion = {8: 105, 16: 121, 32: 153}
    pw_layers = []
    for layer in ideal["layers"]:
        if layer["kind"] != "pw1x1":
            continue
        spec = pw_specs[layer["name"]]
        pixels, remainder = divmod(layer["output_values"], spec.output.channels)
        assert remainder == 0
        groups_per_pixel = (spec.output.channels + 7) // 8
        dots = pixels * groups_per_pixel
        terms = spec.input.channels
        pw_layers.append({"name": layer["name"], "input_channels": terms,
                          "output_channels": spec.output.channels, "pixels_including_halo": pixels,
                          "oc8_commands": dots, "weights_per_dot": terms,
                          "store_words_if_reloaded_per_dot": dots * (terms + 9),
                          "last_term_lower_bound_cycles": dots * (71 + 2 * terms),
                          "measured_serial_service_cycles": dots * measured_completion[terms]})
    pw_lower = sum(layer["last_term_lower_bound_cycles"] for layer in pw_layers)
    pw_service = sum(layer["measured_serial_service_cycles"] for layer in pw_layers)
    head = next(layer for layer in ideal["layers"] if layer["name"] == "rgb_head")
    head_padded_values = next(layer["oc8_commands"] * 8 for layer in pw_layers if layer["name"] == "rgb_head")
    head_padded_mac = ((head_padded_values + 223) // 224) * 8 + 2
    head_padded_rq = (head_padded_values + 23) // 24 + 3
    weights_capacity = 5 * 256 * 8
    return {
        "schema": 1, "model": model.model_id, "output_tile_width": 320,
        "assumed_logical_mac_lanes": 224, "assumed_requant_lanes": 24,
        "scenarios": scenarios,
        "one_serial_window_issue_port": {"minimum_cycles": port_words,
                                          "minimum_ms_at_150mhz": port_words / 150000},
        "one_dw8_slice_all_dw_only": {"output_values": dw_values,
                                       "minimum_cycles": (dw_values + 7) // 8 * 11,
                                       "minimum_ms_at_150mhz": ((dw_values + 7) // 8 * 11) / 150000},
        "raw_weights": {"bytes": model.weight_count, "five_256x64_bank_bytes": weights_capacity,
                        "capacity_shortfall_bytes": max(0, model.weight_count - weights_capacity),
                        "bias_scale_alignment_included": False},
        "one_serial_banked_pw_wrapper": {
            "scope": "Correctness wrapper repeated serially; NOT the planned shared array.",
            "measurement_evidence": "tmp/modelsim-pw-banked-operator/attempt-20260928-234817-143",
            "commands": sum(layer["oc8_commands"] for layer in pw_layers),
            "last_term_lower_bound_cycles": pw_lower,
            "last_term_lower_bound_ms_at_150mhz": pw_lower / 150000,
            "serial_measured_service_cycles_excluding_dispatch_gaps": pw_service,
            "serial_measured_service_ms_at_150mhz": pw_service / 150000,
            "layers": pw_layers},
        "replicating_28_independent_pw_slices": {
            "mac_physical_dsps": 28 * 4, "requant_physical_dsps": 28 * 2,
            "total_physical_dsps": 28 * 6, "device_dsps": 160,
            "fits": 28 * 6 <= 160,
            "required_direction": "Share 24 RQ lanes; do not replicate 28 independent RQ slices."},
        "rgb_oc8_padding_sensitivity": {
            "scope": "Sensitivity only: retaining current OC8 padding in a 224-MAC/24-RQ array.",
            "valid_values": head["output_values"], "padded_values": head_padded_values,
            "historical_head_cycles": head["estimated_cycles"],
            "padded_mac_cycles": head_padded_mac, "padded_rq_cycles": head_padded_rq,
            "frame_cycles_if_all_padding_computed_and_requantized": ideal["total_cycles"] - head["estimated_cycles"] + max(head_padded_mac, head_padded_rq),
            "frame_cycles_if_padding_requant_skipped_only": ideal["total_cycles"] - head["estimated_cycles"] + max(head_padded_mac, head["requant_cycles"])},
        "limits": ["No measured full-array scheduler, DDR stalls or actual frame-time here.",
                   "DW II11 is the retained old isolated slice; the newer II9 slice has separate numeric/physical evidence.",
                   "Historical zero-bubble assumptions require matching DSP packing and port evidence.",
                   "The single issue-port bound assumes all stage reads serialized; it forbids blindly sharing one port.",
                   "Weight-bank folding tests do not prove capacity for deployed coefficients."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = audit()
    for scenario in result["scenarios"]:
        print(f"{scenario['name']}: {scenario['cycles']} cycles, {scenario['ms_at_150mhz']:.6f} ms, "
              f"early30ms={scenario['early_30ms_gate']}")
    print("serial_window_port:", result["one_serial_window_issue_port"])
    print("raw_weights:", result["raw_weights"])
    pw = result["one_serial_banked_pw_wrapper"]
    print("serial_banked_pw:", pw["commands"], "OC8 commands; last-term lower-bound ms=",
          pw["last_term_lower_bound_ms_at_150mhz"], "measured-service ms=", pw["serial_measured_service_ms_at_150mhz"])
    print("replicated_28_slices:", result["replicating_28_independent_pw_slices"])
    print("rgb_padding_sensitivity:", result["rgb_oc8_padding_sensitivity"])
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
