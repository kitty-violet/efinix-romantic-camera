import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hardware_schedule_model import (  # noqa: E402
    AXI_DATA_BYTES,
    FEATURE0_BASE,
    FEATURE1_BASE,
    RAM_HARD_LIMIT_BLOCKS,
    RAM_TARGET_BLOCKS,
    RGB_INPUT_A_BASE,
    RGB_OUTPUT_C_BASE,
    build_dma_context,
    build_groups,
    build_tile_dma_command_streams,
    build_tile_descriptor_sequence,
    build_tile_manifest,
    evaluate_candidate,
    split_axi_bursts,
)


class HardwareScheduleModelTests(unittest.TestCase):
    def test_320_bandwidth_gate_counts_transferred_alignment_bytes(self) -> None:
        # Independently frozen sums of the 128-bit row command trace. Logical
        # RGB halo payload excludes prefix/suffix bytes, but the bus carries them.
        result = evaluate_candidate(320, "sapphire_lite_4k_i1_local_ram_saver_measured")
        ddr = result["ddr"]
        self.assertEqual(ddr["traffic_bytes_per_frame"], 35_825_040)
        self.assertEqual(ddr["axi_alignment_overhead_bytes_per_frame"], 36_720)
        self.assertEqual(ddr["axi_bus_bytes_per_frame_lower_bound"], 35_861_760)
        self.assertEqual(ddr["axi_bus_bytes_per_second_at_30fps_lower_bound"], 1_075_852_800)
        self.assertAlmostEqual(
            ddr["required_measured_sustained_bytes_per_second_for_15pct_margin"],
            1_265_709_176.4705882,
        )
        self.assertIn("CPU", ddr["warning"])

    def test_320_dma_stream_counts_keep_read_and_write_heights_independent(self) -> None:
        streams = build_tile_dma_command_streams(320)
        self.assertEqual(len(streams["main"]), 5_760)
        self.assertEqual(len(streams["skip"]), 2_160)
        self.assertEqual(len(streams["write"]), 5_760)

        encoder_main = [c for c in streams["main"] if c["group_id"] == 0]
        encoder_write = [c for c in streams["write"] if c["group_id"] == 0]
        decoder_main = [c for c in streams["main"] if c["group_id"] == 2]
        decoder_write = [c for c in streams["write"] if c["group_id"] == 2]
        self.assertEqual(len(encoder_main), 4 * 720)
        self.assertEqual(len(encoder_write), 4 * 180)
        self.assertEqual(len(decoder_main), 4 * 180)
        self.assertEqual(len(decoder_write), 4 * 720)

    def test_dma_ping_pong_ownership_matches_three_residual_blocks(self) -> None:
        residual_descriptors = build_tile_descriptor_sequence(320)[4:16:4]
        ownership = []
        for descriptor in residual_descriptors:
            context = build_dma_context(descriptor)
            ownership.append(
                (
                    context["main"]["base"],
                    context["skip"]["base"],
                    context["write"]["base"],
                )
            )
        self.assertEqual(
            ownership,
            [
                (FEATURE0_BASE, FEATURE0_BASE, FEATURE1_BASE),
                (FEATURE1_BASE, FEATURE1_BASE, FEATURE0_BASE),
                (FEATURE0_BASE, FEATURE0_BASE, FEATURE1_BASE),
            ],
        )
        sequence = build_tile_descriptor_sequence(320)
        self.assertEqual(build_dma_context(sequence[0])["main"]["base"], RGB_INPUT_A_BASE)
        self.assertEqual(build_dma_context(sequence[-1])["write"]["base"], RGB_OUTPUT_C_BASE)

    def test_dma_write_commands_cover_each_destination_once_without_overlap(self) -> None:
        streams = build_tile_dma_command_streams(320)
        for group_id, residual_block, width, height, bytes_per_pixel in (
            (0, 0, 320, 180, 32),
            (1, 1, 320, 180, 32),
            (1, 2, 320, 180, 32),
            (1, 3, 320, 180, 32),
            (2, 0, 1280, 720, 3),
        ):
            commands = [
                command
                for command in streams["write"]
                if command["group_id"] == group_id
                and command["residual_block"] == residual_block
            ]
            self.assertEqual(len(commands), 4 * height)
            for row in range(height):
                row_commands = [c for c in commands if c["row"] == row]
                self.assertEqual(len(row_commands), 4)
                intervals = sorted((c["x0"], c["x1"]) for c in row_commands)
                self.assertEqual(intervals[0][0], 0)
                self.assertEqual(intervals[-1][1], width - 1)
                for left, right in zip(intervals, intervals[1:]):
                    self.assertEqual(left[1] + 1, right[0])
                self.assertEqual(
                    sum(c["payload_bytes"] for c in row_commands),
                    width * bytes_per_pixel,
                )

    def test_dma_read_alignment_contains_exact_payload_and_bursts_never_cross_4k(self) -> None:
        streams = build_tile_dma_command_streams(320)
        saw_rgb_prefix = False
        saw_4k_split = False
        for stream in ("main", "skip", "write"):
            for command in streams[stream]:
                self.assertEqual(command["address"] % AXI_DATA_BYTES, 0)
                self.assertEqual(
                    command["beats"] * AXI_DATA_BYTES,
                    command["prefix_bytes"]
                    + command["payload_bytes"]
                    + command["suffix_bytes"],
                )
                if stream == "main" and command["prefix_bytes"]:
                    saw_rgb_prefix = True
                bursts = split_axi_bursts(command["address"], command["beats"])
                saw_4k_split |= len(bursts) > 1
                self.assertEqual(sum(beats for _, beats in bursts), command["beats"])
                for address, beats in bursts:
                    self.assertLessEqual(beats, 128)
                    self.assertLessEqual(
                        (address & 0xFFF) + beats * AXI_DATA_BYTES, 4096
                    )
        self.assertTrue(saw_rgb_prefix)
        self.assertTrue(saw_4k_split)

    def test_320_runtime_descriptor_sequence_repeats_three_residual_blocks(self) -> None:
        sequence = build_tile_descriptor_sequence(320)
        self.assertEqual(len(sequence), 20)
        self.assertEqual([item["group_id"] for item in sequence[:4]], [0] * 4)
        self.assertEqual([item["group_id"] for item in sequence[4:16]], [1] * 12)
        self.assertEqual([item["group_id"] for item in sequence[16:]], [2] * 4)
        self.assertEqual(
            [item["residual_block"] for item in sequence[4:16]],
            [1] * 4 + [2] * 4 + [3] * 4,
        )

    def test_320_tile_manifest_has_exact_nonoverlapping_commit_coverage(self) -> None:
        manifest = build_tile_manifest(320)
        for group in manifest["groups"]:
            tiles = group["tiles"]
            self.assertEqual(len(tiles), 4)
            self.assertEqual(tiles[0]["commit_interval"]["x0"], 0)
            self.assertEqual(
                tiles[-1]["commit_interval"]["x1"], group["output_width"] - 1
            )
            for left, right in zip(tiles, tiles[1:]):
                self.assertEqual(
                    left["commit_interval"]["x1"] + 1,
                    right["commit_interval"]["x0"],
                )

    def test_320_tile_manifest_matches_frozen_source_halo_windows(self) -> None:
        manifest = build_tile_manifest(320)
        groups = {group["name"]: group for group in manifest["groups"]}
        expected = {
            "encoder": [(0, 320), (316, 640), (636, 960), (956, 1279)],
            "residual": [(0, 81), (78, 161), (158, 241), (238, 319)],
            "decoder": [(0, 82), (77, 162), (157, 242), (237, 319)],
        }
        for name, intervals in expected.items():
            actual = [
                (tile["source_interval"]["x0"], tile["source_interval"]["x1"])
                for tile in groups[name]["tiles"]
            ]
            self.assertEqual(actual, intervals)

    def test_tile_manifest_propagates_exact_intervals_between_steps(self) -> None:
        manifest = build_tile_manifest(320)
        for group in manifest["groups"]:
            for tile in group["tiles"]:
                steps = tile["steps"]
                self.assertEqual(tile["source_interval"], steps[0]["input_interval"])
                self.assertEqual(tile["commit_interval"], steps[-1]["output_interval"])
                for step in steps:
                    self.assertGreaterEqual(step["input_interval"]["x0"], 0)
                    self.assertLess(step["input_interval"]["x1"], step["input_width"])
                    self.assertGreaterEqual(step["output_interval"]["x0"], 0)
                    self.assertLess(step["output_interval"]["x1"], step["output_width"])
                for previous, following in zip(steps, steps[1:]):
                    self.assertEqual(
                        previous["output_interval"], following["input_interval"]
                    )

    def test_full_width_decoder_line_buffers_are_not_feasible(self) -> None:
        _, _, decoder = build_groups(1280)
        self.assertEqual(64, decoder.line_buffer_blocks)
        result = evaluate_candidate(1280, "onchip32k_plus_8k_caches_estimator")
        self.assertFalse(result["ram"]["gate_85_percent"])

    def test_tile_256_measured_sapphire_does_not_meet_early_ram_gate(self) -> None:
        result = evaluate_candidate(256, "sapphire_sc431_standard_measured")
        self.assertGreater(result["ram"]["total"], RAM_TARGET_BLOCKS)
        self.assertFalse(result["ram"]["gate_85_percent"])

    def test_traffic_includes_frozen_buffers_and_passes_only_theoretical_gate(self) -> None:
        result = evaluate_candidate(256, "sapphire_sc431_standard_measured")
        traffic = result["ddr"]["traffic_breakdown"]
        self.assertEqual(2_073_600, traffic["camera_raw_write"])
        self.assertEqual(2_764_800, traffic["rgb_input_write"])
        self.assertEqual(2_764_800, traffic["rgb_output_write"])
        self.assertTrue(result["ddr"]["theoretical_gate_15pct_margin"])
        self.assertGreater(
            result["ddr"]["required_measured_sustained_bytes_per_second_for_15pct_margin"],
            result["ddr"]["payload_bytes_per_second_at_30fps"],
        )

    def test_tiling_never_reduces_operation_work_below_nominal(self) -> None:
        result = evaluate_candidate(256, "sapphire_sc431_standard_measured")
        self.assertGreaterEqual(result["cycles"]["tile_mac_overhead_percent"], 0.0)

    def test_tile_256_independent_64bit_line_ports_fit_ideal_group_cycles(self) -> None:
        result = evaluate_candidate(256, "sapphire_sc431_standard_measured")
        self.assertTrue(result["line_port_64bit"]["all_groups_prefetch_overlap_gate"])
        decoder = next(
            group
            for group in result["line_port_64bit"]["groups"]
            if group["group"] == "decoder"
        )
        self.assertGreater(decoder["port_utilization_percent"], 80.0)

    def test_tile_320_measured_sapphire_exceeds_hard_90_percent_limit(self) -> None:
        result = evaluate_candidate(320, "sapphire_sc431_standard_measured")
        scenarios = result["ram"]["scenarios"]
        self.assertFalse(scenarios["port_aware_current_scaler"]["gate_85_percent"])
        self.assertFalse(
            scenarios["port_aware_area_downscaler_candidate"]["gate_85_percent"]
        )
        self.assertGreater(scenarios["port_aware_area_downscaler_candidate"]["total"], 230)

    def test_tile_320_lite_sapphire_passes_hard_but_not_target_ram_gate(self) -> None:
        result = evaluate_candidate(320, "sapphire_lite_4k_i4_measured")
        scenario = result["ram"]["scenarios"]["port_aware_area_downscaler_candidate"]
        self.assertEqual(225, scenario["total"])
        self.assertLessEqual(scenario["total"], RAM_HARD_LIMIT_BLOCKS)
        self.assertTrue(scenario["gate_90_percent"])
        self.assertGreater(scenario["total"], RAM_TARGET_BLOCKS)
        self.assertFalse(scenario["gate_85_percent"])

    def test_cacheless_lite_plus_control_logic_reaches_85_percent_target(self) -> None:
        result = evaluate_candidate(320, "sapphire_lite_2k_nocache_measured")
        scenarios = result["ram"]["scenarios"]
        area_only = scenarios["port_aware_area_downscaler_candidate"]
        control_logic = scenarios["area_downscaler_control_in_logic_candidate"]
        self.assertEqual(219, area_only["total"])
        self.assertFalse(area_only["gate_85_percent"])
        self.assertEqual(217, control_logic["total"])
        self.assertTrue(control_logic["gate_85_percent"])

    def test_lite_local_ram_saver_reaches_85_percent_without_global_side_effects(self) -> None:
        result = evaluate_candidate(320, "sapphire_lite_4k_i1_local_ram_saver_measured")
        scenarios = result["ram"]["scenarios"]
        area_only = scenarios["port_aware_area_downscaler_candidate"]
        control_logic = scenarios["area_downscaler_control_in_logic_candidate"]
        self.assertEqual(214, area_only["total"])
        self.assertTrue(area_only["gate_85_percent"])
        self.assertEqual(212, control_logic["total"])
        self.assertTrue(control_logic["gate_85_percent"])
        self.assertIn("v17 integrated pressure mapping", control_logic["evidence"])
        self.assertTrue(result["ddr"]["theoretical_gate_15pct_margin"])

    def test_lite_local_balanced_is_hard_pass_but_one_block_over_target(self) -> None:
        result = evaluate_candidate(320, "sapphire_lite_4k_i1_local_balanced_measured")
        scenario = result["ram"]["scenarios"]["port_aware_area_downscaler_candidate"]
        self.assertEqual(218, scenario["total"])
        self.assertFalse(scenario["gate_85_percent"])
        self.assertTrue(scenario["gate_90_percent"])


if __name__ == "__main__":
    unittest.main()
