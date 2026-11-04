import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddr_memory_map import (  # noqa: E402
    DEFAULT_CONFIG, RAW_PAYLOAD_BYTES, RTL_HEADER, load_memory_map, sync_headers, validate_document,
)


class DdrMemoryMapTests(unittest.TestCase):
    def setUp(self):
        self.document = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def test_frozen_partition_and_explicit_cpu_alias(self):
        memory_map = load_memory_map()
        self.assertEqual(len(memory_map.regions), 15)
        for index, suffix in enumerate("abcdef"):
            slot = memory_map.region(f"frame_{suffix}")
            self.assertEqual((slot.base, slot.size), (index * 0x400000, 0x400000))
        self.assertEqual(memory_map.region("unallocated").base, 0x06C00000)
        self.assertEqual(memory_map.region("unallocated").end, 0x10000000)
        for name, base in (("isolation", 0x01800000), ("feature0", 0x02000000),
                           ("feature1", 0x03000000), ("temporary", 0x04000000),
                           ("development", 0x05000000)):
            self.assertEqual(memory_map.region(name).base, base)
        constants = memory_map.constants()
        self.assertEqual(constants["CPU_SOURCE_BASE"], 0xE1000000)
        self.assertEqual(constants["CPU_SOURCE_LIMIT"], 0xE1FFFFFF)
        self.assertEqual(constants["CPU_DDR_BASE"], 0x05000000)
        self.assertEqual(constants["CPU_DDR_LIMIT"], 0x05FFFFFF)
        self.assertEqual((constants["RAW_BASE"], constants["RAW_BANK_STRIDE"],
                          constants["RAW_BANK_COUNT"]), (0x06000000, 0x00400000, 3))

    def test_raw_pool_prevents_legacy_rgb_overwrite(self):
        memory_map = load_memory_map()
        rgb = [region for region in memory_map.regions if region.kind == "frame"]

        def overlaps(base, end, region):
            return base < region.end and region.base < end

        # Actual v23 top: WORDS_PER_FRAME=518400, BASE=0, STRIDE=0x007e9200.
        # Each 32-bit camera word packs four RAW8 pixels, so the payload is
        # 0x1fa400 bytes, not the full stride-sized nominal reservation.
        self.assertEqual(RAW_PAYLOAD_BYTES, 518400 * 4)
        legacy_ranges = [(index * 0x007E9200, index * 0x007E9200 + RAW_PAYLOAD_BYTES)
                         for index in range(3)]
        self.assertEqual([(base, end - 1) for base, end in legacy_ranges],
                         [(0x00000000, 0x001FA3FF), (0x007E9200, 0x009E35FF),
                          (0x00FD2400, 0x011CC7FF)])
        self.assertEqual([[region.name for region in rgb if overlaps(base, end, region)]
                          for base, end in legacy_ranges],
                         [["frame_a"], ["frame_b", "frame_c"], ["frame_d", "frame_e"]])
        for index, raw in enumerate(memory_map.raw_frames):
            self.assertEqual((raw.base, raw.size), (0x06000000 + index * 0x400000, 0x400000))
            self.assertLessEqual(raw.base + RAW_PAYLOAD_BYTES, raw.end)
            for other in memory_map.regions:
                if other != raw:
                    self.assertFalse(overlaps(raw.base, raw.end, other))

    def test_reject_invalid_raw_pool_geometry(self):
        def by_name(document, name):
            return next(region for region in document["regions"] if region["name"] == name)

        document = copy.deepcopy(self.document)
        by_name(document, "raw_frame_2")["kind"] = "reserved"
        with self.assertRaisesRegex(ValueError, "exactly three"):
            validate_document(document)

        document = copy.deepcopy(self.document)
        by_name(document, "raw_frame_0")["size"] = "0x00300000"
        by_name(document, "raw_frame_1").update(base="0x06300000", size="0x00500000")
        with self.assertRaisesRegex(ValueError, "equal size"):
            validate_document(document)

        document = copy.deepcopy(self.document)
        by_name(document, "raw_frame_1")["kind"] = "reserved"
        document["regions"].append({"name": "raw_frame_3", "kind": "raw_frame",
                                    "base": "0x06c00000", "size": "0x00400000"})
        by_name(document, "unallocated").update(base="0x07000000", size="0x09000000")
        with self.assertRaisesRegex(ValueError, "contiguous"):
            validate_document(document)

        document = copy.deepcopy(self.document)
        for index in range(3):
            by_name(document, f"raw_frame_{index}").update(
                base=0x06000000 + index * 0x100000, size=0x100000)
        by_name(document, "unallocated").update(base="0x06300000", size="0x09d00000")
        with self.assertRaisesRegex(ValueError, "RAW8 payload"):
            validate_document(document)

    def test_reject_overlap_range_and_alignment_errors(self):
        for field, value, message in (
            ("base", "0x00000000", "overlapping"),
            ("base", "0x10000000", "outside physical"),
            ("base", "0x00400001", "aligned"),
            ("size", "0x00000000", "zero size"),
        ):
            with self.subTest(field=field, value=value):
                document = copy.deepcopy(self.document)
                document["regions"][1][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_document(document)

    def test_reject_cpu_region_escape_and_address_wrap(self):
        for field, value, message in (
            ("target_region", "frame_a", "development region"),
            ("target_region", "missing", "unknown DDR region"),
            ("target_offset", "0x10", "exceeds"),
            ("target_offset", "0x1", "aligned"),
            ("source_base", "0xfffffff0", "wrap"),
        ):
            with self.subTest(field=field, value=value):
                document = copy.deepcopy(self.document)
                document["cpu_window"][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_document(document)

    def test_check_detects_generated_drift_without_rewriting(self):
        memory_map = load_memory_map()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sync_headers(memory_map, check=False, root=root)
            sync_headers(memory_map, check=True, root=root)
            stale = root / RTL_HEADER
            stale.write_text("stale\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "drift"):
                sync_headers(memory_map, check=True, root=root)
            self.assertEqual(stale.read_text(encoding="utf-8"), "stale\n")

    def test_schedule_addresses_and_existing_budget_stay_unchanged(self):
        from hardware_schedule_model import (
            RGB_INPUT_A_BASE, RGB_OUTPUT_C_BASE, FEATURE0_BASE, FEATURE1_BASE,
            build_tile_dma_command_streams, evaluate_candidate,
        )
        self.assertEqual((RGB_INPUT_A_BASE, RGB_OUTPUT_C_BASE, FEATURE0_BASE, FEATURE1_BASE),
                         (0x00000000, 0x00800000, 0x02000000, 0x03000000))
        streams = build_tile_dma_command_streams(320)
        self.assertEqual({name: len(commands) for name, commands in streams.items()},
                         {"main": 5760, "skip": 2160, "write": 5760})
        result = evaluate_candidate(320, "sapphire_lite_4k_i1_local_ram_saver_measured")
        self.assertEqual(result["ddr"]["traffic_bytes_per_frame"], 35825040)
        self.assertEqual(result["cycles"]["estimated_cycles"], 4439524)


if __name__ == "__main__":
    unittest.main()
