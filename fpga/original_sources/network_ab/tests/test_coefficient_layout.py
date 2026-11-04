import hashlib
import json
import struct
import sys
import tempfile
import unittest
import zlib
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coefficient_layout import (  # noqa: E402
    ALIGNMENT, BANKS, HEADER, IMAGE_FLAGS, I32_MIN, I32_MAX, LayoutError,
    OP_ENTRY, Q17_MAX, Q17_MIN, SECTION_ENTRY, bank_location, build_layout,
    capacity_report, decode_image, emit_artifacts, encode_parameters,
    expanded_operations, load_banks, read_banks, synthetic_parameters,
    weight_positions,
)
from model_ab_spec import build_model_b  # noqa: E402
from q17_scale_audit import known_operation_contract  # noqa: E402


class CoefficientLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = build_layout()
        cls.parameters = synthetic_parameters()
        cls.image = encode_parameters(cls.parameters)
        cls.sha = hashlib.sha256(cls.image).hexdigest()

    def repair_checksums(self, image):
        """Keep integrity checks valid to test deeper padding/schema rejection."""
        image = bytearray(image)
        payload_start = self.layout["sections"][1]["offset"]
        struct.pack_into("<I", image, 32, zlib.crc32(image[payload_start:]) & 0xFFFFFFFF)
        for index, section in enumerate(self.layout["sections"]):
            payload = image[section["offset"]:section["offset"] + section["length"]]
            struct.pack_into("<I", image, HEADER.size + index * SECTION_ENTRY.size + 16,
                             zlib.crc32(payload) & 0xFFFFFFFF)
        return bytes(image)

    def test_frozen_model_and_quantized_operation_order(self):
        operations = expanded_operations()
        self.assertEqual(len(build_model_b().layers), 21)
        self.assertEqual(len(operations), 24)
        self.assertEqual([(op.name, op.output_channels) for op in operations],
                         known_operation_contract(build_model_b().model_id))
        self.assertEqual(sum(op.raw_weights for op in operations), 10256)
        self.assertEqual(sum(op.output_channels for op in operations), 563)
        self.assertEqual(sum(op.padded_channels for op in operations), 568)
        self.assertEqual([op.stride for op in operations if op.name in ("down1.dw", "down2.dw")], [2, 2])

    def test_exact_section_sizes_and_capacity(self):
        self.assertEqual(self.layout["total_bytes"], 17792)
        self.assertEqual(self.layout["oc_padded_weight_bytes"], 10296)
        sizes = {section["name"]: section["length"] for section in self.layout["sections"]}
        self.assertEqual(sizes, dict(operators=1536, weights=10816, bias=2432,
                                     multiplier=2432, relu=128, residual_reserved=192))
        small, large = capacity_report(256), capacity_report(512)
        self.assertFalse(small["fits"])
        self.assertEqual(small["capacity_bytes"], 10240)
        self.assertEqual(small["shortfall_bytes"], 7552)
        self.assertTrue(large["fits"])
        self.assertEqual(large["unused_bytes"], 2688)
        self.assertEqual(large["used_words_per_bank"], [445, 445, 445, 445, 444])
        with self.assertRaisesRegex(LayoutError, "capacity"):
            load_banks(self.image, depth=256, expected_sha256=self.sha)

    def test_ranges_alignment_and_no_overlap(self):
        sections = self.layout["sections"]
        previous = HEADER.size + len(sections) * SECTION_ENTRY.size
        for section in sections:
            self.assertEqual(section["offset"] % ALIGNMENT, 0)
            self.assertGreaterEqual(section["offset"], previous)
            previous = section["offset"] + section["length"]
            self.assertLessEqual(previous, len(self.image))
        section_map = {section["name"]: section for section in sections}
        for kind in ("weights", "bias", "multiplier", "relu"):
            previous = section_map[kind]["offset"]
            end = previous + section_map[kind]["length"]
            for record in self.layout["operators"]:
                field = record[kind]
                self.assertGreaterEqual(field["offset"], previous)
                self.assertLessEqual(field["offset"] + field["length"], end)
                if kind != "relu":
                    self.assertEqual(field["offset"] % 64, 0)
                previous = field["offset"] + field["length"]

    def test_bank_mapping_is_bijective_and_bounded(self):
        seen = set()
        for offset in range(5 * 512 * 8):
            bank, address, lane = bank_location(offset, 512)
            self.assertLess(bank, 5)
            self.assertLess(address, 512)
            self.assertLess(lane, 8)
            self.assertEqual((address * 5 + bank) * 8 + lane, offset)
            self.assertNotIn((bank, address, lane), seen)
            seen.add((bank, address, lane))
        self.assertEqual(bank_location(0, 512), (0, 0, 0))
        self.assertEqual(bank_location(39, 512), (4, 0, 7))
        self.assertEqual(bank_location(40, 512), (0, 1, 0))
        for invalid in (-1, 20480, True):
            with self.assertRaises(LayoutError):
                bank_location(invalid, 512)

    def test_independent_tensor_reconstruction_and_output_padding(self):
        # Deliberately do not call weight_positions here: reconstruct the wire
        # order from independent tensor coordinates and assert every original
        # value appears once, with every extra output lane equal to zero.
        for op, record in zip(expanded_operations(), self.layout["operators"]):
            actual = self.image[record["weights"]["offset"]:
                                record["weights"]["offset"] + record["weights"]["length"]]
            expected, recovered = [], {}
            for group in range(op.padded_channels // 8):
                input_groups = [0] if op.kind == "dw3x3" else range((op.input_channels + 7) // 8)
                for input_group in input_groups:
                    for ky in range(op.kernel):
                        for kx in range(op.kernel):
                            input_lanes = [0] if op.kind == "dw3x3" else range(min(8, op.input_channels - input_group * 8))
                            for input_lane in input_lanes:
                                term = (ky * op.kernel + kx) * (1 if op.kind == "dw3x3" else op.input_channels)
                                if op.kind != "dw3x3":
                                    term += input_group * 8 + input_lane
                                for lane in range(8):
                                    oc = group * 8 + lane
                                    if oc < op.output_channels:
                                        self.assertNotIn((oc, term), recovered)
                                        value = self.parameters[op.name].weights[oc][term]
                                        recovered[(oc, term)] = value
                                        expected.append(value & 255)
                                    else:
                                        expected.append(0)
            self.assertEqual(actual, bytes(expected), op.name)
            self.assertEqual(len(recovered), op.raw_weights)
        stem = self.layout["operators"][0]
        head = self.layout["operators"][-1]
        self.assertEqual(stem["weights"]["length"], 216)  # IC3 was not expanded to IC8.
        self.assertEqual(head["weights"]["length"], 64)  # OC3 was zero-padded to OC8.

    def test_full_image_and_loaded_bank_roundtrip(self):
        self.assertEqual(decode_image(self.image, expected_sha256=self.sha), self.parameters)
        banks = load_banks(self.image, depth=512, expected_sha256=self.sha)
        self.assertEqual(len(banks), BANKS)
        self.assertTrue(all(len(bank) == 4096 for bank in banks))
        loaded = read_banks(banks, len(self.image))
        self.assertEqual(loaded, self.image)
        self.assertEqual(decode_image(loaded, expected_sha256=self.sha), self.parameters)
        for offset in range(len(self.image), 20480):
            bank, address, lane = bank_location(offset, 512)
            self.assertEqual(banks[bank][address * 8 + lane], 0)

    def test_all_alignment_and_invalid_lane_padding_is_zero(self):
        for kind in ("weights", "bias", "multiplier", "relu"):
            section = next(item for item in self.layout["sections"] if item["name"] == kind)
            used = set()
            for op, record in zip(expanded_operations(), self.layout["operators"]):
                start = record[kind]["offset"]
                if kind == "weights":
                    used.update(start + i for i, (oc, _) in enumerate(weight_positions(op)) if oc < op.output_channels)
                elif kind in ("bias", "multiplier"):
                    used.update(range(start, start + op.output_channels * 4))
                else:
                    used.update(range(start, start + op.padded_channels // 8))
                    for group in range(op.padded_channels // 8):
                        valid_mask = (1 << min(8, op.output_channels - group * 8)) - 1
                        self.assertEqual(self.image[start + group] & ~valid_mask, 0)
            for offset in range(section["offset"], section["offset"] + section["length"]):
                if offset not in used:
                    self.assertEqual(self.image[offset], 0, f"nonzero {kind} padding {offset:#x}")

    def test_signed_bytes_i32_limits_and_q17_sign_extension(self):
        decoded = decode_image(self.image)
        values = [weight for parameters in decoded.values() for row in parameters.weights for weight in row]
        self.assertIn(-128, values)
        self.assertIn(127, values)
        scales = [value for parameters in decoded.values() for value in parameters.multiplier]
        self.assertIn(Q17_MIN, scales)
        self.assertIn(Q17_MAX, scales)
        first = self.layout["operators"][0]
        self.assertEqual(self.image[first["bias"]["offset"]:first["bias"]["offset"] + 8],
                         b"\x00\x00\x00\x80\xff\xff\xff\x7f")
        self.assertEqual(self.image[first["multiplier"]["offset"]:first["multiplier"]["offset"] + 4],
                         b"\x00\x00\xfe\xff")
        self.assertEqual(decoded["stem"].bias[:2], (I32_MIN, I32_MAX))

    def test_missing_reordered_or_invalid_parameters_rejected(self):
        broken = dict(self.parameters)
        broken.pop("rgb_head")
        with self.assertRaises(LayoutError):
            encode_parameters(broken)
        with self.assertRaises(LayoutError):
            encode_parameters(dict(reversed(list(self.parameters.items()))))
        for bad_scale in (Q17_MAX + 1, Q17_MIN - 1, True):
            broken = dict(self.parameters)
            broken["stem"] = replace(broken["stem"], multiplier=(bad_scale,) + broken["stem"].multiplier[1:])
            with self.assertRaises(LayoutError):
                encode_parameters(broken)
        for bad_bias in (I32_MAX + 1, I32_MIN - 1):
            broken = dict(self.parameters)
            broken["stem"] = replace(broken["stem"], bias=(bad_bias,) + broken["stem"].bias[1:])
            with self.assertRaises(LayoutError):
                encode_parameters(broken)

    def test_corrupt_headers_ranges_crc_sha_and_truncation_rejected(self):
        for offset in (0, 8, 12, 24, 40, HEADER.size + 4, self.layout["operators"][0]["weights"]["offset"]):
            damaged = bytearray(self.image)
            damaged[offset] ^= 1
            with self.assertRaises(LayoutError):
                decode_image(bytes(damaged))
        with self.assertRaisesRegex(LayoutError, "SHA"):
            decode_image(self.image, expected_sha256="0" * 64)
        with self.assertRaises(LayoutError):
            decode_image(self.image[:-1])

    def test_nonzero_padding_is_rejected_even_with_repaired_crc(self):
        stem, head = self.layout["operators"][0], self.layout["operators"][-1]
        residual = self.layout["sections"][-1]
        offsets = (stem["weights"]["offset"] + stem["weights"]["length"],
                   head["weights"]["offset"] + 3,
                   head["bias"]["offset"] + 3 * 4,
                   head["multiplier"]["offset"] + 3 * 4,
                   residual["offset"])
        for offset in offsets:
            damaged = bytearray(self.image)
            damaged[offset] = 1
            with self.assertRaises(LayoutError):
                decode_image(self.repair_checksums(damaged))
        damaged = bytearray(self.image)
        damaged[head["relu"]["offset"]] |= 0x80
        with self.assertRaises(LayoutError):
            decode_image(self.repair_checksums(damaged))

    def test_residual_scaling_and_synthetic_start_are_never_claimed_complete(self):
        residual = self.layout["sections"][-1]
        self.assertEqual(self.image[residual["offset"]:], bytes(192))
        self.assertEqual(self.layout["residual_scaling"]["status"], "UNDEFINED")
        self.assertFalse(self.layout["residual_scaling"]["valid"])
        self.assertFalse(self.layout["complete_network_start_allowed"])
        self.assertEqual(self.layout["quality_gate"], "UNMEASURED")
        with self.assertRaisesRegex(LayoutError, "residual scaling UNDEFINED"):
            decode_image(self.image, require_complete_network=True)
        self.assertEqual(HEADER.unpack_from(self.image)[8], IMAGE_FLAGS)

    def test_unused_bank_corruption_rejected(self):
        banks = list(load_banks(self.image, depth=512, expected_sha256=self.sha))
        damaged = bytearray(banks[4])
        damaged[-1] = 1
        banks[4] = bytes(damaged)
        with self.assertRaisesRegex(LayoutError, "unused bank capacity"):
            read_banks(tuple(banks), len(self.image))

    def test_artifacts_match_the_loader_and_are_deterministic(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            manifest = emit_artifacts(path)
            self.assertEqual(manifest["sha256"], self.sha)
            self.assertEqual((path / "coefficients.bin").read_bytes(), self.image)
            json_manifest = json.loads((path / "coefficient_layout.json").read_text(encoding="utf-8"))
            self.assertEqual(json_manifest["roundtrip"], "PASS")
            banks = []
            for bank in range(5):
                words = (path / f"bank{bank}.hex").read_text().splitlines()
                self.assertEqual(len(words), 512)
                banks.append(b"".join(int(word, 16).to_bytes(8, "little") for word in words))
            self.assertEqual(read_banks(tuple(banks), len(self.image)), self.image)


if __name__ == "__main__":
    unittest.main()
