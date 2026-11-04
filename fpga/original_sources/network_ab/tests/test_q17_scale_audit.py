import sys
import unittest
from decimal import Decimal
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q17_scale_audit import (  # noqa: E402
    AuditInputError,
    Q17_MAX,
    audit_document,
    known_operation_contract,
    quantize_ideal,
    quantize_q17,
)


class Q17ScaleAuditTests(unittest.TestCase):
    def test_exact_representable_scales_and_samples(self) -> None:
        result = audit_document(
            {
                "schema": "q17-scale-audit-v1",
                "model_id": "synthetic-exact",
                "layers": [
                    {
                        "name": "stem",
                        "input_scale": "0.0078125",
                        "input_zero_point": 0,
                        "weight_scales": ["0.25", "0.5"],
                        "weight_zero_points": [0, 0],
                        "output_scale": "0.03125",
                        "output_zero_point": 0,
                        "relu": False,
                        "accumulator_samples": [
                            [-1024, -3, -2, -1, 0, 1, 2, 3, 1024],
                            [-512, -3, -2, -1, 0, 1, 2, 3, 512],
                        ],
                    }
                ],
            }
        )
        self.assertTrue(result["hardware_range_pass"])
        self.assertTrue(result["sample_equivalence_pass"])
        self.assertEqual(18, result["samples_evaluated"])
        self.assertEqual("UNMEASURED", result["quality_gate"])

    def test_scale_that_rounds_to_positive_one_is_rejected(self) -> None:
        result = audit_document(
            {
                "schema": "q17-scale-audit-v1",
                "model_id": "synthetic-unfit",
                "layers": [
                    {
                        "name": "head",
                        "input_scale": "1",
                        "input_zero_point": 0,
                        "weight_scales": ["1"],
                        "weight_zero_points": [0],
                        "output_scale": "1",
                        "output_zero_point": 0,
                    }
                ],
            }
        )
        channel = result["layers"][0]["channels"][0]
        self.assertEqual(Q17_MAX + 1, channel["q17_raw_integer"])
        self.assertFalse(channel["q17_fits_signed_i18"])
        self.assertFalse(result["hardware_range_pass"])

    def test_half_away_from_zero_matches_rtl_contract(self) -> None:
        self.assertEqual(1, quantize_q17(1, 1 << 16, False))
        self.assertEqual(-1, quantize_q17(-1, 1 << 16, False))
        self.assertEqual(1, quantize_ideal(1, Decimal("0.5"), False))
        self.assertEqual(-1, quantize_ideal(-1, Decimal("0.5"), False))

    def test_missing_scales_are_not_silently_accepted(self) -> None:
        with self.assertRaises(AuditInputError):
            audit_document(
                {
                    "schema": "q17-scale-audit-v1",
                    "model_id": "invalid",
                    "layers": [{"name": "stem", "weight_scales": ["0.1"]}],
                }
            )

    def test_nonzero_zero_point_is_rejected(self) -> None:
        with self.assertRaises(AuditInputError):
            audit_document(
                {
                    "schema": "q17-scale-audit-v1",
                    "model_id": "invalid-asymmetric",
                    "layers": [
                        {
                            "name": "stem",
                            "input_scale": "0.01",
                            "input_zero_point": 1,
                            "weight_scales": ["0.01"],
                            "weight_zero_points": [0],
                            "output_scale": "0.01",
                            "output_zero_point": 0,
                        }
                    ],
                }
            )

    def test_known_model_rejects_incomplete_layer_manifest(self) -> None:
        with self.assertRaises(AuditInputError):
            audit_document(
                {
                    "schema": "q17-scale-audit-v1",
                    "model_id": "B_BALANCED_DECODER_8_16_32_R3",
                    "layers": [
                        {
                            "name": "stem",
                            "input_scale": "0.01",
                            "input_zero_point": 0,
                            "weight_scales": ["0.01"] * 8,
                            "weight_zero_points": [0] * 8,
                            "output_scale": "0.01",
                            "output_zero_point": 0,
                        }
                    ],
                }
            )

    def test_complete_model_b_manifest_contract(self) -> None:
        model_id = "B_BALANCED_DECODER_8_16_32_R3"
        contract = known_operation_contract(model_id)
        self.assertIsNotNone(contract)
        layers = []
        for name, channels in contract or []:
            layers.append(
                {
                    "name": name,
                    "input_scale": "0.01",
                    "input_zero_point": 0,
                    "weight_scales": ["0.01"] * channels,
                    "weight_zero_points": [0] * channels,
                    "output_scale": "0.02",
                    "output_zero_point": 0,
                }
            )
        result = audit_document(
            {"schema": "q17-scale-audit-v1", "model_id": model_id, "layers": layers}
        )
        self.assertEqual(24, result["layer_count"])
        self.assertTrue(result["hardware_range_pass"])
        self.assertIsNone(result["sample_equivalence_pass"])


if __name__ == "__main__":
    unittest.main()
