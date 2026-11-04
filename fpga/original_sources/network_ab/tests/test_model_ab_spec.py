import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_ab_spec import (  # noqa: E402
    EXPECTED,
    TensorShape,
    build_model_a,
    build_model_b,
    validate_model,
)


class ModelABSpecTests(unittest.TestCase):
    def test_model_a_exact_budget_and_shape(self) -> None:
        model = build_model_a()
        validate_model("A", model)
        self.assertEqual(EXPECTED["A"]["macs_per_frame"], model.macs_per_frame)
        self.assertEqual(EXPECTED["A"]["weight_count"], model.weight_count)
        self.assertEqual(TensorShape(1280, 720, 3), model.layers[-1].output)

    def test_model_b_exact_budget_and_shape(self) -> None:
        model = build_model_b()
        validate_model("B", model)
        self.assertEqual(EXPECTED["B"]["macs_per_frame"], model.macs_per_frame)
        self.assertEqual(EXPECTED["B"]["weight_count"], model.weight_count)
        self.assertEqual(TensorShape(1280, 720, 3), model.layers[-1].output)

    def test_model_b_reduces_macs_but_keeps_three_residual_blocks(self) -> None:
        model_a = build_model_a()
        model_b = build_model_b()
        self.assertLess(model_b.macs_per_frame, model_a.macs_per_frame)
        for model in (model_a, model_b):
            adds = [layer for layer in model.layers if layer.operator == "residual_add"]
            self.assertEqual(3, len(adds))

    def test_only_decoder_and_head_differ(self) -> None:
        model_a = build_model_a()
        model_b = build_model_b()
        common_a = [layer for layer in model_a.layers if not layer.name.startswith("up") and layer.name != "rgb_head"]
        common_b = [layer for layer in model_b.layers if not layer.name.startswith("up") and layer.name != "rgb_head"]
        self.assertEqual(common_a, common_b)


if __name__ == "__main__":
    unittest.main()
