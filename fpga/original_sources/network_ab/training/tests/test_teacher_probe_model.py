"""Small shape and gradient checks for the non-deployable teacher."""
import unittest

import torch

from network_ab.training.teacher_probe_model import TrainingOnlyTeacher


class TeacherProbeModelTests(unittest.TestCase):
    def test_shape_and_backpropagation(self):
        model = TrainingOnlyTeacher()
        image = torch.rand(1, 3, 64, 96, requires_grad=True) * 1.5 - 0.75
        result = model(image)
        self.assertEqual(tuple(result.shape), tuple(image.shape))
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreaterEqual(result.min().item(), -1.0)
        self.assertLessEqual(result.max().item(), 127 / 128)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(model.head[1].weight.grad).all())
        self.assertGreater(model.head[1].weight.grad.abs().sum().item(), 0)

    def test_dimensions_are_checked(self):
        with self.assertRaisesRegex(ValueError, "divisible by four"):
            TrainingOnlyTeacher()(torch.zeros(1, 3, 65, 64))


if __name__ == "__main__":
    unittest.main()
