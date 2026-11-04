"""CPU arithmetic and scheduling tests; no VGG download or content training."""
import json
import unittest

from network_ab.training.runtime import HERE, configure_workspace
configure_workspace()

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Run in the approved training environment")
class TorchComponentsTests(unittest.TestCase):
    def test_all_256_rgb_values_roundtrip(self):
        from network_ab.training.losses import rgb01
        source = torch.arange(256, dtype=torch.float32)
        value = (source-128)/128
        torch.testing.assert_close(rgb01(value)*255, source, atol=2e-5, rtol=0)
        self.assertNotEqual(rgb01(torch.tensor(0.)).item(), 0.5)

    def test_gram_normalization_and_gradient(self):
        from network_ab.training.losses import gram
        value = torch.tensor([[[[1., 2.]], [[3., 4.]]]], requires_grad=True)
        torch.testing.assert_close(gram(value), torch.tensor([[[1.25, 2.75], [2.75, 6.25]]]))
        gram(value).sum().backward()
        torch.testing.assert_close(value.grad, torch.tensor([[[[2., 3.]], [[2., 3.]]]]))

    def test_color_moment_loss_tracks_mean_and_variance(self):
        from network_ab.training.losses import color_moment_loss, color_moments
        image = torch.tensor([[[[-1., 1.]], [[0., 0.]], [[1., -1.]]]], requires_grad=True)
        target = color_moments(image.detach()).clone()
        self.assertAlmostEqual(color_moment_loss(image, target).item(), 0.0)
        target[:, :3] -= 0.1
        loss = color_moment_loss(image, target)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(image.grad.abs().sum().item(), 0)

    def test_frozen_schedule_boundaries(self):
        from network_ab.training.engine import learning_rate_factor
        config = json.loads((HERE / "fp32_config.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(learning_rate_factor(0, config), 0.001)
        self.assertAlmostEqual(learning_rate_factor(999, config), 1.)
        self.assertAlmostEqual(learning_rate_factor(1000, config), 1.)
        self.assertAlmostEqual(learning_rate_factor(99999, config), 0.01)
        self.assertGreater(learning_rate_factor(50000, config), 0.01)

    def test_official_vgg_feature_indices(self):
        from torchvision.models.vgg import cfgs, make_layers
        from network_ab.training.losses import FEATURES
        feature_stack = make_layers(cfgs["E"], batch_norm=False)
        channels = {"relu1_2": 64, "relu2_2": 128, "relu3_3": 256, "relu4_3": 512}
        for name, index in FEATURES.items():
            self.assertIsInstance(feature_stack[index], torch.nn.ReLU)
            self.assertIsInstance(feature_stack[index-1], torch.nn.Conv2d)
            self.assertEqual(feature_stack[index-1].out_channels, channels[name])

    def test_rgb_initialization_preserves_signed_input_sampling(self):
        from network_ab.contract_torch import build_fp32_model
        from network_ab.training.optimization_policy import initialize_model
        model = build_fp32_model()
        before = [(n, type(m)) for n, m in model.named_modules()]
        config = {"initialization": "rgb_path_v1", "bn_policy": "frozen_running_stats_train_affine"}
        initialize_model(model, config)
        x = torch.linspace(-1, 127/128, 3*16*32).reshape(1, 3, 16, 32)
        y = model(x)
        expected = x[:, :, ::4, ::4].repeat_interleave(4, -2).repeat_interleave(4, -1)
        torch.testing.assert_close(y, expected, atol=1e-6, rtol=0)
        self.assertEqual(before, [(n, type(m)) for n, m in model.named_modules()])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 11379)

    def test_frozen_bn_keeps_affine_gradients_and_deployment_result(self):
        from network_ab.contract_torch import build_fp32_model
        from network_ab.training.optimization_policy import initialize_model, set_optimization_mode
        model = build_fp32_model()
        config = {"initialization": "rgb_path_v1", "bn_policy": "frozen_running_stats_train_affine"}
        initialize_model(model, config)
        buffers = {n:v.clone() for n,v in model.named_buffers()}
        x = torch.randn(1,3,16,32).clamp(-0.8,0.8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(2):
            set_optimization_mode(model, config)
            optimizer.zero_grad(set_to_none=True)
            y = model(x)
            (y-x).square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
        self.assertTrue(all(torch.equal(v, dict(model.named_buffers())[n]) for n,v in buffers.items()))
        a = model(x)
        model.eval()
        torch.testing.assert_close(a, model(x), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
