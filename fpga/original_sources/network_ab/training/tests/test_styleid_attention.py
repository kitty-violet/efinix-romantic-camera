"""Numerical comparison against the downloaded, pinned author's attention."""
import importlib.util
from pathlib import Path
import unittest

import torch
from diffusers.models.attention_processor import Attention
from network_ab.training.styleid_attention import InjectionState, StyleIDProcessor


class StyleIDTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[3] / "model-training-workspace/runs/styleid-material-probe-20261002/upstream/diffusers_implementation_stable_diffusion.py"
        spec = importlib.util.spec_from_file_location("styleid_author_reference", path)
        cls.upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.upstream)

    def setUp(self):
        torch.manual_seed(7)
        self.attn = Attention(query_dim=320, heads=8, dim_head=40).cuda().eval()
        self.x = torch.randn(1, 77, 320, device="cuda")
        self.state = InjectionState()
        self.processor = StyleIDProcessor(self.state, "layer7_attn")

    def test_unmodified_matches_author(self):
        with torch.no_grad():
            expected = self.upstream.attention_op(self.attn, self.x)[-1]
            actual = self.processor(self.attn, self.x)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_content_query_style_kv_matches_author_and_changes_output(self):
        with torch.no_grad():
            self.state.mode = "content"
            self.processor(self.attn, self.x * 1.1)
            self.state.mode = "style"
            style = torch.randn(1, 91, 320, device="cuda")
            self.processor(self.attn, style)
            self.state.mode = "inject"
            actual = self.processor(self.attn, self.x)
            qc = self.upstream.attention_op(self.attn, self.x * 1.1)[1]
            _, _, ks, vs, _ = self.upstream.attention_op(self.attn, style)
            qs = self.upstream.attention_op(self.attn, self.x)[1]
            expected = self.upstream.attention_op(self.attn, self.x,
                query=.75 * qc + .25 * qs, key=ks, value=vs, temperature=1.5)[-1]
            original = self.upstream.attention_op(self.attn, self.x)[-1]
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        self.assertGreater(float((actual-original).abs().mean()), 1e-3)
        self.assertTrue(all(t.device.type == "cpu" for bank in (self.state.content, self.state.style) for ts in bank.values() for t in ts))


if __name__ == "__main__":
    unittest.main()
