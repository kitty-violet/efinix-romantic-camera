import unittest
import torch
from network_ab.training.coherent_texture_train import gray,centered_gram,edge_floor,flat_residual


class TextureObjectiveTests(unittest.TestCase):
    def test_luminance_and_centered_statistics_do_not_reward_constant_color(self):
        torch.manual_seed(17)
        x=torch.randn(1,5,16,20)
        offsets=torch.arange(5).reshape(1,5,1,1).float()
        self.assertTrue(torch.allclose(centered_gram(x),centered_gram(x+offsets),atol=1e-6))
        self.assertTrue(torch.allclose(gray(torch.ones(1,3,10,12)*.4),torch.ones(1,1,10,12)*.4))

    def test_edge_floor_penalizes_erasure_not_uniform_tone_offset(self):
        x=torch.zeros(1,3,20,20);x[:,:,:,10:]=.5
        self.assertEqual(float(edge_floor(x,x+.1)),0.)
        self.assertGreater(float(edge_floor(x,torch.zeros_like(x))),.1)

    def test_flat_region_penalty_rejects_sparkles_but_allows_uniform_tone(self):
        x=torch.zeros(1,3,20,20)
        self.assertLess(float(flat_residual(x,x+.2)),1e-6)
        y=x.clone();y[:,:,10,10]=.6;y.requires_grad_()
        loss=flat_residual(x,y);loss.backward()
        self.assertGreater(float(loss),0.)
        self.assertTrue(torch.isfinite(y.grad).all())


if __name__=='__main__':unittest.main()
