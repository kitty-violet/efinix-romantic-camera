import unittest
import torch
from network_ab.training.coherent_guarded_train import chroma,gradient_order


class GuardedObjectiveTests(unittest.TestCase):
    def test_chroma_anchor_allows_luminance_change(self):
        x=torch.rand(1,3,16,20)*.6-.3
        self.assertTrue(torch.allclose(chroma(x),chroma(x+.2),atol=1e-6))

    def test_edge_order_detects_halo_reversal_without_locking_contrast(self):
        x=torch.zeros(1,3,20,20);x[:,:,:,10:]=.5
        self.assertEqual(float(gradient_order(x,x*.6+.1)),0.)
        y=(-x).detach().requires_grad_();loss=gradient_order(x,y)
        self.assertGreater(float(loss.detach()),.1)
        loss.backward();self.assertTrue(torch.isfinite(y.grad).all())


if __name__=='__main__':unittest.main()
