import unittest
import torch
from torch.nn import functional as F
from network_ab.training.coherent_material_loss import texture_view,nearest_cosine_loss
from network_ab.training.coherent_material_match import reference_targets,fixed_material_loss


class MaterialTests(unittest.TestCase):
    def test_changing_output_cannot_switch_to_an_easier_target(self):
        ref=torch.zeros(1,2,8,8);ref[:,0]=1
        changed=torch.zeros_like(ref);changed[:,1]=1;changed.requires_grad_()
        bank={k:torch.eye(2) for k in ('relu2_2','relu3_3')}
        targets=reference_targets({k:ref for k in bank},bank)
        loss=fixed_material_loss({k:changed for k in bank},targets)
        self.assertAlmostEqual(float(loss.detach()),1.)
        loss.backward();self.assertGreater(float(changed.grad.norm()),0.)

    def test_gray_affine_changes_cannot_improve_texture_view(self):
        torch.manual_seed(42);x=torch.rand(2,3,32,48)*.8-.4
        self.assertLess(float((texture_view(x)-texture_view(x*.7+.15)).abs().max()),2e-6)

    def test_reference_features_affect_loss_and_gradient(self):
        q=torch.tensor([[.8,.6]],requires_grad=True)
        a=torch.tensor([[1.,0.]]);b=torch.tensor([[0.,1.]])
        la=nearest_cosine_loss(q,a);ga,=torch.autograd.grad(la,q,retain_graph=True)
        lb=nearest_cosine_loss(q,b);gb,=torch.autograd.grad(lb,q)
        self.assertGreater(float((ga-gb).norm()),1)
        self.assertGreater(float((la-lb).abs()),.1)

    def test_chunked_nearest_matches_full_and_is_finite(self):
        torch.manual_seed(44);q=F.normalize(torch.rand(27,8),dim=1).requires_grad_(True)
        bank=F.normalize(torch.rand(19,8),dim=1)
        actual=nearest_cosine_loss(q,bank,chunk=4);expected=(1-(q@bank.T).max(1).values).mean()
        self.assertLess(float((actual-expected).abs()),1e-6)
        actual.backward();self.assertTrue(torch.isfinite(q.grad).all())


if __name__=='__main__':unittest.main()
