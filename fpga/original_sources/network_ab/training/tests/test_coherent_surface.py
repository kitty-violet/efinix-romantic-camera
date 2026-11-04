import unittest
import torch
from network_ab.training.coherent_shared_model import SharedAppearanceNet
from network_ab.training.coherent_surface_model import SurfaceAppearanceNet
from network_ab.training.coherent_surface_prior import bilateral_surface,apply_tone_prior


class SurfaceTests(unittest.TestCase):
    def test_previous_weights_are_exact_at_initialization(self):
        torch.manual_seed(27);old=SharedAppearanceNet(8,1)
        torch.nn.init.normal_(old.head[1].weight,std=.02)
        new=SurfaceAppearanceNet(8,1);new.load_previous(old.state_dict())
        x=torch.rand(1,3,32,48)*1.8-.9
        self.assertEqual(float((old(x)-new(x)).abs().max()),0.)

    def test_correction_can_darkenen_white_beyond_previous_limit(self):
        net=SurfaceAppearanceNet(8,1)
        with torch.no_grad():net.correction[1].bias.fill_(-1.2)
        x=torch.full((1,3,32,32),127/128)
        out=net(x);self.assertLess(float(out.max()*128+128),139.8)
        out.mean().backward();self.assertTrue(torch.isfinite(net.correction[1].bias.grad).all())

    def test_surface_suppresses_small_noise_but_preserves_hard_boundary(self):
        torch.manual_seed(29);x=torch.full((1,3,48,64),.2);x[:,:,:,32:]=.8
        noisy=x+(torch.rand_like(x)-.5)*.04
        filtered=bilateral_surface(noisy,noisy)
        self.assertLess(float((filtered-x)[:,:,:,5:27].std()),float((noisy-x)[:,:,:,5:27].std())*.6)
        self.assertGreater(float(filtered[:,:,:,33].mean()-filtered[:,:,:,30].mean()),.57)
        self.assertGreaterEqual(float(filtered.min()),float(noisy.min())-1e-6)
        self.assertLessEqual(float(filtered.max()),float(noisy.max())+1e-6)

    def test_surface_shift_is_consistent_away_from_boundary(self):
        torch.manual_seed(31);x=torch.rand(1,3,32,48)
        a=bilateral_surface(x,x);y=torch.roll(x,1,3);b=bilateral_surface(y,y)
        self.assertLess(float((torch.roll(a,1,3)-b)[:,:,8:-8,8:-8].abs().max()),1e-6)

    def test_shared_neutral_tone_mapping_is_monotone(self):
        prior={'luma_knots':[0,.15,.3,.4,.5],'chroma_matrix':[[0,0,0],[0,0,0],[0,0,0]]}
        x=torch.linspace(0,1,100)[None,None,None,:].repeat(1,3,4,1)
        out=apply_tone_prior(x,prior)
        self.assertTrue(torch.all(out[:,:,:,1:]>=out[:,:,:,:-1]))
        self.assertAlmostEqual(float(out.max()),.5,places=6)


if __name__=='__main__':unittest.main()
