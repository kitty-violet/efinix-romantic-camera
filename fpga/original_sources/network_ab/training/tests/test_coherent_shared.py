import unittest
import torch
from torch import nn

from network_ab.training.coherent_shared_model import (
    SharedAppearanceNet, temporal_residual_loss, warp,
)


class SharedCoherenceTests(unittest.TestCase):
    def test_identity_initialization_preserves_pixels_and_shape(self):
        model=SharedAppearanceNet(width=4,blocks=1)
        image=torch.rand(1,3,41,63)*1.8-.9
        with torch.no_grad():result=model(image)
        self.assertEqual(result.shape,image.shape)
        self.assertTrue(torch.equal(result,image))
        self.assertFalse(any(isinstance(m,(nn.BatchNorm2d,nn.InstanceNorm2d)) for m in model.modules()))

    def test_backward_flow_tracks_translated_appearance(self):
        current=torch.rand(1,3,32,48)*.5
        delta=torch.rand_like(current)*.1
        flow=torch.zeros(1,2,32,48);flow[:,0]=-1
        next_frame=warp(current,flow)+.03  # real exposure change is allowed
        next_delta=warp(delta,flow)
        mask=torch.ones(1,1,32,48);mask[:,:,:,0]=0
        loss=temporal_residual_loss(current,next_frame,current+delta,
                                    next_frame+next_delta,flow,mask)
        self.assertLess(float(loss),1e-7)
        self.assertGreater(float(temporal_residual_loss(current,next_frame,current+delta,
                            next_frame+next_delta+.1,flow,mask)),.09)

    def test_occluded_region_does_not_force_a_false_correspondence(self):
        a=torch.zeros(1,3,16,24);b=a.clone();out=b.clone()
        out[:,:,:,12:]=1
        mask=torch.ones(1,1,16,24);mask[:,:,:,12:]=0
        loss=temporal_residual_loss(a,b,a,out,torch.zeros(1,2,16,24),mask)
        self.assertEqual(float(loss),0.)


if __name__=='__main__':unittest.main()
