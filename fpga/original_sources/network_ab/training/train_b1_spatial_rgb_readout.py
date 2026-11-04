"""Supervise actual RGB boundaries and actual rendered Y after spare features.

An unchanged node0 Y loss cannot train corrections entering at RGB-head
columns3:8. Add actual RGB-Y and shared-location RGB profiles explicitly.
The latent known-Y reference remains frozen and is not a moving target.
"""
import torch
from torch.nn import functional as F
from .train_b1_relaxed_count_interfaces import RelaxedCountInterfaceTraining
from .train_b1_reference_boundary_profile import reference_boundary_profile


class SpatialRGBReadoutTraining(RelaxedCountInterfaceTraining):
    def additional_loss(self,p,t,native):
        loss,v=super().additional_loss(p,t,native)
        y=lambda z:(z[:,:1]+2*z[:,1:2]+z[:,2:3])/4
        pixel=F.smooth_l1_loss(y(p),y(t),beta=2/128)
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        profile,detail=reference_boundary_profile(p,t,labels)
        v.update(actual_rgb_luma_pixel=pixel,actual_rgb_boundary_profile=profile)
        return loss+8*pixel+4*profile,v
