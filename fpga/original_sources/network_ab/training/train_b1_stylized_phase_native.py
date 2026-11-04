"""Teach recoverable stylized RGB sampling phases using neighbouring features.

Three-by-three heads are training-only. Their reconstruction quality is NOT
a guarantee that the fixed separable B1 decoder can duplicate those heads.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_geometry_first_native import GeometryFirstTraining,evaluate
from .train_b1_fine_rgb_native import FineRGBNativeTraining
from .train_b1_balanced_phase_shape import feature_gradient_budget
from .train_b1_contour_coherence import contour_coherence
from .train_b1_late_region_supervision import main as driver
from . import train_b1_minimum_area as m


def centred_phases(rgb,scale):
    n,c,h,w=rgb.shape
    z=F.pixel_unshuffle(rgb,scale).reshape(n,c,scale*scale,h//scale,w//scale)
    return (z-z.mean(2,keepdim=True)).flatten(1,2)


class StylizedPhaseTraining(GeometryFirstTraining):
    phase_head_specs={'phase2':(16,12,3),'phase4':(32,48,3)}

    def batch_style(self,native):
        x,t,specs,flow,valid=super().batch_style(native)
        self.last_style_target=t.detach()
        return x,t,specs,flow,valid

    def additional_loss(self,p,t,native):
        loss,values=FineRGBNativeTraining.additional_loss(self,p,t,native)
        if native:
            yy=self.rng.randrange(0,p.shape[-2]-384+1);xx=self.rng.randrange(0,p.shape[-1]-384+1)
            curve=contour_coherence(p[...,yy:yy+384,xx:xx+384],t[...,yy:yy+384,xx:xx+384])
        else:curve=contour_coherence(p,t)
        reference=loss+self.contour_weight*curve;total=reference;values['short_contour_turn_excess']=curve
        for scale,node,key in ((2,'down1.pw','phase2'),(4,'down2.pw','phase4')):
            feature=self.current_nodes[node];target=centred_phases(self.last_style_target,scale)
            q=self.heads[key](feature/64-1);n,_,h,w=q.shape
            q=q.reshape(n,3,scale*scale,h,w);q=(q-q.mean(2,keepdim=True)).flatten(1,2)
            margin=40//scale;q=q[...,margin:-margin,margin:-margin];target=target[...,margin:-margin,margin:-margin]
            part=F.smooth_l1_loss(q,target,beta=2/128)
            coefficient,ratio=feature_gradient_budget(reference,part,feature)
            total=total+coefficient*part
            values.update({f'phase{scale}_feature_coefficient':coefficient,f'phase{scale}_actual_feature_gradient_ratio':ratio,
                f'stylized_feature_phase{scale}_mae_byte':(q-target).abs().mean()*128})
        return total,values

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(source_phase_auxiliary_heads=False,stylized_phase_auxiliary_heads=True,
            phase_auxiliary_targets='Within-cell RGB residual of actual stylized target; neighbour3x3 readout at down1.pw/down2.pw',
            phase_head_specs=self.phase_head_specs,phase_heads_exported=False,
            phase_auxiliary_limitation='Training heads are not runtime decoder; readout quality does not prove B1 render capacity',visual_acceptance=False)
        torch.save(s,path)


def regression():
    x=torch.arange(3*32*48).float().reshape(1,3,32,48)/1000
    for scale in (2,4):
        q=centred_phases(x,scale).reshape(1,3,scale*scale,32//scale,48//scale)
        assert q.mean(2).abs().max()<1e-6
        assert torch.allclose(centred_phases(x+.25,scale),q.flatten(1,2),atol=1e-6)
        assert centred_phases(torch.ones_like(x),scale).abs().max()==0
    print('STYLIZED_PHASE_PASS shapes/channel_means/constant_colour_invariance',flush=True)


if __name__=='__main__':
    torch.set_num_threads(2);regression()
    driver(training_factory=StylizedPhaseTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=m.t.preview)
