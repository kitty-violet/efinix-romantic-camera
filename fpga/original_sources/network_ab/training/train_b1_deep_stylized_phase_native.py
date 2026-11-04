"""Supervise stylized RGB subpixel geometry after B1's residual context.

Unlike early down1/down2 feature supervision, both selected tensors have
passed all three residual blocks. Heads are discarded before byte export.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_stylized_phase_native import StylizedPhaseTraining,centred_phases,regression
from .train_b1_geometry_first_native import evaluate
from .train_b1_fine_rgb_native import FineRGBNativeTraining
from .train_b1_balanced_phase_shape import feature_gradient_budget
from .train_b1_contour_coherence import contour_coherence
from .train_b1_late_region_supervision import main as driver
from . import train_b1_minimum_area as m


class DeepStylizedPhaseTraining(StylizedPhaseTraining):
    def __init__(self,args,data):
        warm=torch.load(args.warm,map_location='cpu',weights_only=True)
        if warm.get('stylized_phase_auxiliary_heads'):
            assert warm.get('deep_stylized_phase_nodes')=={'phase2':'up1_high_dw','phase4':'res3_add'},'Different head attachment requires explicit reinitialization'
        super().__init__(args,data)

    def additional_loss(self,p,t,native):
        loss,values=FineRGBNativeTraining.additional_loss(self,p,t,native)
        if native:
            yy=self.rng.randrange(0,p.shape[-2]-384+1);xx=self.rng.randrange(0,p.shape[-1]-384+1)
            curve=contour_coherence(p[...,yy:yy+384,xx:xx+384],t[...,yy:yy+384,xx:xx+384])
        else:curve=contour_coherence(p,t)
        reference=loss+self.contour_weight*curve;total=reference;values['short_contour_turn_excess']=curve
        for scale,node,key in ((2,'up1_high_dw','phase2'),(4,'res3_add','phase4')):
            feature=self.current_nodes[node];target=centred_phases(self.last_style_target,scale)
            q=self.heads[key](feature/64-1);n,_,h,w=q.shape
            q=q.reshape(n,3,scale*scale,h,w);q=(q-q.mean(2,keepdim=True)).flatten(1,2)
            margin=40//scale;q=q[...,margin:-margin,margin:-margin];target=target[...,margin:-margin,margin:-margin]
            assert q.shape==target.shape,(node,q.shape,target.shape)
            part=F.smooth_l1_loss(q,target,beta=2/128)
            coefficient,ratio=feature_gradient_budget(reference,part,feature)
            total=total+coefficient*part
            values.update({f'phase{scale}_feature_coefficient':coefficient,f'phase{scale}_actual_feature_gradient_ratio':ratio,
                f'deep_stylized_phase{scale}_mae_byte':(q-target).abs().mean()*128})
        return total,values

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(phase_auxiliary_targets='Within-cell RGB residual of stylized target at existing deep180/360 features',
            deep_stylized_phase_nodes={'phase2':'up1_high_dw','phase4':'res3_add'},auxiliary_heads_exported=False,visual_acceptance=False)
        torch.save(s,path)


if __name__=='__main__':
    torch.set_num_threads(2);regression()
    driver(training_factory=DeepStylizedPhaseTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=m.t.preview)
