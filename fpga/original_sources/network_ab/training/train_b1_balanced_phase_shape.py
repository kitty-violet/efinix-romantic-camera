"""Give sampling-phase preservation a measured feature-gradient budget.

Both heads and adaptive coefficients exist in training only. Final RGB keeps
its stylized target. Source phase recovery is not an information-theory bound.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_shape_curriculum_native import ShapeCurriculumTraining
from .train_b1_fine_rgb_native import FineRGBNativeTraining,evaluate
from .train_b1_contour_coherence import contour_coherence
from .train_b1_late_region_supervision import main as driver
from . import train_b1_minimum_area as m


def feature_gradient_budget(reference,auxiliary,feature,fraction=.25):
    main=torch.autograd.grad(reference,feature,retain_graph=True)[0].detach().norm()
    side=torch.autograd.grad(auxiliary,feature,retain_graph=True)[0].detach().norm()
    coefficient=(fraction*main/side.clamp_min(1e-12)).clamp_max(256.)
    return coefficient,coefficient*side/main.clamp_min(1e-12)


class BalancedPhaseShapeTraining(ShapeCurriculumTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        warm=torch.load(args.warm,map_location='cpu',weights_only=True)
        if 'synthetic_shape_curriculum' in warm:
            geometry=warm['synthetic_shape_curriculum']
            self.shape_colours.copy_(geometry['colours']);self.shape_anchors.copy_(geometry['immutable_warm_anchors'])

    def additional_loss(self,p,t,native):
        loss,values=FineRGBNativeTraining.additional_loss(self,p,t,native)
        if native:
            yy=self.rng.randrange(0,p.shape[-2]-384+1);xx=self.rng.randrange(0,p.shape[-1]-384+1)
            curve=contour_coherence(p[...,yy:yy+384,xx:xx+384],t[...,yy:yy+384,xx:xx+384])
        else:curve=contour_coherence(p,t)
        reference=loss+getattr(self,'contour_weight',4)*curve;total=reference
        y=(self.last_source[:,:1]+2*self.last_source[:,1:2]+self.last_source[:,2:3])/4
        values['short_contour_turn_excess']=curve
        for scale,node,key in ((2,'down1.pw','phase2'),(4,'down2.pw','phase4')):
            target=F.pixel_unshuffle(y,scale);target=target-target.mean(1,keepdim=True)
            feature=self.current_nodes[node]
            q=self.heads[key](feature/64-1);q=q-q.mean(1,keepdim=True);margin=40//scale
            q=q[...,margin:-margin,margin:-margin];target=target[...,margin:-margin,margin:-margin]
            part=F.smooth_l1_loss(q,target,beta=2/128)
            coefficient,ratio=feature_gradient_budget(reference,part,feature)
            total=total+coefficient*part
            values.update({f'phase{scale}_feature_coefficient':coefficient,f'phase{scale}_actual_feature_gradient_ratio':ratio,
                f'feature_phase{scale}_mae_byte':(q-target).abs().mean()*128})
        return total,values

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.pop('phase_auxiliary_weight',None)
        state.update(phase_feature_gradient_budget=dict(per_head_fraction=.25,coefficient_cap=256,
            reference='actual stylized RGB geometry + contour coherence at the same feature tensor',
            meaning='norm before shared parameter Jacobians/optimizer, not final parameter-step ratio'),visual_acceptance=False)
        torch.save(state,path)

    def preflight_extras(self):
        result=super().preflight_extras();values=result['update']['values']
        for scale in (2,4):
            ratio=values[f'phase{scale}_actual_feature_gradient_ratio']
            assert 0<ratio<=.250001
        result['phase_feature_gradient_budget_checked']=True
        return result


def regression():
    torch.manual_seed(77);z=torch.randn(3,4,5,6,requires_grad=True)
    main=(z-.5).square().mean();aux=.001*(z+.2).square().mean()
    scale,ratio=feature_gradient_budget(main,aux,z)
    assert scale<=256 and .0<ratio<=.250001 and not scale.requires_grad
    a=torch.autograd.grad(main,z,retain_graph=True)[0];b=torch.autograd.grad(scale*aux,z,retain_graph=True)[0]
    assert abs(float(b.norm()/a.norm()-ratio))<1e-6
    scale,ratio=feature_gradient_budget(z.sum()*0,aux,z);assert scale==0 and ratio==0
    print('FEATURE_GRADIENT_BUDGET_PASS actual_ratio/detached/cap/zero_reference',flush=True)


if __name__=='__main__':
    import sys
    torch.set_num_threads(2);regression()
    if '--check' not in sys.argv:driver(training_factory=BalancedPhaseShapeTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=m.t.preview)
