"""Penalize short-range contour turns, retaining teacher corners and contrast.

RGB structure-tensor coherence is a TRAIN-ONLY geometric constraint, inspired
by vector-valued structure tensors (Weickert1999). This is not a reproduction
of his diffusion solver. Nothing is added to deployed B1 or its input/output.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_projected_count_regions import ProjectedCountTraining,evaluate as base_evaluate,preview
from .train_b1_joint_rgb_edges import rgb_edge_vector_loss
from . import train_b1_late_region_supervision as driver
from . import train_b1_minimum_area as m


def orientation_defect(rgb):
    channels=rgb.shape[1]
    kernel=rgb.new_tensor([[1,2,1],[2,4,2],[1,2,1]])[None,None]/16
    smooth=F.conv2d(F.pad(rgb,(1,1,1,1),mode='replicate'),kernel.expand(channels,1,3,3),groups=channels)
    gx=(F.pad(smooth,(1,1,0,0),mode='replicate')[...,2:]-F.pad(smooth,(1,1,0,0),mode='replicate')[...,:-2])/2
    gy=(F.pad(smooth,(0,0,1,1),mode='replicate')[...,2:,:]-F.pad(smooth,(0,0,1,1),mode='replicate')[...,:-2,:])/2
    tensor=torch.cat((gx.square().sum(1,keepdim=True),gy.square().sum(1,keepdim=True),(gx*gy).sum(1,keepdim=True)),1)
    tensor=F.avg_pool2d(F.pad(tensor,(4,4,4,4),mode='replicate'),9,1)
    a,b,c=tensor.unbind(1);trace=a+b
    # lambda_min/trace measures incompatible edge orientations within a9px
    # neighbourhood. Opposite colour polarity on one line has no penalty.
    discriminant=((a-b).square()+4*c.square()+1e-16).sqrt()
    defect=((trace-discriminant)/2).clamp_min(0)/(trace+1e-6)
    return defect,trace


def contour_coherence(p,t):
    dp,ep=orientation_defect(p)
    with torch.no_grad():
        dt,et=orientation_defect(t)
        # Preserve permitted target corners/junctions. Target edges also remain
        # weighted if a prediction tries to evade this loss by becoming flat.
        weight=torch.maximum(ep.detach(),et).sqrt().clamp_max(.2)
        weight=weight*(torch.maximum(ep.detach(),et)>1e-5)
        weight[...,:12,:]=0;weight[...,-12:,:]=0;weight[...,:,:12]=0;weight[...,:,-12:]=0
    excess=F.relu(dp-dt-.015)
    return (excess*weight).sum()/weight.sum().clamp_min(1e-6)


@torch.inference_mode()
def evaluate(net,rows):
    result=base_evaluate(net,rows)
    for record,row in zip(result['records'],rows):
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];target=m.t.r.tensor(m.t.r.read(row['target']))[None]
        record['short_contour_turn_excess']=float(contour_coherence(net(x)[0],target))
    result['mean']['short_contour_turn_excess']=sum(r['short_contour_turn_excess'] for r in result['records'])/len(rows)
    # Preserve the existing selection score for comparable reporting. The new
    # geometric criterion and actual previews remain separate review checks.
    return result


class ContourCoherenceTraining(ProjectedCountTraining):
    def additional_loss(self,p,t,native):
        loss,values=super().additional_loss(p,t,native)
        # At720p this targets turns over9px, not all broad object curvature.
        if native:
            samples=[];truth=[]
            for _ in range(2):
                y=self.rng.randrange(0,p.shape[-2]-384+1);x=self.rng.randrange(0,p.shape[-1]-384+1)
                samples.append(p[...,y:y+384,x:x+384]);truth.append(t[...,y:y+384,x:x+384])
            coherence=contour_coherence(torch.cat(samples),torch.cat(truth))
        else:coherence=contour_coherence(p,t)
        values['short_contour_turn_excess']=coherence
        return loss+4*coherence,values

    def preflight_extras(self):
        row=super().preflight_extras()
        assert row['values']['short_contour_turn_excess']>0
        return row

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(contour_coherence_training=dict(window720=9,target_corner_allowance=.015,weight=4),
            person_detail_policy='No identity/original-photo preservation; same stylized region and contour losses as other objects',
            external_runtime_filter=False,visual_acceptance=False)
        torch.save(state,path)


def regression():
    yy,xx=torch.meshgrid(torch.arange(128),torch.arange(160),indexing='ij')
    def edge(offset):return torch.where(xx<80+offset,-.5,.5)[None,None].expand(1,3,-1,-1).clone()
    straight=edge(0);zig=edge(torch.where((yy//3)%2==0,-3,3))
    slant=edge((yy-64)*.35);coarse=edge(((yy-64).abs()-20)*.5)
    exact=contour_coherence(straight,straight);bad=contour_coherence(zig,straight)
    assert exact==0 and bad>.04,(exact,bad)
    assert contour_coherence(slant,slant)==0 and contour_coherence(coarse,coarse)==0
    assert contour_coherence(straight*0,straight*0)==0
    leaf=zig.clone().requires_grad_();loss=contour_coherence(leaf,straight);loss.backward()
    assert torch.isfinite(leaf.grad).all() and leaf.grad.abs().sum()>0
    # Coherence alone cannot certify sharpness: the independent RGB edge term
    # must reject a wide blur even if its contour direction is straight.
    blur=F.avg_pool2d(F.pad(straight,(6,6,6,6),mode='replicate'),13,1)
    assert rgb_edge_vector_loss(blur,straight)>.1
    print('CONTOUR_COHERENCE_REGRESSION_PASS',dict(straight=float(exact),zigzag=float(bad),
        target_corners_preserved=True,blur_rejected_by_separate_edge_term=True),flush=True)


if __name__=='__main__':
    import sys
    torch.set_num_threads(2)
    regression()
    if '--check' not in sys.argv:
        driver.main(training_factory=ContourCoherenceTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
