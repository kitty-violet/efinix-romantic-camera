"""Require definite field decisions in known region interiors.

The preceding RGB training could use the clipping ramps across almost the
entire image, yielding a soft photograph. Here a supervised score margin is
measured per field only away from its true boundary, with worst-quarter focus.
All margin/mask operations are training/evaluation only; B1 forward is fixed.
"""
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from .train_b1_symmetric_binary_coverage import BinaryCoverageTraining,evaluate as base_evaluate,driver,preview
from .b1_binary_layout import fields_for_layout
from .train_b1_ternary_regions import region_codes
from . import train_b1_minimum_area as m


def core_confidence(score,truth,margin=8):
    core=(F.max_pool2d(truth,5,1,2)==0)|(-F.max_pool2d(-truth,5,1,2)==1)
    core=core.clone();core[...,:4,:]=False;core[...,-4:,:]=False;core[...,:,:4]=False;core[...,:,-4:]=False
    signed=(2*truth-1)*(score-64)
    if not core.any():
        zero=score.sum()*0
        return zero,dict(core_margin_loss=zero,core_unconfident_fraction=zero,core_wrong_fraction=zero)
    violation=F.relu(1-signed[core]/margin)
    errors=F.smooth_l1_loss(violation,torch.zeros_like(violation),beta=1,reduction='none')
    loss=errors.topk(max(1,len(errors)//4)).values.mean()
    return loss,dict(core_margin_loss=loss,core_unconfident_fraction=(signed[core]<margin).float().mean(),
        core_wrong_fraction=(signed[core]<0).float().mean())


def score360(net,nodes):
    key='up1_high_dw';s=(getattr(net,key+'_m')/131072).float().flatten()
    w=net.round(net.weight[key]/s[:,None,None,None]).clamp(-127,127)*s[:,None,None,None]
    b=net.round(net.bias[key]/s)*s
    return F.conv2d(F.pad(nodes['up1_nn'],(1,1,1,1),mode='replicate'),w,b,groups=16)[:,:15]


class CoreConfidenceTraining(BinaryCoverageTraining):
    def __init__(self,args,data):
        super().__init__(args,data);self.net.confidence_palette_key=self.pk

    def additional_loss(self,p,t,native):
        loss,values=super().additional_loss(p,t,native)
        score=score360(self.net,self.nodes);truth=F.avg_pool2d(self.target_fields(self.class_labels),2)
        if not native:score=score[...,20:-20,20:-20];truth=truth[...,20:-20,20:-20]
        extra,terms=core_confidence(score,truth);values.update(terms)
        return loss+4*extra,values

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(supervised_binary_core_margin=dict(score_units=8,weight=4,worst_fraction=.25,boundary_exclusion360=2),
            confidence_inference_ops=0,visual_acceptance=False);torch.save(state,path)


@torch.inference_mode()
def evaluate(net,rows):
    result=base_evaluate(net,rows)
    for record,row in zip(result['records'],rows):
        code=torch.from_numpy(region_codes(row['target'],row['region'],net.confidence_palette_key).astype(np.int64))[None].cuda()
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];nodes=net(x,all_nodes=True)
        _,terms=core_confidence(score360(net,nodes),F.avg_pool2d(fields_for_layout(code,net.binary_layout_setup),2));record.update({k:float(v) for k,v in terms.items()})
        field=nodes['up2_low_dw'][:,:15];record['any_field_in_transition']=float(((field>0)&(field<127)).any(1).float().mean())
    for k in ('core_margin_loss','core_unconfident_fraction','core_wrong_fraction','any_field_in_transition'):
        result['mean'][k]=sum(r[k] for r in result['records'])/len(rows)
    mean=result['mean'];mean['without_core_selection_score']=mean['score'];mean['score']+=4*mean['core_margin_loss']
    return result


def regression():
    truth=torch.zeros(1,2,48,64);truth[...,32:]=1;sign=2*truth-1
    exact,terms=core_confidence(64+16*sign,truth);assert exact==0
    uncertain,terms=core_confidence(torch.full_like(truth,64),truth);assert torch.allclose(uncertain,torch.tensor(.5))
    wrong,terms=core_confidence(64-16*sign,truth);assert torch.allclose(wrong,torch.tensor(2.5))
    x=torch.full_like(truth,64,requires_grad=True);loss,terms=core_confidence(x,truth);loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert (x.grad[truth==0]>=0).all() and (x.grad[truth==1]<=0).all()
    boundary=64+16*sign;boundary[...,30:34]=64;loss,_=core_confidence(boundary,truth);assert loss==0
    soft=torch.full_like(truth,.5);loss,_=core_confidence(torch.full_like(truth,64),soft);assert loss==0
    print('CORE_CONFIDENCE_REGRESSION_PASS exact0/uncertain.5/wrong2.5/boundary_excluded/gradient',flush=True)


if __name__=='__main__':
    import sys
    torch.set_num_threads(2);regression()
    if '--check' not in sys.argv:
        driver.main(training_factory=CoreConfidenceTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
