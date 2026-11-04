"""Make explicit colour-field scores flat before threshold amplification.

This replaces the probabilistic count/canonical/weak readout objective. RGB
boundary and temporal checks remain. It is a controlled reuse of region
consistency on explicit colour decisions, not a new inference operator.
"""
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from .train_b1_symmetric_binary_coverage import BinaryCoverageTraining,driver,preview
from .train_b1_binary_core_confidence import score360,evaluate as confidence_evaluate
from .train_b1_ternary_regions import region_codes
from .b1_binary_layout import fields_for_layout
from .train_b1_winner_surrogate import ExactValueProxyGradient
from . import train_b1_minimum_area as m


def score_region_loss(score,truth,labels):
    # A +/-48 target provides a finite distance from the threshold. In contrast
    # to an interior-only hinge, confidence cannot grow without spatial control.
    wanted=16+96*truth
    # Already-confident correct fields need no pull back from0/127 to16/112.
    # STE retains correction of confidently wrong fields beyond these limits.
    bounded=ExactValueProxyGradient.apply(score.clamp(16,112),score)
    direct=F.smooth_l1_loss((bounded-wanted)/16,torch.zeros_like(score),beta=1)
    lo=-F.max_pool2d(-labels[:,None].float(),2,2)[:,0]
    hi=F.max_pool2d(labels[:,None].float(),2,2)[:,0]
    valid=lo==hi;terms=[]
    for ss,ll,ok in zip(bounded,lo.long(),valid):
        ids=ll[ok];unique,index=torch.unique(ids,return_inverse=True)
        if len(ids)==0:continue
        value=ss[:,ok];count=torch.bincount(index,minlength=len(unique)).float()
        mu=ss.new_zeros(ss.shape[0],len(unique)).scatter_add(1,index[None].expand_as(value),value)/count.clamp_min(1)
        error=F.smooth_l1_loss((value-mu[:,index])/16,torch.zeros_like(value),beta=1,reduction='none').mean(0)
        per=error.new_zeros(len(unique)).scatter_add(0,index,error)/count.clamp_min(1)
        weight=count.sqrt();terms.append((per*weight).sum()/weight.sum())
    flat=torch.stack(terms).mean() if terms else score.sum()*0
    return direct+2*flat,dict(predecision_target_score=direct,predecision_region_variation=flat)


class PredecisionTraining(BinaryCoverageTraining):
    def __init__(self,args,data):
        super().__init__(args,data);self.net.confidence_palette_key=self.pk

    def additional_loss(self,p,t,native):
        loss,v=super().additional_loss(p,t,native)
        previous=.5*v['weighted_binary_count_ce']+.5*v['canonical_binary_ce']+.2*v['binary_score_margin']
        loss=loss-self.args.classification*previous
        score=score360(self.net,self.nodes);truth=F.avg_pool2d(self.target_fields(self.class_labels),2);labels=self.labels
        if not native:
            score=score[...,20:-20,20:-20];truth=truth[...,20:-20,20:-20];labels=labels[...,40:-40,40:-40]
        extra,terms=score_region_loss(score,truth,labels);v.update(terms)
        return loss+self.args.classification*extra,v

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(predecision_region_supervision=dict(endpoints=[16,112],score_normalization=16,
            within_region_weight=2,replaces_count_canonical_readout_loss=True),visual_acceptance=False)
        torch.save(state,path)


@torch.inference_mode()
def evaluate(net,rows):
    result=confidence_evaluate(net,rows)
    for record,row in zip(result['records'],rows):
        code=torch.from_numpy(region_codes(row['target'],row['region'],net.confidence_palette_key).astype(np.int64))[None].cuda()
        labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None].cuda()
        nodes=net(m.t.r.tensor(m.t.r.read(row['path']))[None],all_nodes=True)
        _,terms=score_region_loss(score360(net,nodes),F.avg_pool2d(fields_for_layout(code,net.binary_layout_setup),2),labels)
        record.update({k:float(v) for k,v in terms.items()})
    for key in ('predecision_target_score','predecision_region_variation'):
        result['mean'][key]=sum(r[key] for r in result['records'])/len(rows)
    result['mean']['score']=result['mean']['without_core_selection_score']+result['mean']['predecision_target_score']+2*result['mean']['predecision_region_variation']
    return result


def regression():
    labels=torch.zeros(1,64,80,dtype=torch.long);labels[...,40:]=1
    truth=torch.zeros(1,3,32,40);truth[...,20:]=1;score=16+96*truth
    loss,terms=score_region_loss(score,truth,labels);assert loss==0
    bad=score.clone();bad[...,8:12,8:12]+=32;bad.requires_grad_()
    loss,terms=score_region_loss(bad,truth,labels);assert terms['predecision_region_variation']>0 and loss>0
    loss.backward();assert torch.isfinite(bad.grad).all() and bad.grad[...,8:12,8:12].mean()>0
    offset=score+2;_,terms=score_region_loss(offset,truth,labels)
    assert terms['predecision_region_variation']==0 and terms['predecision_target_score']>0
    extremes=truth*127;loss,_=score_region_loss(extremes,truth,labels);assert loss==0
    inverse=(127-extremes).requires_grad_();loss,_=score_region_loss(inverse,truth,labels);loss.backward()
    assert loss>0 and inverse.grad.abs().sum()>0
    print('PREDECISION_REGION_REGRESSION_PASS exact0/speckle/gradient/constant_offset/separate_boundaries',flush=True)


if __name__=='__main__':
    import sys
    torch.set_num_threads(2);regression()
    if '--check' not in sys.argv:
        driver.main(training_factory=PredecisionTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
