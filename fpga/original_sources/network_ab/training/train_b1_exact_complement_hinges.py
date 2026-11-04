"""Train the byte-exact complementary code with matching0/127 supervision."""
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from .train_b1_common_colour_hinges import CommonHingeTraining,driver,preview
from .train_b1_predecision_regions import score_region_loss
from .train_b1_symmetric_binary_coverage import evaluate as base_evaluate
from .train_b1_binary_core_confidence import score360,core_confidence
from .train_b1_ternary_regions import region_codes
from .b1_binary_layout import fields_for_layout
from . import train_b1_minimum_area as m


def calibrated_region_loss(score,truth,labels):
    # Reuse the calibrated confidence loss in its original units, explicitly
    # mapping the new integer endpoints to16/112. Forward B1 is not transformed.
    return score_region_loss(16+score*(96/127),truth,labels)


class ExactComplementTraining(CommonHingeTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        assert self.setup.get('exact_integer_complement') and self.setup['score_endpoints']==[0,127]
        assert self.setup['score_center']==63.5 and self.setup['gain']%2==1

    def additional_loss(self,p,t,native):
        loss,v=super().additional_loss(p,t,native)
        previous=v['predecision_target_score']+2*v['predecision_region_variation']
        score=score360(self.net,self.nodes);truth=F.avg_pool2d(self.target_fields(self.class_labels),2);labels=self.labels
        if not native:score=score[...,20:-20,20:-20];truth=truth[...,20:-20,20:-20];labels=labels[...,40:-40,40:-40]
        extra,terms=calibrated_region_loss(score,truth,labels);v.update(terms)
        return loss+self.args.classification*(extra-previous),v

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(exact_complement_training=True,predecision_units='0/127 mapped to16/112 for loss only',visual_acceptance=False)
        torch.save(state,path)


@torch.inference_mode()
def evaluate(net,rows):
    result=base_evaluate(net,rows)
    for record,row in zip(result['records'],rows):
        code=torch.from_numpy(region_codes(row['target'],row['region'],net.confidence_palette_key).astype(np.int64))[None].cuda()
        labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None].cuda()
        nodes=net(m.t.r.tensor(m.t.r.read(row['path']))[None],all_nodes=True);score=score360(net,nodes)
        truth=F.avg_pool2d(fields_for_layout(code,net.binary_layout_setup),2)
        _,terms=calibrated_region_loss(score,truth,labels);_,conf=core_confidence(score+.5,truth)
        record.update({k:float(v) for k,v in (terms|conf).items()})
        field=nodes['up2_low_dw'][:,:15];record['any_field_in_transition']=float(((field>0)&(field<127)).any(1).float().mean())
    for key in ('predecision_target_score','predecision_region_variation','core_margin_loss','core_unconfident_fraction','core_wrong_fraction','any_field_in_transition'):
        result['mean'][key]=sum(r[key] for r in result['records'])/len(rows)
    mean=result['mean'];mean['before_score_supervision_score']=mean['score'];mean['score']+=mean['predecision_target_score']+2*mean['predecision_region_variation']
    return result


def regression():
    truth=torch.zeros(1,3,32,40);truth[...,20:]=1
    labels=torch.zeros(1,64,80,dtype=torch.long);labels[...,40:]=1
    loss,_=calibrated_region_loss(127*truth,truth,labels);assert loss==0
    bad=(127*(1-truth)).requires_grad_();loss,_=calibrated_region_loss(bad,truth,labels);loss.backward()
    assert loss>0 and torch.isfinite(bad.grad).all() and bad.grad.abs().sum()>0
    print('EXACT_COMPLEMENT_LOSS_PASS correct_endpoints0/wrong_endpoints_gradient',flush=True)


if __name__=='__main__':
    import sys
    torch.set_num_threads(2);regression()
    if '--check' not in sys.argv:
        driver.main(training_factory=ExactComplementTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
