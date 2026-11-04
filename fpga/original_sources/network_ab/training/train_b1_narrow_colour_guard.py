"""Joint thin-structure recall, RGB fringe limits and interior flatness.

Existing B1 only. The PCA colour-line constraint is a TRAIN loss and never
part of video inference or the style package. Original target PNGs unchanged.
"""
from pathlib import Path
import torch
from .train_b1_narrow_recall_native import NarrowRecallTraining,objective as recall_objective
from .b1_local_rgb_envelope import local_envelope,core_error_variation,regression
from .b1_complete_rgb_profiles import complete_rgb_profiles
from .train_b1_fine_rgb_native import evaluate as base_evaluate
from .train_b1_late_region_supervision import main as driver
from . import train_b1_minimum_area as m


def objective(p,t,lab):
    primary,values=recall_objective(p,t,lab)
    colour,details=local_envelope(p,t,lab);core=core_error_variation(p,t,lab)
    values.update(details);values['rgb_core_error_variation']=core
    # Three fixed TRAIN crops: at8x the fringe-gradient norm was only6--12%
    # of primary and core2--3%. These fixed weights bring guards to roughly
    # one quarter and one twentieth, without replacing the structure target.
    return primary+32*colour+16*core,values


class NarrowColourGuardTraining(NarrowRecallTraining):
    def additional_loss(self,p,t,native):
        lab=self.labels if native else self.labels[...,40:-40,40:-40]
        loss,values=objective(p,t,lab)
        with torch.no_grad():values['complete_rgb_profile']=complete_rgb_profiles(p,t,lab)[0]
        return loss,values

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(rgb_line_guard=dict(patch720=9,rank2_to_rank1_max=.02,
            orthogonal_and_endpoint_tolerance_byte=2,weight=32,core_margin720=3,core_weight=16,
            mean_accumulation='float64',extra_inference_operations=0),visual_acceptance=False)
        torch.save(state,path)


@torch.inference_mode()
def evaluate(net,rows):
    result=base_evaluate(net,rows)
    for row,rec in zip(rows,result['records']):
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];t=m.t.r.tensor(m.t.r.read(row['target']))[None]
        lab=torch.from_numpy(m.load_region(row['region']).astype('int64'))[None].to(x.device)
        loss,values=objective(net(x)[0],t,lab)
        rec['historical_score']=rec.pop('score');rec.update(score=float(loss),**{k:float(v) for k,v in values.items()})
    result['mean']={k:sum(r[k] for r in result['records'])/len(rows) for k in result['records'][0] if k!='id'}
    return result


if __name__=='__main__':
    torch.set_num_threads(2);regression()
    driver(training_factory=NarrowColourGuardTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=m.t.preview)
