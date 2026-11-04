"""Preserve thin structure contrast with ordinary actual B1 coefficients.

Supervision is balanced per1--4px width, rather than proportional to foreground
pixel area. Full11x11 coherent shape plus coarse and pixel RGB anchor remain.
All source flow/phase/perturbation guards retained; no inference operations.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .b1_narrow_rgb_recall import narrow_recall,regression
from .b1_coherent_rgb_patch_profiles import coherent_profiles
from .b1_complete_rgb_profiles import complete_rgb_profiles
from .train_b1_correspondence_native import CorrespondenceTraining
from .train_b1_fine_rgb_native import evaluate as base_evaluate
from .train_b1_late_region_supervision import main as driver
from . import train_b1_minimum_area as m


def objective(p,t,lab):
    fine,values=narrow_recall(p,t,lab)
    shape,sv=coherent_profiles(p,t,lab)
    pixel=(p-t).abs().mean();coarse=F.l1_loss(F.avg_pool2d(p,16),F.avg_pool2d(t,16))
    values.update(sv);values.update(raw_rgb_l1=pixel,coarse_rgb_l1=coarse)
    return 8*pixel+2*coarse+2*shape+2*fine,values


class NarrowRecallTraining(CorrespondenceTraining):
    # Inherit only exact-source/exact-target batch handling from the recent
    # correspondence experiment, NOT its patch matching appearance objective.
    def additional_loss(self,p,t,native):
        lab=self.labels if native else self.labels[...,40:-40,40:-40]
        loss,values=objective(p,t,lab)
        with torch.no_grad():values['complete_rgb_profile']=complete_rgb_profiles(p,t,lab)[0]
        return loss,values

    def save(self,path):
        # Bypass CorrespondenceTraining.save so checkpoints do not falsely
        # claim the discarded9x9 bidirectional matching objective is active.
        from .train_b1_fine_rgb_native import FineRGBNativeTraining
        FineRGBNativeTraining.save(self,path)
        state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(narrow_recall_supervision=dict(widths=[1,2,3,4],whole_patch_xy_shared=True,
            raw_target=True,loss_weights=dict(pixel=8,coarse=2,coherent=2,narrow=2),
            extra_inference_operations=0),visual_acceptance=False)
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
    driver(training_factory=NarrowRecallTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=m.t.preview)
