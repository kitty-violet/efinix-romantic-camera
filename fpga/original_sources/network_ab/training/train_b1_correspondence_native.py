"""Replace ambiguous exact region alignment by complete local correspondences.

All211 TRAIN source frames, unchanged B1. No region matching/pasting in
inference. Existing source-flow, phase and perturbation stability stay on.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .b1_bidirectional_patch_matching import matching,regression
from .b1_complete_rgb_profiles import complete_rgb_profiles
from .train_b1_fine_rgb_native import FineRGBNativeTraining,evaluate as base_evaluate
from .train_b1_late_region_supervision import main as driver
from . import train_b1_minimum_area as m


def objective(p,t):
    shape,values=matching(p,t)
    coarse=F.l1_loss(F.avg_pool2d(p,32),F.avg_pool2d(t,32))
    values['coarse_colour_anchor']=coarse
    return 8*shape+2*coarse,values


class CorrespondenceTraining(FineRGBNativeTraining):
    def batch_style(self,native):
        # Correspondence already tolerates geometry displacement. Do not
        # bilinearly warp/blend region RGB labels before matching their shape.
        x,t,specs,flow,valid=m.AreaTraining.batch_style(self,native)
        for i,s in enumerate(specs):
            row=self.by_id.get(s['id']) or self.by_stem[s['id']];xx,yy=s['xy'];w,h=s['wh']
            for k,z in (('path',x),('target',t)):
                ar=m.t.r.read(row[k])[yy:yy+h,xx:xx+w]
                if s['flip']:ar=ar[:,::-1]
                expected=torch.from_numpy(ar.transpose(2,0,1).copy()).to(x.device).float()/128-1
                assert torch.equal(z[i],expected)
        self.last_source=x.detach()
        return x,t,specs,flow,valid

    def additional_loss(self,p,t,native):
        loss,values=objective(p,t)
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        with torch.no_grad():values['complete_rgb_profile']=complete_rgb_profiles(p,t,labels)[0]
        return loss,values

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(bidirectional_correspondence=dict(patch=9,search_radius720=8,search_stride=2,samples=384,
            paired_RGB_offset=True,raw_target_no_flow_blending=True,appearance_weights=[8,2],exact_pixel_loss=False,
            region_mean_loss=False,region_flatness_loss=False,coarse_colour_scale=32,extra_inference_operations=0),
            visual_acceptance=False)
        torch.save(state,path)


@torch.inference_mode()
def evaluate(net,rows):
    result=base_evaluate(net,rows)
    for row,rec in zip(rows,result['records']):
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];t=m.t.r.tensor(m.t.r.read(row['target']))[None]
        loss,values=objective(net(x)[0],t)
        rec['historical_score']=rec.pop('score');rec.update(score=float(loss),**{k:float(v) for k,v in values.items()})
    result['mean']={k:sum(r[k] for r in result['records'])/len(rows) for k in result['records'][0] if k!='id'}
    return result


if __name__=='__main__':
    torch.set_num_threads(2);regression()
    driver(training_factory=CorrespondenceTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=m.t.preview)
