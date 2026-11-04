"""Correct scanline-independent matching in the real fixed-B1 RGB objective.

All training/evaluation inputs are real source. Use one2-D offset per11x11
boundary neighbourhood. Region mean/flatness, RGB colour distributions,
phase/noise/actual source-flow guards remain. No inference filters or heads.
"""
from pathlib import Path
import torch
from .train_b1_fine_rgb_native import FineRGBNativeTraining,evaluate as base_evaluate
from .train_b1_late_region_supervision import main as driver
from .b1_geometry_first_objective import geometry_first,WEIGHTS
from .b1_coherent_rgb_patch_profiles import coherent_profiles,regression
from .train_b1_contour_coherence import contour_coherence
from . import train_b1_minimum_area as m


def objective(p,t,lab):
    loss,values=geometry_first(p,t,lab)
    coherent,details=coherent_profiles(p,t,lab)
    # Replace the independently movable 1-D profiles, rather than piling a
    # contradictory second alignment on top. Retain the old value for audit.
    loss=loss-WEIGHTS['complete_rgb_profile']*values['complete_rgb_profile']+1.5*coherent
    values.update(details)
    return loss,values


class CoherentPatchTraining(FineRGBNativeTraining):
    def additional_loss(self,p,t,native):
        lab=self.labels if native else self.labels[...,40:-40,40:-40]
        loss,values=objective(p,t,lab)
        if native:
            yy=self.rng.randrange(p.shape[-2]-384+1);xx=self.rng.randrange(p.shape[-1]-384+1)
            curve=contour_coherence(p[...,yy:yy+384,xx:xx+384],t[...,yy:yy+384,xx:xx+384])
        else:curve=contour_coherence(p,t)
        values['short_contour_turn_excess']=curve
        return loss+.5*curve,values

    def preflight_extras(self):
        r=super().preflight_extras()
        assert r['update']['values']['coherent_patch_samples']>0
        assert r['update']['values']['coherent_rgb_patch']>0
        r['two_dimensional_shared_alignment_active']=True
        return r

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(coherent_patch_training=dict(patch720=11,offset='one joint xy +/-2 per full patch',
            channels='joint RGB',relative_contrast='target max8byte floor',weight=1.5,
            old_independent_profile_training_weight=0,old_profile_still_reported=True),
            training_provenance=dict(training_class=type(self).__name__,warm=self.args.warm,
                data_manifest=self.args.data_manifest,training_input='untouched real source',
                auxiliary_phase_heads=False,synthetic_shape_guard=False,extra_inference_operations=0),
            visual_acceptance=False)
        torch.save(state,path)


@torch.inference_mode()
def evaluate(net,rows):
    result=base_evaluate(net,rows)
    for row,rec in zip(rows,result['records']):
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];t=m.t.r.tensor(m.t.r.read(row['target']))[None]
        lab=torch.from_numpy(m.load_region(row['region']).astype('int64'))[None].to(x.device)
        p=net(x)[0];loss,values=objective(p,t,lab);curve=contour_coherence(p,t)
        rec['historical_score']=rec.pop('score');rec.update({k:float(v) for k,v in values.items()})
        rec['short_contour_turn_excess']=float(curve);rec['score']=float(loss+.5*curve)
    result['mean']={k:sum(r[k] for r in result['records'])/len(rows) for k in result['records'][0] if k!='id'}
    return result


if __name__=='__main__':
    torch.set_num_threads(2);regression()
    driver(training_factory=CoherentPatchTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=m.t.preview)
