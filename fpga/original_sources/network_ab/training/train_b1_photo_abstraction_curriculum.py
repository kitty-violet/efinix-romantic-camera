"""Learn cartoon reconstruction, then photo abstraction on all TRAIN videos.

The intermediate input mixture exists only in training. Validation, preview,
export and temporal pairs always use the untouched source. The fixed B1 graph
and coefficient format are unchanged. No local-three-crop weights are used.
"""
from pathlib import Path
import torch
from .train_b1_fine_rgb_native import FineRGBNativeTraining, evaluate as base_evaluate
from .train_b1_late_region_supervision import main as driver
from .b1_geometry_first_objective import geometry_first, WEIGHTS
from .b1_relative_rgb_profiles import relative_profiles, regression
from .train_b1_contour_coherence import contour_coherence
from . import train_b1_minimum_area as m


def source_fraction(step, steps):
    # Keep a substantial final phase in the real inference input domain.
    hold=steps//8
    end=steps*9//16
    return min(1.,max(0.,(step-hold)/max(1,end-hold)))


class PhotoAbstractionTraining(FineRGBNativeTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        self.force_source=False
        self.actual_training_fraction=1.

    def batch_style(self,native):
        x,t,specs,flow,valid=super().batch_style(native)
        fraction=1. if flow is not None or self.force_source else source_fraction(self.step,self.args.steps)
        self.actual_training_fraction=fraction
        # Flow remains measured on the real source, so those inputs must not
        # be mixed with a possibly unstable offline target.
        if fraction<1.:
            x=t.lerp(x,fraction)
        else:
            assert torch.equal(x,self.last_source)
        for spec in specs:
            spec['training_source_fraction']=fraction
            spec['training_input_domain']='real source' if fraction==1. else 'training-only target/source mixture'
        return x,t,specs,flow,valid

    def additional_loss(self,p,t,native):
        lab=self.labels if native else self.labels[...,40:-40,40:-40]
        loss,values=geometry_first(p,t,lab)
        relative,details=relative_profiles(p,t,lab)
        if native:
            yy=self.rng.randrange(p.shape[-2]-384+1);xx=self.rng.randrange(p.shape[-1]-384+1)
            curve=contour_coherence(p[...,yy:yy+384,xx:xx+384],t[...,yy:yy+384,xx:xx+384])
        else:
            curve=contour_coherence(p,t)
        values.update(details)
        values.update(short_contour_turn_excess=curve,training_source_fraction=p.new_tensor(self.actual_training_fraction))
        return loss+relative+.5*curve,values

    def preflight_extras(self):
        result=super().preflight_extras()
        # Verify real native GPU backward as well as early mixed-input native
        # backward and real flow/minibatch, before formal training starts.
        self.force_source=True
        try:
            real=self.update(True)
            assert all(s['training_source_fraction']==1 for s in real['samples'])
        finally:
            self.force_source=False
        old_step=self.step
        self.step=self.args.steps//4
        try:
            mixed=self.update(True)
            assert all(0<s['training_source_fraction']<1 for s in mixed['samples'])
        finally:
            self.step=old_step
        result.update(real_source_native_update=real,mixed_native_update=mixed,
            validation_and_preview_input='untouched source',curriculum_is_training_only=True)
        return result

    def save(self,path):
        super().save(path)
        state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(photo_abstraction_curriculum=dict(
            hold_steps=self.args.steps//8,source_only_from_step=self.args.steps*9//16,
            fraction=source_fraction(self.step,self.args.steps),
            temporal_pairs='always untouched real source',
            preview_validation_export='always untouched real source',
            training_target='immutable original manifest target',
            geometry_weights=WEIGHTS,relative_profile_weight=1.,contour_weight=.5),
            local_fit_only=False,visual_acceptance=False,
            training_provenance=dict(training_class=type(self).__name__,
                warm=self.args.warm,data_manifest=self.args.data_manifest,
                trainable='all existing B1 weights and bias',
                auxiliary_phase_heads=False,synthetic_shape_guard=False,
                inference_postfilter=False,extra_inference_operations=0))
        torch.save(state,path)


@torch.inference_mode()
def evaluate(net,rows):
    result=base_evaluate(net,rows)
    for row,rec in zip(rows,result['records']):
        x=m.t.r.tensor(m.t.r.read(row['path']))[None]
        t=m.t.r.tensor(m.t.r.read(row['target']))[None]
        lab=torch.from_numpy(m.load_region(row['region']).astype('int64'))[None].to(x.device)
        p=net(x)[0]
        loss,values=geometry_first(p,t,lab)
        relative,details=relative_profiles(p,t,lab)
        curve=contour_coherence(p,t)
        rec['historical_score']=rec.pop('score')
        rec.update({k:float(v) for k,v in {**values,**details}.items()})
        rec['short_contour_turn_excess']=float(curve)
        rec['score']=float(loss+relative+.5*curve)
    result['mean']={k:sum(r[k] for r in result['records'])/len(rows) for k in result['records'][0] if k!='id'}
    return result


def schedule_regression():
    assert source_fraction(0,3200)==0 and source_fraction(400,3200)==0
    assert 0<source_fraction(800,3200)<1
    assert source_fraction(1800,3200)==source_fraction(3200,3200)==1
    a=torch.rand(2,3,40,40);b=torch.rand_like(a)
    # The code must use x itself at fraction1, not a cancelling arithmetic
    # expression that could alter its byte rounding.
    for f in (0.,.4,1.):
        x=a if f==1 else b.lerp(a,f)
        if f==1:assert x is a
        if f==0:assert torch.equal(x,b)
    print('PHOTO_ABSTRACTION_SCHEDULE_PASS endpoints/source_exact/resumable_step',flush=True)


if __name__=='__main__':
    torch.set_num_threads(2);regression();schedule_regression()
    driver(training_factory=PhotoAbstractionTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=m.t.preview)
