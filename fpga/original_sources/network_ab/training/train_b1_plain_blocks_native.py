"""Controlled reuse of sparse colour transitions on the current actual B1.

Unlike the older rank-seed sparse branch, this starts from the complete
modern decoder, trains all211 raw TRAIN frames, and uses only pixel fidelity
plus joint RGB gradient sparsity as appearance loss. The existing shift,
real-flow and noise terms remain. No test-image fitting or inference filters.
"""
from pathlib import Path
import numpy as np,torch
from .fit_b1_fresh_contract_local import sparse_pixel_loss,sparse_regression
from .train_b1_fine_rgb_native import FineRGBNativeTraining,evaluate as base_evaluate
from .train_b1_late_region_supervision import main as driver
from .b1_complete_rgb_profiles import complete_rgb_profiles
from .b1_coherent_rgb_patch_profiles import coherent_profiles
from . import train_b1_minimum_area as m


def objective(p,t):
    pixel=(p-t).abs().mean();total=sparse_pixel_loss(p,t)
    # Keep the whole appearance scale comparable to the historical ~1 loss;
    # this is an explicit fixed choice, not automatic gradient balancing.
    return 16*total,dict(plain_pixel=pixel,plain_sparse=(total-pixel)/.04,
        plain_appearance=total)


class PlainBlockTraining(FineRGBNativeTraining):
    def additional_loss(self,p,t,native):
        loss,values=objective(p,t)
        # This metric satisfies the inherited diagnostic contract only. It
        # explicitly contributes no training gradient or loss coefficient.
        lab=self.labels if native else self.labels[...,40:-40,40:-40]
        with torch.no_grad():
            profile,_=complete_rgb_profiles(p,t,lab)
        values['complete_rgb_profile']=profile
        return loss,values

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(plain_block_training=dict(pixel_weight=16,sparse_weight=.64,sigma_byte=4,
            region_means=False,coherent_profile_training=False,prior_sparse_principle_reused=True),
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
        lab=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None].to(x.device)
        p=net(x)[0];loss,values=objective(p,t);coherent,_=coherent_profiles(p,t,lab)
        rec['historical_score']=rec.pop('score');rec.update({k:float(v) for k,v in values.items()})
        rec.update(score=float(loss),coherent_rgb_patch=float(coherent))
    result['mean']={k:sum(v[k] for v in result['records'])/len(rows) for k in result['records'][0] if k!='id'}
    return result


if __name__=='__main__':
    torch.set_num_threads(2);sparse_regression()
    driver(training_factory=PlainBlockTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=m.t.preview)
