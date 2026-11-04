"""Seven full720 TRAIN scenes, supervised at the final colour coordinates.

An explicitly limited fitting experiment. Actual integer previews determine
whether this objective is useful; no video/generalization claim is implied.
"""
import argparse
import copy
import json
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from .train_b1_seven_scene_lbfgs import BoundedSevenFit
from .probe_b1_native_region_objective import native_region_loss
from .train_b1_count_regions import count_loss
from .fit_b1_sixteen_regions import region_class_map
from .train_b1_temporal_count_regions import revised_classes
from .minimum_area_blocks import components, boundary_mask
from . import train_b1_minimum_area as m
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .engine import set_reproducible


class NativeRegionFit(BoundedSevenFit):
    def __init__(self,args,data):
        super().__init__(args,data)
        self.full_targets={}
        for key,x,_ in self.training.fit_cache:
            row=self.training.by_id[key];entry=self.training.class_overrides.get(key)
            codes=revised_classes(entry['path'],entry['sha256']) if entry else region_class_map(row['target'],row['region'],self.training.pk)
            labs=components(codes);valid=cv2.dilate(boundary_mask(codes),np.ones((3,3),np.uint8))==0
            c=torch.tensor(codes.astype('int64'),device=x.device)[None]
            wanted=torch.stack((c//25,(c//5)%5,c%5),1).float()
            self.full_targets[key]=(wanted,torch.tensor(labs.astype('int64'),device=x.device)[None],torch.tensor(valid,device=x.device)[None])

    def update(self,crop=False):
        started=time.perf_counter();before=self.calls;details=[]
        def closure():
            self.net.quantized=False;self.opt.zero_grad(set_to_none=True);total=0.;details.clear()
            for key,x,truth in self.training.fit_cache:
                wanted,regions,valid=self.full_targets[key]
                if crop:
                    xx=x[...,192:512,384:704];tt=[v[...,116:236,212:332] for v in truth]
                    wanted=wanted[...,192:512,384:704];regions=regions[...,192:512,384:704];valid=valid[...,192:512,384:704]
                else:xx=x;tt=truth
                nodes=self.net(xx,all_nodes=True);logits=(nodes['up1_high_dw'][:,:15]-63.5)/4
                ce=count_loss(logits[...,20:-20,20:-20] if crop else logits,tt)
                native,parts=native_region_loss(self.net,logits,wanted,regions,valid,40 if crop else 0)
                loss=(ce+self.args.region_weight*native)/len(self.training.fit_cache)
                loss.backward();total+=loss.detach()
                details.append(dict(id=key,count_ce=float(ce.detach()),**parts))
                del nodes,logits,loss,native,ce
            self.calls+=1
            assert torch.isfinite(total) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.parameters)
            return total
        before_loss=self.opt.step(closure);self.step+=1;self.training.step=self.step;torch.cuda.synchronize()
        return dict(step=self.step,loss_before_step=float(before_loss),closure_calls=self.calls-before,
            cumulative_closures=self.calls,native720=not crop,per_scene=copy.deepcopy(details),seconds=time.perf_counter()-started)

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(native_region_coordinate_objective=True,region_weight=self.args.region_weight,
            objective='balanced count CE + weight*(final720 coordinate interval + within-labelled-region variance)',
            trainable_native_postfilter=False,minimum_area_inference_guarantee=False)
        torch.save(s,path)

    def restore(self,path):
        s=torch.load(path,map_location='cpu',weights_only=True)
        assert s.get('native_region_coordinate_objective') and s['region_weight']==self.args.region_weight
        super().restore(path)

    @torch.no_grad()
    def report(self,data,dest):
        result=super().report(data,dest);self.net.quantized=True;records=[]
        try:
            for key,x,_ in self.training.fit_cache:
                nodes=self.net(x,all_nodes=True);wanted,_,valid=self.full_targets[key]
                actual=nodes['up2_high_dw'][:,:3]
                exact=(actual==wanted).all(1)
                records.append(dict(id=key,final720_colour_accuracy=float(exact.float().mean()),
                    interior_colour_accuracy=float(exact[valid].float().mean())))
            result['final720_training_fit']=records
            print('NATIVE_REGION_FULL720_FIT',self.step,records,flush=True)
            return result
        finally:self.net.quantized=False


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--steps',type=int,default=150);ap.add_argument('--region-weight',type=float,default=1.)
    ap.add_argument('--resume',default='');a=ap.parse_args()
    configure_workspace();contract_guard();set_reproducible(20261006149);torch.set_num_threads(2);cv2.setNumThreads(1)
    data=m.load_data();dest=m.OUT/a.name;dest.mkdir(exist_ok=False);here=Path(__file__).parent
    sources=[Path(__file__),here/'probe_b1_native_region_objective.py',here/'train_b1_seven_scene_lbfgs.py',
        here/'probe_b1_bounded_lbfgs.py',here/'train_b1_seven_scene_batch.py',here/'train_b1_seven_scene_fit.py',
        here/'train_b1_count_regions.py',here/'train_b1_learned_colour.py',here/'minimum_area_blocks.py']
    for p in sources:shutil.copy2(p,dest/p.name)
    write_json(dest/'config.json',dict(args=vars(a),warm_sha256=file_hash(a.warm),
        code={str(p):file_hash(p) for p in sources},data_sha256=file_hash(m.OUT/'data.json'),
        fp32_no_amp_no_tf32=True,actual_integer_preview=True,extra_inference_operations=0))
    tr=NativeRegionFit(a,data)
    if a.resume:tr.restore(a.resume)
    first=tr.update();crop=tr.update(True);tr.save(dest/'preflight.pt')
    expected=tr.update();state=copy.deepcopy(tr.net.state_dict());other=NativeRegionFit(a,data)
    other.restore(dest/'preflight.pt');actual=other.update()
    error=max(float((v-state[k]).abs().max()) for k,v in other.net.state_dict().items())
    assert error<1e-5 and actual['closure_calls']==expected['closure_calls']
    write_json(dest/'preflight.json',dict(status='PASS',native=first,crop=crop,resumed=actual,weight_error=error))
    print('NATIVE_REGION_GPU_PREFLIGHT_PASS',error,flush=True)
    del tr,other;torch.cuda.empty_cache();tr=NativeRegionFit(a,data)
    if a.resume:tr.restore(a.resume)
    history=[tr.report(data,dest)];tr.save(dest/f'candidate_{tr.step:06d}.pt');began=time.perf_counter()
    with (dest/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        while tr.step<a.steps:
            assert shutil.disk_usage(dest).free>512*1024**2
            row=tr.update();log.write(json.dumps(row)+'\n')
            if tr.step%10==0:print('NATIVE_REGION_STEP',row,flush=True)
            if tr.step%50==0 or tr.step==a.steps:
                history.append(tr.report(data,dest));tr.save(dest/f'candidate_{tr.step:06d}.pt');tr.save(dest/'last.pt')
                write_json(dest/'history.json',history)
                if (dest/'STOP_REQUEST.json').exists():break
    write_json(dest/'completion.json',dict(executed_steps=tr.step,calls=tr.calls,seconds=time.perf_counter()-began,
        history=history,visual_acceptance=False,generalization_claim=False))


if __name__=='__main__':main()
