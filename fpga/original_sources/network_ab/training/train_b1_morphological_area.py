"""Learn region decisions/colours around fixed B1 minimum-size region operators.

The binary threshold and morphology coefficients are immutable. An auxiliary
level-set classification loss trains the upstream integer features, avoiding
the zero gradients through hard erosion/dilation. No auxiliary loss is exported.
"""
import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import cv2
import torch
from torch.nn import functional as F

from . import train_b1_minimum_area as m
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible

THRESHOLDS=[(0,v) for v in [14,28,42,56,70,84,98,112]]+[(1,v) for v in [49,59,69,79]]+[(2,v) for v in [49,59,69,79]]


class MorphTraining(m.AreaTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        learned=('stem','down1__dw','down1__pw','down2__dw','up2_low_pw','rgb_head')
        self.frozen={}
        for key in self.net.weight:
            enabled=key in learned
            self.net.weight[key].requires_grad_(enabled);self.net.bias[key].requires_grad_(enabled)
            if not enabled:
                self.frozen[key]=(self.net.weight[key].detach().clone(),self.net.bias[key].detach().clone())
        # Keep the four simplified-contour channels intact; fit RGB combinations.
        self.net.weight['up2_low_pw'].register_hook(lambda g:torch.cat((g[:3],torch.zeros_like(g[3:])),0))
        self.net.bias['up2_low_pw'].register_hook(lambda g:torch.cat((g[:3],torch.zeros_like(g[3:])),0))
        mask=torch.zeros_like(self.net.weight['rgb_head']);mask[:,:3]=1
        self.net.weight['rgb_head'].register_hook(lambda g:g*mask)
        def remember(module,inputs,result):
            if isinstance(result,dict):self.nodes=result
        self.net.register_forward_hook(remember)

    def additional_loss(self,p,target,native):
        loss,values=super().additional_loss(p,target,native)
        feature=self.nodes['down2.dw']
        # target has already had the 40px crop margin removed; match that at180p.
        if not native:feature=feature[...,10:-10,10:-10]
        z=F.avg_pool2d((target+1)*64,4);r,g,b=z[:,0:1],z[:,1:2],z[:,2:3]
        colour=torch.cat(((r+2*g+b)/4,(r-b)/2+64,(-r+2*g-b)/4+64),1)
        logits=[];truth=[]
        for c,threshold in THRESHOLDS:
            logits.append(((feature[:,c:c+1]+feature[:,c+3:c+4])/2-(threshold+.5))/3)
            truth.append((colour[:,c:c+1]>threshold+.5).float())
        region=F.binary_cross_entropy_with_logits(torch.cat(logits,1),torch.cat(truth,1))
        values['predict_region_bce']=region
        return loss+self.args.classification*region,values

    def invariant(self):
        for key,(w,b) in self.frozen.items():
            assert torch.equal(self.net.weight[key],w) and torch.equal(self.net.bias[key],b),key
        for key in ('down2.pw','res2_add','res3_add'):
            a=self.nodes[key];assert ((a==0)|(a==127)).all(),key


def preflight(args,d,dest,factory=MorphTraining):
    a=factory(args,d);native=a.update(True);a.invariant();crop=a.update(False);a.update(False);a.update(False)
    a.save(dest/'preflight.pt');expected=a.update(False);a.invariant();state=copy.deepcopy(a.net.state_dict())
    b=factory(args,d);b.restore(dest/'preflight.pt');actual=b.update(False);b.invariant()
    error=max(float((v-state[k]).abs().max()) for k,v in b.net.state_dict().items())
    assert error<1e-5 and actual['samples']==expected['samples']
    write_json(dest/'preflight.json',dict(status='PASS',native=native,crop=crop,resume_error=error,
        frozen_region_weights_unchanged=True,binary_region_planes_verified=True))
    print('MORPH_TRAIN_PREFLIGHT',error,flush=True)
    del a,b;torch.cuda.empty_cache()


def main(training_factory=MorphTraining,extra_sources=()):
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True)
    ap.add_argument('--warm',default=str(m.OUT/'morphological-outlined/initial.pt'))
    ap.add_argument('--steps',type=int,default=2500);ap.add_argument('--interval',type=int,default=500)
    ap.add_argument('--lr',type=float,default=.00015);ap.add_argument('--classification',type=float,default=.4)
    ap.add_argument('--area-weight',type=float,default=.7);ap.add_argument('--preflight-only',action='store_true')
    ap.set_defaults(seed=2026100661,minimum_area=256,freeze_head=False,projection=0.,init='identity',
        aux=0,perceptual=0,qat=True,scale_lr=0,crop=256,batch=3,hard=False,shift=.15,temporal=.25,
        mode='direct',adversarial=0,regions=0,qat_teacher='',critic_warm='',critic_bands=False,
        tied_regions=False,colour=False,rotate=False,latent_refine=0)
    args=ap.parse_args();configure_workspace();set_reproducible(args.seed);cv2.setNumThreads(2);contract_guard()
    args.training_class=training_factory.__name__
    d=m.load_data();dest=m.OUT/args.name;dest.mkdir(exist_ok=False)
    paths=[Path(__file__),Path(m.__file__),Path(m.t.__file__),Path(m.t.s.__file__),Path(m.t.r.__file__),Path(m.t.r.c.__file__)]
    paths.extend(extra_sources)
    for p in paths:shutil.copy2(p,dest/p.name)
    write_json(dest/'config.json',dict(args=vars(args),code={str(p):file_hash(p) for p in paths},
        data_sha256=file_hash(m.OUT/'data.json'),warm_sha256=file_hash(args.warm),contract_sha256=m.t.r.CONTRACT_SHA,
        precision='FP32_without_AMP_or_TF32',train_only_auxiliary='pre-threshold region classification'))
    preflight(args,d,dest,training_factory)
    if args.preflight_only:return
    tr=training_factory(args,d);e=m.evaluate(tr.net,d['validation']);best=e['mean']['score'];history=[dict(step=0,validation=e)]
    tr.save(dest/'best.pt');m.t.preview(tr.net,d['validation'],dest/'initial-preview');start=time.perf_counter();stop=None
    with (dest/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        while tr.step<args.steps:
            assert shutil.disk_usage(m.OUT).free>512*1024**2
            row=tr.update();log.write(json.dumps(row)+'\n')
            if tr.step%100==0:tr.invariant();print('MORPH_TRAIN_STEP',tr.step,round(row['loss'],5),flush=True)
            if tr.step%args.interval==0 or tr.step==args.steps:
                tr.invariant();e=m.evaluate(tr.net,d['validation']);history.append(dict(step=tr.step,validation=e))
                tr.save(dest/f'candidate_{tr.step:06d}.pt');tr.save(dest/'last.pt')
                m.t.preview(tr.net,d['validation'],dest/f'preview_{tr.step:06d}')
                if e['mean']['score']<best:best=e['mean']['score'];tr.save(dest/'best.pt')
                write_json(dest/'history.json',history);print('MORPH_TRAIN_EVAL',tr.step,e['mean'],flush=True)
                req=dest/'STOP_REQUEST.json'
                if req.exists():stop=json.loads(req.read_text(encoding='utf-8'));assert stop.get('reason');break
    state=torch.load(dest/'best.pt',map_location='cpu',weights_only=True);tr.net.load_state_dict(state['model'])
    m.t.preview(tr.net,d['validation'],dest/'preview')
    write_json(dest/'completion.json',dict(executed_steps=tr.step,selected_step=state['step'],
        elapsed_seconds=time.perf_counter()-start,validation=m.evaluate(tr.net,d['validation']),
        stop='reviewed_stop' if stop else 'stage_budget',stop_evidence=stop,visual_acceptance=False))
    print('MORPH_TRAIN_COMPLETE',tr.step,state['step'],flush=True)


if __name__=='__main__':main()
