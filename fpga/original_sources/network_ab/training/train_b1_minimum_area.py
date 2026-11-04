"""Connected-area supervision with the unchanged deployable B1.

All connected-component processing is training-only. Actual exports are decoded
and run without this module, and their small-region violations are measured.
"""
import argparse
import copy
from functools import lru_cache
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from . import train_b1_style_recovery as t
from .minimum_area_blocks import OUT, make_target
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .engine import set_reproducible


@lru_cache(maxsize=48)
def load_region(path):
    with np.load(path) as z:return z['labels'].copy()


def region_loss(p, target, labels):
    """A whole labelled area must agree, rather than only neighbouring pixels."""
    flats=[];means=[];edges=[]
    for pi,ti,li in zip(p,target,labels):
        index=li.flatten().long();n=int(index.max())+1
        count=torch.bincount(index,minlength=n).float().clamp_min(1)
        ids=index[None].expand(3,-1)
        mu=pi.new_zeros(3,n).scatter_add(1,ids,pi.flatten(1))/count[None]
        mt=ti.new_zeros(3,n).scatter_add(1,ids,ti.flatten(1))/count[None]
        flats.append((pi.flatten(1)-mu[:,index]).abs().mean())
        means.append((mu[:,index]-mt[:,index]).abs().mean())
        for dim in (-1,-2):
            a=li.narrow(dim,1,li.shape[dim]-1);b=li.narrow(dim,0,li.shape[dim]-1)
            mask=(a!=b).float()[None]
            gp,gt=torch.diff(pi,dim=dim),torch.diff(ti,dim=dim)
            # Signed edge contrast: a constant image and a blurred boundary
            # cannot obtain a good score merely by having fewer edge pixels.
            edges.append(((gp-gt).abs()*mask).sum()/(3*mask.sum()).clamp_min(1))
    flat=torch.stack(flats).mean();mean=torch.stack(means).mean();edge=torch.stack(edges).mean()
    return 2.5*flat+.8*mean+.65*edge,dict(area_flatness=flat,area_colour=mean,area_boundary=edge)


class AreaTraining(t.Training):
    def __init__(self,args,data):
        super().__init__(args,data)
        self.by_id={r['id']:r for r in data['training']}
        self.by_stem={Path(r['path']).stem:r for r in data['training']}
        if args.freeze_head:
            for key in ('up2_high_dw','rgb_head'):
                self.net.weight[key].requires_grad_(False);self.net.bias[key].requires_grad_(False)

    def batch_style(self,native):
        x,target,specs,flow,valid=super().batch_style(native)
        labels=[]
        for spec in specs:
            row=self.by_id.get(spec['id']) or self.by_stem[spec['id']]
            xx,yy=spec['xy'];w,h=spec['wh']
            label=load_region(row['region'])[yy:yy+h,xx:xx+w]
            if spec['flip']:label=label[:,::-1]
            labels.append(torch.from_numpy(np.ascontiguousarray(label.astype(np.int64))).cuda())
        self.labels=torch.stack(labels)
        return x,target,specs,flow,valid

    def additional_loss(self,p,target,native):
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        loss,values=region_loss(p,target,labels)
        if self.args.projection and self.step%4==1 and not native:
            # Stop-gradient alternating region cleanup. This provides direct
            # correction of the actual prediction's tiny islands and teeth.
            rgb=np.array(t.r.c.rgb_image(p[0]))
            corrected,_,_=make_target(rgb,self.args.minimum_area)
            q=t.r.tensor(corrected)[None]
            projection=F.l1_loss(p[:1],q)
            loss+=self.args.projection*projection;values['small_island_correction']=projection
        return self.args.area_weight*loss,values


def load_data(manifest=None):
    d=json.loads(Path(manifest or OUT/'data.json').read_text(encoding='utf-8'))
    assert d['contract_sha256']==t.r.CONTRACT_SHA
    for row in d['training']+d['validation']:
        for key,sha in [('path','sha256'),('target','target_sha256'),('region','region_sha256')]:
            assert file_hash(row[key])==row[sha]
    for row in d['pairs']:assert file_hash(row['flow'])==row['flow_sha256']
    return d


@torch.inference_mode()
def evaluate(net,rows):
    records=[]
    for row in rows:
        x=t.r.tensor(t.r.read(row['path']))[None];target=t.r.tensor(t.r.read(row['target']))[None]
        p=net(x)[0];labels=torch.from_numpy(load_region(row['region']).astype(np.int64))[None].cuda()
        region,values=region_loss(p,target,labels)
        appearance,_=t.appearance(p,target,'structured')
        metrics=t.r.metrics(p,target);metrics.update({k:float(v) for k,v in values.items()})
        metrics.update(score=float(region+.25*appearance))
        records.append(dict(id=row['id'],**metrics))
    mean={k:sum(v[k] for v in records)/len(records) for k in records[0] if k!='id'}
    return dict(records=records,mean=mean)


def preflight(args,data,dest):
    a=AreaTraining(args,data);torch.cuda.reset_peak_memory_stats()
    native=a.update(True);crop=a.update(False);a.update(False);a.update(False)
    a.save(dest/'preflight.pt');expected=a.update(False);state=copy.deepcopy(a.net.state_dict())
    b=AreaTraining(args,data);b.restore(dest/'preflight.pt');actual=b.update(False)
    err=max(float((state[k]-v).abs().max()) for k,v in b.net.state_dict().items())
    assert err<1e-5 and expected['samples']==actual['samples']
    # The new region objective must penalize within-block noise, all-flat
    # collapse and softened edges on a two-region known example.
    lab=torch.zeros(1,96,96,dtype=torch.long,device='cuda');lab[:,:,48:]=1
    target=torch.zeros(1,3,96,96,device='cuda');target[:,:,:,:48]=-.5;target[:,:,:,48:]=.5
    noisy=(target+.15*torch.randn_like(target)).clamp(-1,1)
    blurred=F.avg_pool2d(F.pad(target,(3,3,3,3),mode='replicate'),7,1)
    scores={k:float(region_loss(z,target,lab)[0]) for k,z in
            [('exact',target),('noise',noisy),('constant',target*0),('blurred',blurred)]}
    assert scores['exact']==0 and min(scores[k] for k in ('noise','constant','blurred'))>.1
    write_json(dest/'preflight.json',dict(status='PASS',native=native,crop=crop,resume_error=err,
        region_loss_tests=scores,peak_gpu_bytes=torch.cuda.max_memory_allocated(),qat=args.qat))
    print('AREA_PREFLIGHT_PASS',err,scores,flush=True)
    del a,b;torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--warm',required=True)
    ap.add_argument('--steps',type=int,default=3000);ap.add_argument('--interval',type=int,default=500)
    ap.add_argument('--lr',type=float,default=.00025);ap.add_argument('--area-weight',type=float,default=1.)
    ap.add_argument('--projection',type=float,default=.7);ap.add_argument('--minimum-area',type=int,default=256)
    ap.add_argument('--freeze-head',action='store_true');ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--seed',type=int,default=2026100617)
    ap.set_defaults(init='identity',aux=0,perceptual=0,qat=True,scale_lr=0,crop=256,batch=3,
        hard=False,shift=.15,temporal=.25,mode='direct',adversarial=0,regions=0,qat_teacher='',
        critic_warm='',critic_bands=False,tied_regions=False,colour=False,rotate=False,latent_refine=0)
    args=ap.parse_args();configure_workspace();set_reproducible(args.seed);cv2.setNumThreads(2);contract_guard()
    d=load_data();assert d['minimum_area720']==args.minimum_area
    dest=OUT/args.name;dest.mkdir(exist_ok=False)
    paths=[Path(__file__),Path(t.__file__),Path(t.s.__file__),Path(t.r.__file__),Path(t.r.c.__file__),
           Path(__file__).with_name('minimum_area_blocks.py')]
    for path in paths:shutil.copy2(path,dest/path.name)
    write_json(dest/'config.json',dict(args=vars(args),data_sha256=file_hash(OUT/'data.json'),
        code={str(p):file_hash(p) for p in paths},warm_sha256=file_hash(args.warm),
        contract_sha256=t.r.CONTRACT_SHA,precision='FP32_without_AMP_or_TF32',
        inference='Unchanged integer B1; region segmentation/projection training only'))
    preflight(args,d,dest)
    if args.preflight_only:return
    tr=AreaTraining(args,d);initial=evaluate(tr.net,d['validation']);best=initial['mean']['score']
    history=[dict(step=0,validation=initial)];tr.save(dest/'best.pt');started=time.perf_counter();stop=None
    t.preview(tr.net,d['validation'],dest/'initial-preview')
    with (dest/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        while tr.step<args.steps:
            assert shutil.disk_usage(OUT).free>512*1024**2
            row=tr.update();log.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('AREA_STEP',args.name,tr.step,round(row['loss'],5),round(row['seconds'],3),flush=True)
            if tr.step%args.interval==0 or tr.step==args.steps:
                e=evaluate(tr.net,d['validation']);history.append(dict(step=tr.step,validation=e))
                tr.save(dest/f'candidate_{tr.step:06d}.pt');tr.save(dest/'last.pt')
                t.preview(tr.net,d['validation'],dest/f'preview_{tr.step:06d}')
                if e['mean']['score']<best:best=e['mean']['score'];tr.save(dest/'best.pt')
                write_json(dest/'history.json',history)
                print('AREA_EVAL',args.name,tr.step,e['mean'],flush=True)
                req=dest/'STOP_REQUEST.json'
                if req.exists():stop=json.loads(req.read_text(encoding='utf-8'));assert stop.get('reason');break
    state=torch.load(dest/'best.pt',map_location='cpu',weights_only=True);tr.net.load_state_dict(state['model'])
    t.preview(tr.net,d['validation'],dest/'preview')
    write_json(dest/'completion.json',dict(executed_steps=tr.step,selected_step=state['step'],
        elapsed_seconds=time.perf_counter()-started,validation=evaluate(tr.net,d['validation']),
        stop='reviewed_stop' if stop else 'stage_budget',stop_evidence=stop,visual_acceptance=False))
    print('AREA_COMPLETE',args.name,tr.step,state['step'],flush=True)


if __name__=='__main__':main()
