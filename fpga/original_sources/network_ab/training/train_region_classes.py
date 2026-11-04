"""Learn region color membership, not RGB regression averages.

32-color palette fitted ONLY to training algorithm targets. Categorical output
needs an argmax/palette writer not present in production B1. No FPGA fit claim.
"""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import random
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch import nn
from torch.nn import functional as F
from . import region_budget_net as base
from .region_mechanism_probe import ROOT,RUN,tensor,rgb,l0_surface,gaussian,outline
from .runtime import configure_workspace,write_json,file_hash,contract_guard
from .engine import set_reproducible
from .train_b1_distillation import as_tensor,read_image


def class_plan():
    nodes=copy.deepcopy(base.plan()['nodes'])
    nodes=[n for n in nodes if not n['id'].startswith('res2_')]
    for n in nodes:
        n['inputs']=['res1_add' if p=='res2_add' else p for p in n['inputs']]
    nodes[-1].update(channels=32,output_shape_hwc=[720,1280,32],activation='none',macs=720*1280*8*32)
    return dict(nodes=nodes,macs=sum(n['macs'] for n in nodes),production=False,classes=32)


class ClassNet(base.RegionNet):
    def __init__(self):
        super().__init__();self.nodes=class_plan()['nodes']
        for key in list(self.layers):
            if key.startswith('res2_'):del self.layers[key]
        self.layers['rgb']=nn.Sequential(nn.Conv2d(8,32,1))
        nn.init.normal_(self.layers['rgb'][0].weight,std=.04);nn.init.zeros_(self.layers['rgb'][0].bias)
        # Zero mix blocks the trunk's gradients on step1 only; it learns after
        # the first update. Native features remain trainable throughout.


def source_manifest():
    return json.loads((ROOT/'runs/b1-cartoon-budget-20261004/data_manifest.json').read_text(encoding='utf-8'))


@torch.no_grad()
def labels(target,palette):
    result=[]
    for y in range(0,target.shape[-2],64):
        t=target[:,:,y:y+64]
        distance=(t[:,None]-palette[None,:,:,None,None]).square().sum(2)
        result.append(distance.argmin(1))
    return torch.cat(result,1)


def output(net,x,palette):
    return palette[net(x).argmax(1)].permute(0,3,1,2)


def prepare(out):
    out.mkdir(exist_ok=False);manifest=source_manifest();groups={}
    for row in manifest['pairs']:groups.setdefault(row['source'],[]).append(row)
    selected=[]
    for rows in groups.values():
        for i in sorted({0,len(rows)//2,len(rows)-1}):selected.append(rows[i])
    folder=out/'targets';folder.mkdir();records=[];pixels=[];rng=np.random.default_rng(2026100416)
    for i,row in enumerate(selected):
        for key in ('current','next'):
            path=Path(row[key]);assert file_hash(path)==row[key+'_sha256']
            source=read_image(str(path))
            with torch.no_grad():target=rgb(outline(l0_surface(gaussian(tensor(source),.9),.015)))
            target_path=folder/(f'{i:03d}_{key}.png');Image.fromarray(target).save(target_path)
            samples=target.reshape(-1,3);pixels.append(samples[rng.choice(len(samples),2048,replace=False)])
            records.append(dict(source=str(path),source_sha256=file_hash(path),target=str(target_path),target_sha256=file_hash(target_path),raw_source=row['source'],pair=i,position=key))
        print('CLASS_TARGET_PAIR',i+1,len(selected),flush=True)
    cv2.setRNGSeed(20261004)
    sampled=np.concatenate(pixels).astype(np.float32)/255
    _,_,centers=cv2.kmeans(sampled,32,None,(cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS,100,.0001),3,cv2.KMEANS_PP_CENTERS)
    order=np.argsort(centers@np.array([.299,.587,.114]));palette=(centers[order]*255).round().clip(0,255).astype(np.uint8)
    assert len(np.unique(palette,axis=0))==32
    pal=torch.tensor(palette.astype(np.float32)/255,device='cuda');counts=np.zeros(32,np.int64)
    for row in records:
        with torch.no_grad():lab=labels(tensor(read_image(row['target'])),pal)[0].cpu().numpy().astype(np.uint8)
        path=Path(row['target']).with_suffix('.labels.png');Image.fromarray(lab).save(path)
        counts+=np.bincount(lab.ravel(),minlength=32);row.update(labels=str(path),labels_sha256=file_hash(path))
    write_json(out/'data.json',dict(records=records,palette=palette.tolist(),class_counts=counts.tolist(),validation=manifest['validation'],
        source_splits=manifest['source_splits'],note='21 gap1 pairs/7 train sources. No validation frames fitted palette or labels.'))
    cfg=dict(seed=2026100416,steps=2000,lr=.003,lr_floor=.0003,crop=320,margin=20,batch=2,native_interval=100,
        representation='32 class logits -> argmax -> fixed training-fitted RGB palette',
        target='Gaussian0.9 -> L0 lambda0.015 -> continuous surface contours; algorithm targets, NOT original AnimeGAN.',
        temporal='Shift consistency on logits; no inference frame blending, source paste or flow.',
        precision='FP32; no AMP,TF32,EMA',new_inference_ops='ARGMAX32 + PALETTE LOOKUP; not production B1')
    write_json(out/'config.json',cfg);write_json(out/'graph.json',class_plan())
    write_json(out/'binding.json',dict(code={str(p):file_hash(p) for p in [Path(__file__),Path(base.__file__),Path(__file__).with_name('region_mechanism_probe.py')]},
        data_sha256=file_hash(out/'data.json'),config_sha256=file_hash(out/'config.json')))


class Training:
    def __init__(self,out):
        self.out=out;self.cfg=json.loads((out/'config.json').read_text());self.data=json.loads((out/'data.json').read_text(encoding='utf-8'))
        self.binding=json.loads((out/'binding.json').read_text());assert all(file_hash(p)==h for p,h in self.binding['code'].items())
        self.net=ClassNet().cuda().eval();self.opt=torch.optim.Adam(self.net.parameters(),lr=self.cfg['lr']);self.rng=random.Random(self.cfg['seed']);self.step=0
        self.palette=torch.tensor(self.data['palette'],device='cuda',dtype=torch.float32)/255
        counts=torch.tensor(self.data['class_counts'],device='cuda',dtype=torch.float32)
        self.weights=(counts.mean()/counts).sqrt().clamp(.5,3);self.weights/=self.weights.mean()
        self.cache=[(read_image(r['source']),np.asarray(Image.open(r['labels'])).copy()) for r in self.data['records']]

    def batch(self,native):
        xs=[];ys=[];spec=[];size=self.cfg['crop']
        for _ in range(1 if native else self.cfg['batch']):
            i=self.rng.randrange(len(self.cache));source,target=self.cache[i]
            h,w=(720,1280) if native else (size,size);x=self.rng.randrange(1280-w+1);y=self.rng.randrange(720-h+1)
            xs.append(as_tensor(source[y:y+h,x:x+w]));ys.append(torch.from_numpy(target[y:y+h,x:x+w].copy()).long())
            spec.append([i,x,y,w,h])
        return torch.stack(xs).cuda(),torch.stack(ys).cuda(),spec

    def update(self,native=None):
        start=time.perf_counter();native=(self.step+1)%self.cfg['native_interval']==0 if native is None else native
        x,t,spec=self.batch(native);p=self.net(x);m=self.cfg['margin'];pc=p[:,:,m:-m,m:-m];tc=t[:,m:-m,m:-m]
        ce=F.cross_entropy(pc,tc,weight=self.weights)
        # A one-pixel translation exposes the sole stride2 phase. Keep this
        # auxiliary weak enough to avoid erasing genuine class boundaries.
        shifted=F.pad(x[:1],(1,0,0,0),mode='replicate')[:,:,:,:x.shape[-1]];q=self.net(shifted)
        a=F.softmax(q[:,:,m:-m,m+1:-m],1);b=F.softmax(p[:1,:,m:-m,m:-m-1],1)
        phase=(a-b).square().mean()
        loss=ce+.5*phase
        lr=self.cfg['lr_floor']+(self.cfg['lr']-self.cfg['lr_floor'])*.5*(1+math.cos(math.pi*self.step/self.cfg['steps']))
        for g in self.opt.param_groups:g['lr']=lr
        self.opt.zero_grad(set_to_none=True);loss.backward()
        assert torch.isfinite(loss)
        norm=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.,error_if_nonfinite=True);self.opt.step();self.step+=1
        torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),ce=float(ce.detach()),phase=float(phase.detach()),grad=float(norm),lr=lr,
                    native=native,samples=spec,seconds=time.perf_counter()-start,accuracy=float((pc.argmax(1)==tc).float().mean().detach()))

    def state(self):return dict(model=self.net.state_dict(),optimizer=self.opt.state_dict(),step=self.step,sampler=self.rng.getstate(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),binding=self.binding,palette=self.data['palette'])

    def restore(self,s):
        assert s['binding']==self.binding and s['palette']==self.data['palette']
        self.net.load_state_dict(s['model']);self.opt.load_state_dict(s['optimizer']);self.step=s['step'];self.rng.setstate(s['sampler'])
        torch.set_rng_state(s['torch_rng']);torch.cuda.set_rng_state_all(s['cuda_rng'])

    @torch.no_grad()
    def evaluate(self):
        folder=self.out/f'validation_{self.step:06d}';folder.mkdir(exist_ok=False);rows=[]
        for r in self.data['validation']:
            source=read_image(r['source']);x=as_tensor(source)[None].cuda();p=output(self.net,x,self.palette);im=rgb(p)
            Image.fromarray(im).save(folder/(r['id']+'.png'))
            z=output(self.net,F.pad(x,(1,0,0,0),mode='replicate')[:,:,:,:1280],self.palette)
            rows.append(dict(id=r['id'],phase_rgb255=float((z[:,:,32:-32,33:-32]-p[:,:,32:-32,32:-33]).abs().mean()*255)))
        write_json(folder/'scores.json',dict(records=rows,not_blind=True,style_acceptance=False))


def preflight(out):
    tr=Training(out);rows=[];macs=[]
    hooks=[m.register_forward_hook(lambda m,a,y:macs.append(y.numel()*m.weight[0].numel())) for m in tr.net.modules() if isinstance(m,nn.Conv2d)]
    with torch.no_grad():tr.net(torch.zeros(1,3,720,1280,device='cuda'))
    for h in hooks:h.remove()
    assert sum(macs)==class_plan()['macs']<966758400
    rows=[tr.update(True),tr.update(False)];torch.save(tr.state(),out/'preflight.pt');ref=tr.update(False)
    expected={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};del tr;gc.collect();torch.cuda.empty_cache()
    tr=Training(out);tr.restore(torch.load(out/'preflight.pt',map_location='cpu',weights_only=True));rep=tr.update(False)
    err=max(float((expected[k]-v.cpu()).abs().max()) for k,v in tr.net.state_dict().items());assert err<2e-6 and ref['samples']==rep['samples']
    write_json(out/'preflight.json',dict(status='PASS',macs=sum(macs),convolutions=len(macs),parameters=sum(p.numel() for p in tr.net.parameters()),
        full720_backward=True,batch2=True,restore_max_error=err,tests=rows));print('CLASS_PREFLIGHT_PASS',sum(macs),err,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--stop',type=int,default=2000);p.add_argument('--resume',type=Path);a=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100416);cv2.setNumThreads(2);out=ROOT/'runs'/RUN/'classes32'
    if a.action=='prepare':prepare(out);return
    if a.action=='preflight':preflight(out);return
    assert json.loads((out/'preflight.json').read_text())['status']=='PASS'
    tr=Training(out)
    if a.resume:tr.restore(torch.load(a.resume,map_location='cpu',weights_only=True))
    else:assert not (out/'metrics.jsonl').exists();tr.evaluate()
    start=tr.step;write_json(out/f'launch_{start:06d}.json',dict(start=start,resume=str(a.resume),binding=tr.binding))
    with (out/'metrics.jsonl').open('a',encoding='utf-8',buffering=1) as f:
        while tr.step<min(a.stop,tr.cfg['steps']):
            row=tr.update();f.write(json.dumps(row)+'\n')
            if tr.step%50==0:print('CLASS_TRAIN',tr.step,round(row['loss'],4),round(row['accuracy'],3),flush=True)
            if tr.step%250==0 or tr.step==min(a.stop,tr.cfg['steps']):torch.save(tr.state(),out/f'checkpoint_{tr.step:06d}.pt');tr.evaluate()
    print('CLASS_SEGMENT_DONE',tr.step,flush=True)


if __name__=='__main__':main()
