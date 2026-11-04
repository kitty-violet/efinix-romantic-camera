"""Distil the frozen visual recipe into the existing, unchanged B1 graph.

The expensive recipe is used to create training targets only. Inference is B1.
All artifacts belong to model-training; shared hardware and graph stay read-only.
"""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import random
import shutil
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from network_ab.contract_torch import build_fp32_model
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .train_b1_distillation import as_tensor, to_rgb, read_image, read_flow
from .engine import set_reproducible
from .coherent_shared_model import warp

ROOT = Path(r'E:\fpga\model-training-workspace\runs')
OUT = ROOT / 'b1-deploy-preview-20261005-v1'
INITIAL = ROOT / 'b1-cartoon-budget-20261004/final-2000/b1_cartoon_fp32.pt'


@torch.inference_mode()
def prepare():
    from .publish_global_blocks import load, bindings
    from .publish_coarse_people import probability
    from .coarse_people import refined_mask, composite
    from .coarse_people_blocks import person_surface
    from .gentle_block_boundaries import process
    from .refine_block_separation import PARAMETERS
    cfg, net, b, d, p, adapter, clip, state, seg = load()
    OUT.mkdir(exist_ok=False)
    (OUT / 'targets').mkdir()
    old = json.loads((ROOT / 'b1-cartoon-budget-20261004/data_manifest.json').read_text(encoding='utf-8'))
    groups = {}
    for row in old['pairs']:
        assert row['split'] == 'train'
        groups.setdefault(row['source'], []).append(row)
    selected = []
    for rows in groups.values():
        selected.extend(rows[i] for i in sorted({0, len(rows)//2, len(rows)-1}))
    sources = {r[k] for r in selected for k in ('current', 'next')}
    sources.update(r['source'] for r in old['validation'])
    outputs = {}
    for i, source in enumerate(sorted(sources)):
        started = time.perf_counter()
        x = read_image(source)
        assert x.shape == (720, 1280, 3)
        pred = to_rgb(net(as_tensor(x)[None].cuda())[0])
        surface = d.surface(p.tone(pred, adapter), d.PRESETS['direction4'])
        background = b.transform_surface(surface, 'balanced')
        mask = refined_mask(probability(seg, x), x)
        people = person_surface(pred, adapter, p, 'strongblocks', x)
        target = process(composite(background, people, mask), PARAMETERS)
        path = OUT / 'targets' / (file_hash(source)[:20] + '.png')
        Image.fromarray(target).save(path)
        outputs[source] = dict(target=str(path), target_sha256=file_hash(path), source_sha256=file_hash(source))
        print('TARGET', i+1, len(sources), round(time.perf_counter()-started, 2), flush=True)
    pairs = []
    for row in selected:
        new = copy.deepcopy(row)
        for key in ('current', 'next'):
            assert file_hash(row[key]) == row[key+'_sha256']
            new[key+'_target'] = outputs[row[key]]['target']
            new[key+'_target_sha256'] = outputs[row[key]]['target_sha256']
        assert file_hash(row['flow']) == row['flow_sha256']
        pairs.append(new)
    validation = [dict(id=r['id'], source=r['source'], **outputs[r['source']]) for r in old['validation']]
    assert not {r['source'] for r in validation} & {r[k] for r in pairs for k in ('current','next')}
    data = dict(pairs=pairs, validation=validation, source_splits=old['source_splits'],
                parent_sha256=file_hash(ROOT/'b1-cartoon-budget-20261004/data_manifest.json'),
                target='Frozen gentle-v3 full spatial recipe, independently generated at native720; no RGB temporal blend',
                previously_reviewed_validation=True)
    write_json(OUT/'data.json', data)
    config = dict(steps=2000, lr=.0005, lr_floor=.00005, crop=320, margin=40,
                  seed=2026100501, initial=str(INITIAL), initial_sha256=file_hash(INITIAL),
                  graph=contract_guard()[1]['model_id'], full_frame_interval=100,
                  loss=dict(rgb=1., planes=1.5, gradients=2., flat=.8, temporal=.25, phase=.3),
                  inference='Unchanged B1 only; all visual recipe operators discarded after target generation')
    write_json(OUT/'config.json', config)
    files = bindings(cfg, state)
    for name in ('train_b1_deploy_preview.py','gentle_block_boundaries.py','refine_block_separation.py','train_b1_distillation.py'):
        path = Path(__file__).with_name(name); files[str(path)] = file_hash(path)
    for name in ('contract_torch.py','model_b_contract.json'):
        path = Path(__file__).parents[1]/name; files[str(path)] = file_hash(path)
    snap=OUT/'source_code';snap.mkdir()
    for i,(path,h) in enumerate(files.items()):
        shutil.copy2(path,snap/f'{i:02d}_{Path(path).name}')
    write_json(OUT/'binding.json',dict(code=files,config_sha256=file_hash(OUT/'config.json'),data_sha256=file_hash(OUT/'data.json')))
    print('TARGETS_READY',len(pairs),'pairs',len(validation),'validation',flush=True)


def setup():
    configure_workspace();set_reproducible(2026100501);cv2.setNumThreads(2);contract_guard()
    cfg=json.loads((OUT/'config.json').read_text());data=json.loads((OUT/'data.json').read_text(encoding='utf-8'))
    binding=json.loads((OUT/'binding.json').read_text())
    assert all(file_hash(p)==h for p,h in binding['code'].items())
    assert file_hash(OUT/'config.json')==binding['config_sha256'] and file_hash(OUT/'data.json')==binding['data_sha256']
    assert file_hash(INITIAL)==cfg['initial_sha256']
    for row in data['pairs']:
        for key in ('current','next','current_target','next_target','flow'):assert file_hash(row[key])==row[key+'_sha256']
    for row in data['validation']:
        for key in ('source','target'):assert file_hash(row[key])==row[key+'_sha256']
    return cfg,data,binding


def gradients(x):
    return x[:,:,:,1:]-x[:,:,:,:-1], x[:,:,1:,:]-x[:,:,:-1,:]


class Training:
    def __init__(self,cfg,data,binding):
        self.cfg,self.data,self.binding=cfg,data,binding
        self.net=build_fp32_model().cuda().eval()
        self.net.load_state_dict(torch.load(INITIAL,map_location='cpu',weights_only=True)['model'])
        self.opt=torch.optim.Adam(self.net.parameters(),lr=cfg['lr'])
        self.rng=random.Random(cfg['seed']);self.step=0

    def update(self,native=False):
        start=time.perf_counter();r=self.rng.choice(self.data['pairs']);size=self.cfg['crop']
        x0=self.rng.randrange(0,1280-size+1)//4*4;y0=self.rng.randrange(0,720-size+1)//4*4
        if native:x0=y0=0;h,w=720,1280
        else:h=w=size
        crop=lambda a:a[y0:y0+h,x0:x0+w]
        xx=torch.stack([as_tensor(crop(read_image(r[k]))) for k in ('current','next')]).cuda()
        tt=torch.stack([as_tensor(crop(read_image(r[k]))) for k in ('current_target','next_target')]).cuda()
        pp=self.net(xx);m=self.cfg['margin'];cut=lambda v:v[:,:,m:-m,m:-m]
        p,t=cut(pp),cut(tt)
        values=dict(rgb=F.l1_loss(p,t),planes=F.l1_loss(F.avg_pool2d(p,8),F.avg_pool2d(t,8)))
        losses=[];flat=[]
        for scale in (1,2,4):
            a,b=(p,t) if scale==1 else (F.avg_pool2d(p,scale),F.avg_pool2d(t,scale))
            for gp,gt in zip(gradients(a),gradients(b)):
                losses.append((gp-gt).abs().mean())
                smooth=(gt.abs()<.01).float()
                flat.append((F.relu(gp.abs()-gt.abs()-.002)*smooth).sum()/smooth.sum().clamp_min(1))
        values['gradients']=torch.stack(losses).mean();values['flat']=torch.stack(flat).mean()
        flow,mask=read_flow(r['flow']);f=crop(flow);valid=crop(mask).copy()
        yy,xxgrid=np.mgrid[:h,:w];valid*=((xxgrid+f[:,:,0]>=m)&(xxgrid+f[:,:,0]<w-m)&(yy+f[:,:,1]>=m)&(yy+f[:,:,1]<h-m))
        f=torch.from_numpy(np.ascontiguousarray(f)).permute(2,0,1)[None].cuda();valid=torch.from_numpy(valid)[None,None].cuda()
        delta=(pp[1:]-warp(pp[:1],f))-(tt[1:]-warp(tt[:1],f))
        values['temporal']=(cut(delta).abs()*cut(valid)).sum()/(3*cut(valid).sum()+1e-6)
        sx,sy=self.rng.choice([(1,0),(2,0),(3,0),(0,1),(1,1)])
        shifted=F.pad(xx[:1],(sx,0,sy,0),mode='replicate')[:,:,:h,:w]
        ps=self.net(shifted)
        values['phase']=F.l1_loss(ps[:,:,m+sy:-m,m+sx:-m],pp[:1,:,m:-m-sy if sy else -m,m:-m-sx if sx else -m])
        loss=sum(values[k]*weight for k,weight in self.cfg['loss'].items());assert torch.isfinite(loss)
        lr=self.cfg['lr_floor']+(self.cfg['lr']-self.cfg['lr_floor'])*.5*(1+math.cos(math.pi*self.step/self.cfg['steps']))
        for group in self.opt.param_groups:group['lr']=lr
        self.opt.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.net.parameters(),3.,error_if_nonfinite=True);assert norm>0
        self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),values={k:float(v.detach()) for k,v in values.items()},
                    sample=r['current'],crop=[x0,y0,w,h],phase=[sx,sy],native=native,grad=float(norm),lr=lr,
                    seconds=time.perf_counter()-start,peak_mib=torch.cuda.max_memory_allocated()/2**20)

    def save(self,path):
        torch.save(dict(model=self.net.state_dict(),optimizer=self.opt.state_dict(),step=self.step,
                        random=self.rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),binding=self.binding),path)

    def restore(self,path):
        s=torch.load(path,map_location='cpu',weights_only=True);assert s['binding']==self.binding
        self.net.load_state_dict(s['model']);self.opt.load_state_dict(s['optimizer']);self.step=s['step'];self.rng.setstate(s['random'])
        torch.set_rng_state(s['torch_rng']);torch.cuda.set_rng_state_all(s['cuda_rng'])


@torch.inference_mode()
def evaluate(net,data,name):
    out=OUT/name;out.mkdir(exist_ok=False);rows=[]
    for r in data['validation']:
        x=as_tensor(read_image(r['source']))[None].cuda();y=net(x);rgb=to_rgb(y[0]);Image.fromarray(rgb).save(out/(r['id']+'.png'))
        t=read_image(r['target']);rows.append(dict(id=r['id'],mae_target=float(np.abs(rgb.astype(float)-t).mean())))
    write_json(out/'scores.json',rows)


def train(preflight=False):
    cfg,data,binding=setup();tr=Training(cfg,data,binding)
    if preflight:
        d=OUT/'preflight';d.mkdir(exist_ok=False)
        a=tr.update(True);b=tr.update(False);tr.save(d/'resume.pt');ref=tr.update(False)
        expected={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};del tr;gc.collect();torch.cuda.empty_cache()
        tr=Training(cfg,data,binding);tr.restore(d/'resume.pt');got=tr.update(False)
        error=max(float((v.cpu()-expected[k]).abs().max()) for k,v in tr.net.state_dict().items() if v.is_floating_point())
        assert error<2e-6 and ref['sample']==got['sample'] and ref['crop']==got['crop'] and ref['phase']==got['phase']
        write_json(d/'result.json',dict(status='PASS',native=a,crop=b,resume_max_error=error,binding=binding))
        print('PREFLIGHT_PASS',error,flush=True);return
    pre=json.loads((OUT/'preflight/result.json').read_text());assert pre['status']=='PASS' and pre['binding']==binding
    evaluate(tr.net,data,'fp32_000000')
    with (OUT/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        while tr.step<cfg['steps']:
            row=tr.update((tr.step+1)%cfg['full_frame_interval']==0);log.write(json.dumps(row)+'\n')
            if tr.step%25==0:write_json(OUT/'progress.json',row);print('TRAIN_B1',tr.step,round(row['loss'],4),round(row['seconds'],3),flush=True)
            if tr.step%500==0:
                tr.save(OUT/f'checkpoint_{tr.step:06d}.pt');evaluate(tr.net,data,f'fp32_{tr.step:06d}')
    print('B1_FP32_COMPLETE',tr.step,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','preflight','train']);a=parser.parse_args()
    if a.mode=='prepare':prepare()
    else:train(a.mode=='preflight')
