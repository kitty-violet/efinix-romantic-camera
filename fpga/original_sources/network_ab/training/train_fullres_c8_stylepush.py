"""Aggressive material distillation, isolated from B1 and the previous C8 run."""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from .train_fullres_c8_multiscene import Training as PreviousTraining, code_paths
from .probe_fullres_c8 import FullresC8
from .train_b1_distillation import as_tensor, to_rgb, read_image, edges
from .coherent_shared_model import warp
from .losses import gram
from .engine import set_reproducible
from .runtime import configure_workspace, contract_guard, file_hash, write_json

RUN = 'fullres-c8-stylepush-color-20261003'
PREVIOUS = 'fullres-c8-multiscene-20261003'
INITIAL_SHA = '31591006cba3c3e5a6751191246024bce3eb3d227dcb30a841970b54e71827fc'


def lowpass(x, size=9):
    return F.avg_pool2d(F.pad(x, (size//2,)*4, mode='replicate'), size, 1)


def strong_target(source, teacher, gain):
    # Amplify local material changes, not the broad color/exposure shift.
    difference = teacher-source
    extra = ((gain-1)*(difference-lowpass(difference))).clamp(-.20, .20)
    return (teacher+extra).clamp(-1, 1)


class Training(PreviousTraining):
    def __init__(self, root, config, manifest):
        super().__init__(root, config, manifest)
        assert file_hash(config['initial_weights']) == INITIAL_SHA
        self.net.load_state_dict(torch.load(config['initial_weights'], weights_only=True)['model'])

    def terms(self, pred, source, target, flow, mask, native):
        m=self.config['margin'];cut=lambda a:a[:,:,m:-m,m:-m]
        desired=strong_target(source,target,self.config['material_gain'])
        p,t,x=cut(pred),cut(desired),cut(source)
        # RGB is intentionally weak. Normalize material bands by the actual
        # desired change so that near-identity regions cannot dominate the loss.
        bands=[]
        for size in (3,9):
            bp,bt,bx=[v-lowpass(v,size) for v in (p,t,x)]
            scale=(bt-bx).square().mean((1,2,3),keepdim=True).sqrt().clamp_min(.02)
            focus=1+((bt-bx).abs().mean(1,keepdim=True)/.04).clamp(0,7)
            bands.append((((bp-bt).abs()/scale)*focus).sum()/(3*focus.sum()))
        edge_terms=[]
        for ep,et,ex in zip(edges(p),edges(t),edges(x)):
            focus=1+((et-ex).abs().mean(1,keepdim=True)/.025).clamp(0,7)
            edge_terms.append(((ep-et).abs()*focus).sum()/(3*focus.sum()))
        pv,tv=(F.avg_pool2d(p,2),F.avg_pool2d(t,2)) if native else (p,t)
        with torch.no_grad(): tf=self.vgg.extract(tv)
        pf=self.vgg.extract(pv)
        style=[]
        for name in ('relu1_2','relu2_2'):
            gt=gram(tf[name]);gp=gram(pf[name])
            style.append(((gp-gt).abs()/(gt.abs().mean((1,2),keepdim=True)+.01)).mean())
        n=len(pred)//2
        delta=(pred[n:]-source[n:])-warp(pred[:n]-source[:n],flow)
        valid=cut(mask)
        return dict(rgb=F.l1_loss(p,t),color=F.l1_loss(F.avg_pool2d(p,16),F.avg_pool2d(cut(target),16)),material=torch.stack(bands).mean(),
            edge=sum(edge_terms),perceptual=F.l1_loss(pf['relu2_2'],tf['relu2_2']),
            gram=torch.stack(style).mean(),
            temporal=(cut(delta).abs()*valid).sum()/(3*valid.sum()+1e-6))

    def update(self,native=None):
        begin=time.perf_counter();self.net.eval()
        if native is None:native=(self.step+1)%self.config['native_interval']==0
        a,b,ta,tb,flow,mask,spec=self.batch(native)
        source=torch.cat([a,b]);target=torch.cat([ta,tb]);pred=self.net(source)
        terms=self.terms(pred,source,target,flow,mask,native)
        loss=sum(terms[k]*w for k,w in self.config['weights'].items())
        assert torch.isfinite(loss)
        progress=self.step/(self.config['max_steps']-1)
        lr=(self.config['lr_floor']+(self.config['lr']-self.config['lr_floor'])*.5*(1+math.cos(math.pi*progress)))*min(1,(self.step+1)/25)
        for group in self.opt.param_groups:group['lr']=lr
        self.opt.zero_grad(set_to_none=True);loss.backward()
        for name,param in self.net.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(),name
        norm=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.,error_if_nonfinite=True)
        assert norm>0
        self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),grad_norm=float(norm),lr=lr,
            **{k:float(v.detach()) for k,v in terms.items()},native=native,samples=spec,
            seconds=time.perf_counter()-begin,peak_mib=torch.cuda.max_memory_allocated()/2**20)


def prepare(root,run):
    run.mkdir(exist_ok=False);previous=root/'runs'/PREVIOUS
    manifest=json.loads((previous/'data_manifest.json').read_text(encoding='utf-8'))
    checked={}
    for row in manifest['pairs']:
        for key in ('current','next','current_target','next_target','flow'):
            checked[row[key]]=row[key+'_sha256']
    for row in manifest['validation']:
        for key in ('source','target'):checked[row[key]]=row[key+'_sha256']
    for p,digest in checked.items():assert file_hash(p)==digest,p
    train={r['source'] for r in manifest['pairs']}
    heldout={p for p,v in manifest['source_splits'].items() if v['split']!='train'}
    assert len(train)==7 and train.isdisjoint(heldout)
    shutil.copy2(previous/'data_manifest.json',run/'data_manifest.json')
    initial=previous/'final/fullres_c8_fp32.pt';assert file_hash(initial)==INITIAL_SHA
    shutil.copy2(initial,run/'initial_fp32.pt')
    cfg=dict(seed=202610037,max_steps=2000,lr=.0015,lr_floor=.00015,pairs_per_step=2,
        crop=400,margin=16,native_interval=50,material_gain=1.8,
        weights=dict(rgb=.25,color=5.,material=.6,edge=8.,perceptual=.12,gram=.2,temporal=.05),
        initial_weights=str(run/'initial_fp32.pt'),initial_sha256=INITIAL_SHA,
        precision='FP32, no AMP/TF32',model='unchanged FullresC8',ema=False,
        scope='Aggressive material target and objective, no renderer postprocessing or hardware edits')
    write_json(run/'config.json',cfg)
    paths=list(dict.fromkeys(code_paths()+[Path(__file__)]))
    source=run/'source';source.mkdir()
    for p in paths:shutil.copy2(p,source/p.name)
    binding=dict(config_sha256=file_hash(run/'config.json'),manifest_sha256=file_hash(run/'data_manifest.json'),
        initial_sha256=INITIAL_SHA,code={str(p):file_hash(p) for p in paths},
        shared_contract_sha256=file_hash(Path(__file__).parents[1]/'model_b_contract.json'))
    write_json(run/'binding.json',binding)
    write_json(run/'data_check.json',dict(status='PASS',files=len(checked),pairs=95,sources=7,source_isolation=True,
        previous_reviewed_validation_not_blind=True))
    print('PREPARED',run,flush=True)


def validate(net,manifest,path,cfg,root):
    path.mkdir(parents=True,exist_ok=False);rows=[]
    with torch.inference_mode():
        for row in manifest['validation']:
            a=read_image(row['source']);t=read_image(row['target'])
            x=as_tensor(a)[None].cuda();teacher=as_tensor(t)[None].cuda()
            p=net(x);goal=strong_target(x,teacher,cfg['material_gain']);image=to_rgb(p[0])
            Image.fromarray(image).save(path/(row['id']+'.png'))
            hp=lambda v:v-lowpass(v)
            target_delta=hp(teacher)-hp(x);change=hp(p)-hp(x)
            projection=float((change*target_delta).sum()/(target_delta.square().sum()+1e-6))
            cosine=float((change*target_delta).sum()/(change.square().sum().sqrt()*target_delta.square().sum().sqrt()+1e-6))
            rows.append(dict(id=row['id'],teacher_mae=float((p-teacher).abs().mean()*128),
                source_change=float((p-x).abs().mean()*128),goal_mae=float((p-goal).abs().mean()*128),
                material_projection=projection,material_cosine=cosine,
                newly_white_fraction=float(((image.min(2)>245)&(a.max(2)<220)).mean())))
            if row['id'] in ('val_stairs_090','val_plaza_090','val_new_hallway_120'):
                old=read_image(str(root/'runs'/PREVIOUS/'train/validation_002000'/(row['id']+'.png')))
                xy=(424,320) if 'stairs' in row['id'] else (80,400) if 'plaza' in row['id'] else (560,150)
                panel=Image.new('RGB',(1600,350),(24,24,24));draw=ImageDraw.Draw(panel)
                for i,(v,label) in enumerate([(a,'Input'),(t,'Teacher'),(old,'Previous C8'),(image,'Style-push C8'),(to_rgb(goal[0]),'Training target 1.8')]):
                    xx,yy=xy;panel.paste(Image.fromarray(v[yy:yy+320,xx:xx+320]),(320*i,30));draw.text((320*i+6,8),label,fill='white')
                panel.save(path/(row['id']+'_comparison.png'))
    write_json(path/'scores.json',dict(records=rows,reviewed_not_blind=True))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','preflight','train'])
    ap.add_argument('--resume',type=Path);ap.add_argument('--stop-after',type=int,default=2000);args=ap.parse_args()
    root=configure_workspace();contract_guard();cv2.setNumThreads(2);run=root/'runs'/RUN
    if args.mode=='prepare':prepare(root,run);return
    cfg=json.loads((run/'config.json').read_text());binding=json.loads((run/'binding.json').read_text())
    for key,p in [('config_sha256','config.json'),('manifest_sha256','data_manifest.json')]:assert file_hash(run/p)==binding[key]
    for p,digest in binding['code'].items():assert file_hash(p)==digest,p
    manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'))
    set_reproducible(cfg['seed']);trainer=Training(root,cfg,manifest)
    if args.mode=='preflight':
        out=run/'preflight';out.mkdir(exist_ok=False)
        # Measure the weighted output-gradient contributions on a real train pair.
        a,b,ta,tb,flow,mask,spec=trainer.batch(False)
        source=torch.cat([a,b]);target=torch.cat([ta,tb]);pred=trainer.net(source).detach().requires_grad_(True)
        terms=trainer.terms(pred,source,target,flow,mask,False)
        grad={k:float(torch.autograd.grad(v*cfg['weights'][k],pred,retain_graph=True)[0].norm()) for k,v in terms.items()}
        del pred,terms;gc.collect()
        tests=[trainer.update(True),trainer.update(False)]
        torch.save(trainer.state(binding),out/'resume.pt');reference=trainer.update(False)
        expected={k:v.detach().cpu().clone() for k,v in trainer.net.state_dict().items()}
        del trainer;gc.collect();torch.cuda.empty_cache();trainer=Training(root,cfg,manifest)
        trainer.restore(torch.load(out/'resume.pt',weights_only=True,map_location='cpu'),binding)
        replay=trainer.update(False)
        error=max(float((v.cpu()-expected[k]).abs().max()) for k,v in trainer.net.state_dict().items())
        assert error<2e-6 and reference['samples']==replay['samples']
        assert all(p.dtype==torch.float32 for p in trainer.net.parameters())
        write_json(out/'result.json',dict(status='PASS',binding=binding,tests=tests,replay_error=error,weighted_output_gradient=grad))
        print('PREFLIGHT_PASS',error,grad,flush=True);return
    pre=json.loads((run/'preflight/result.json').read_text());assert pre['status']=='PASS' and pre['binding']==binding
    out=run/'train';out.mkdir(exist_ok=True)
    if (out/'metrics.jsonl').exists() and not args.resume:raise FileExistsError('Explicit resume required')
    if args.resume:
        trainer.restore(torch.load(args.resume,map_location='cpu',weights_only=True),binding)
        assert [json.loads(s)['step'] for s in (out/'metrics.jsonl').read_text().splitlines()]==list(range(1,trainer.step+1))
    else:validate(trainer.net,manifest,out/'validation_000000',cfg,root)
    write_json(out/f'launch_{trainer.step:06d}.json',dict(initial_step=trainer.step,resume_from=str(args.resume),binding=binding))
    with (out/'metrics.jsonl').open('a',encoding='utf-8',buffering=1) as log:
        while trainer.step<min(args.stop_after,cfg['max_steps']):
            row=trainer.update();log.write(json.dumps(row,ensure_ascii=False)+'\n')
            if trainer.step%25==0:
                write_json(out/'progress.json',row)
                print(json.dumps({k:row[k] for k in ('step','loss','material','edge','temporal','seconds','peak_mib')}),flush=True)
            if trainer.step%250==0:
                torch.save(trainer.state(binding),out/f'checkpoint_{trainer.step:06d}.pt')
                validate(trainer.net,manifest,out/f'validation_{trainer.step:06d}',cfg,root)
    if trainer.step%250:torch.save(trainer.state(binding),out/f'checkpoint_{trainer.step:06d}.pt')
    write_json(out/'status.json',dict(status='COMPLETE' if trainer.step==cfg['max_steps'] else 'BOUNDED_SEGMENT_COMPLETE',step=trainer.step))
    if trainer.step==cfg['max_steps']:
        final=run/'final';final.mkdir(exist_ok=False);p=final/'fullres_c8_fp32.pt'
        torch.save(dict(schema='fullres-c8-stylepush-fp32-v1',model={k:v.detach().cpu() for k,v in trainer.net.state_dict().items()},
            steps=trainer.step,binding=binding,production_candidate=False),p)
        loaded=FullresC8().cuda().eval();loaded.load_state_dict(torch.load(p,weights_only=True)['model'])
        with torch.inference_mode():
            x=as_tensor(read_image(manifest['validation'][0]['source']))[None].cuda();error=float((loaded(x)-trainer.net(x)).abs().max())
        assert error==0
        write_json(final/'export.json',dict(status='FP32_RELOAD_PASS_NOT_INT8',weights=str(p),sha256=file_hash(p),reload_error=error))
        print('EXPORTED',p,flush=True)


if __name__=='__main__':main()
