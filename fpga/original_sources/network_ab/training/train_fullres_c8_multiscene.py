"""Multisource FP32 training of the isolated FullresC8 research graph."""
import argparse
import gc
import json
import math
from pathlib import Path
import random
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from .probe_fullres_c8 import FullresC8
from .train_b1_distillation import Training as OldSampler, as_tensor, to_rgb, read_image, edges
from .coherent_shared_model import warp
from .engine import set_reproducible
from .losses import PerceptualLoss
from .runtime import configure_workspace, contract_guard, file_hash, write_json

RUN='fullres-c8-multiscene-20261003'


class Training:
    batch=OldSampler.batch

    def __init__(self,root,cfg,manifest):
        self.config=cfg;self.manifest=manifest;self.step=0
        self.rng=random.Random(cfg['seed']);self.groups={}
        for r in manifest['pairs']:
            assert manifest['source_splits'][r['source']]['split']=='train'
            self.groups.setdefault(r['source'],[]).append(r)
        torch.manual_seed(cfg['seed'])
        self.net=FullresC8().cuda().initialize_identity()
        g=torch.Generator(device='cuda').manual_seed(cfg['seed']+100)
        with torch.no_grad():
            for m in self.net.modules():
                if isinstance(m,torch.nn.Conv2d):
                    m.weight.add_(torch.randn(m.weight.shape,generator=g,device='cuda')*(.02/math.sqrt(m.weight[0].numel())))
        self.opt=torch.optim.Adam(self.net.parameters(),lr=cfg['lr'])
        self.vgg=PerceptualLoss(root).cuda();self.vgg.features=self.vgg.features[:9]

    def update(self,native=None):
        self.net.eval();begin=time.perf_counter()
        if native is None:native=(self.step+1)%self.config['native_interval']==0
        a,b,ta,tb,flow,mask,spec=self.batch(native)
        n=len(a);source=torch.cat([a,b]);target=torch.cat([ta,tb]);pred=self.net(source)
        margin=self.config['margin'];cut=lambda v:v[:,:,margin:-margin,margin:-margin]
        p,t,x=cut(pred),cut(target),cut(source)
        # Bounded emphasis on target changes; includes unchanged pixels with weight 1.
        focus=(t-x).abs().mean(1,keepdim=True).div(.08).clamp(0,2)+1
        hp=lambda v:v-F.avg_pool2d(v,5,1,2)
        pv,tv=(F.avg_pool2d(p,2),F.avg_pool2d(t,2)) if native else (p,t)
        with torch.no_grad():tf=self.vgg.extract(tv)['relu2_2']
        pf=self.vgg.extract(pv)['relu2_2']
        delta=(pred[n:]-b)-warp(pred[:n]-a,flow);valid=cut(mask)
        terms=dict(paired=((p-t).abs()*focus).sum()/(3*focus.sum()),
                   edge=sum(F.l1_loss(u,v) for u,v in zip(edges(p),edges(t))),
                   detail=F.l1_loss(hp(p),hp(t)),perceptual=F.l1_loss(pf,tf),
                   temporal=(cut(delta).abs()*valid).sum()/(valid.sum()*3+1e-6))
        loss=sum(terms[k]*weight for k,weight in self.config['weights'].items())
        assert torch.isfinite(loss)
        progress=self.step/(self.config['max_steps']-1)
        lr=(self.config['lr_floor']+(self.config['lr']-self.config['lr_floor'])*.5*(1+math.cos(math.pi*progress)))*min(1,(self.step+1)/25)
        for group in self.opt.param_groups:group['lr']=lr
        self.opt.zero_grad(set_to_none=True);loss.backward()
        for name,param in self.net.named_parameters():
            if param.grad is None or not torch.isfinite(param.grad).all():raise FloatingPointError(name)
        norm=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.,error_if_nonfinite=True)
        assert norm>0
        self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),grad_norm=float(norm),lr=lr,
                    **{k:float(v.detach()) for k,v in terms.items()},native=native,samples=spec,
                    seconds=time.perf_counter()-begin,peak_mib=torch.cuda.max_memory_allocated()/2**20)

    def state(self,binding):
        return dict(schema='fullres-c8-multiscene-checkpoint-v1',model=self.net.state_dict(),optimizer=self.opt.state_dict(),
                    step=self.step,config=self.config,binding=binding,sampler_rng=self.rng.getstate(),
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())

    def restore(self,state,binding):
        assert state['schema']=='fullres-c8-multiscene-checkpoint-v1' and state['config']==self.config and state['binding']==binding
        self.net.load_state_dict(state['model']);self.opt.load_state_dict(state['optimizer']);self.step=state['step']
        self.rng.setstate(state['sampler_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])


def code_paths():
    here=Path(__file__).parent
    return [here/n for n in ('train_fullres_c8_multiscene.py','probe_fullres_c8.py','train_b1_distillation.py',
                             'coherent_shared_model.py','engine.py','losses.py','runtime.py')]


def prepare(root,run):
    run.mkdir(exist_ok=False)
    original=root/'runs/b1-distillation-20261003/data/manifest.json'
    manifest=json.loads(original.read_text(encoding='utf-8'));assert len(manifest['pairs'])==95
    hashes={}
    for row in manifest['pairs']:
        for key in ('current','next','current_target','next_target','flow'):
            hashes[row[key]]=row[key+'_sha256']
    for row in manifest['validation']:
        for key in ('source','target'):hashes[row[key]]=row[key+'_sha256']
    for p,digest in hashes.items():assert file_hash(p)==digest,p
    train_ids={r['source'] for r in manifest['pairs']}
    heldout={k for k,v in manifest['source_splits'].items() if v['split']!='train'}
    assert train_ids.isdisjoint(heldout) and len(train_ids)==7
    shutil.copy2(original,run/'data_manifest.json')
    cfg=dict(seed=202610035,max_steps=2000,lr=.002,lr_floor=.0002,pairs_per_step=2,crop=400,margin=16,
             native_interval=50,weights=dict(paired=1.,edge=2.,detail=1.,perceptual=.02,temporal=.1),
             precision='FP32 without AMP/TF32',model='FullresC8 784 raw convolution weights, 931 parameters',
             initialization='fresh signed RGB identity plus deterministic .02/sqrt(fan_in) perturbation; no old patch weights',
             bn='frozen running statistics; train affine',ema=False,
             paired_focus='1 + clamp(meanRGB(abs(teacher-source))/.08,0,2); normalized weighted L1',
             scope='multisource research; no production graph/RTL/INT8 change; no blind-validation claim')
    write_json(run/'config.json',cfg)
    source=run/'source';source.mkdir()
    for p in code_paths():shutil.copy2(p,source/p.name)
    binding=dict(config_sha256=file_hash(run/'config.json'),manifest_sha256=file_hash(run/'data_manifest.json'),
                 code={str(p):file_hash(p) for p in code_paths()},
                 shared_contract_sha256=file_hash(Path(__file__).parents[1]/'model_b_contract.json'),
                 vgg_manifest_sha256=file_hash(root/'runs/environment/vgg19-weights.json'))
    write_json(run/'binding.json',binding)
    # Review actual training targets at modest size; native files remain untouched.
    selected=[]
    for source_name in sorted(train_ids):
        group=[r for r in manifest['pairs'] if r['source']==source_name]
        selected.extend([group[0],group[len(group)//2]])
    board=Image.new('RGB',(960,204*7),(24,24,24));draw=ImageDraw.Draw(board)
    for i,r in enumerate(selected):
        x=(i%2)*480;y=(i//2)*204
        for j,key in enumerate(('current','current_target')):
            board.paste(Image.open(r[key]).resize((240,180)),(x+j*240,y+24))
        draw.text((x+4,y+5),f'{Path(r["source"]).stem} f{r["current_index"]} / target',fill='white')
    board.save(run/'training_pairs_review.jpg',quality=94)
    write_json(run/'data_check.json',dict(status='PASS',training_sources=7,pairs=95,unique_files_checked=len(hashes),
                source_isolation=True,fresh_start=True,review='existing reviewed teacher pairs, source-balanced sampler',
                original_manifest=str(original),original_manifest_sha256=file_hash(original)))
    print('PREPARED',run,flush=True)


def validate(net,manifest,path):
    path.mkdir(exist_ok=False,parents=True);rows=[]
    with torch.inference_mode():
        for r in manifest['validation']:
            source=read_image(r['source']);teacher=read_image(r['target'])
            x=as_tensor(source)[None].cuda();t=as_tensor(teacher)[None].cuda();p=net(x)
            image=to_rgb(p[0]);Image.fromarray(image).save(path/(r['id']+'.png'))
            rows.append(dict(id=r['id'],teacher_mae=float((p-t).abs().mean()*128),
                source_change=float((p-x).abs().mean()*128),teacher_change=float((t-x).abs().mean()*128),
                gradient_teacher_error=sum(float((u-v).abs().mean()*128) for u,v in zip(edges(p),edges(t)))/2,
                newly_bright_fraction=float(((image.min(2)>245)&(source.max(2)<220)).mean())))
    write_json(path/'scores.json',dict(records=rows,reviewed_not_blind=True))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','preflight','train'])
    ap.add_argument('--resume',type=Path);ap.add_argument('--stop-after',type=int,default=2000);args=ap.parse_args()
    root=configure_workspace();run=root/'runs'/RUN;contract_guard();cv2.setNumThreads(2)
    if args.mode=='prepare':prepare(root,run);return
    cfg=json.loads((run/'config.json').read_text());binding=json.loads((run/'binding.json').read_text())
    assert file_hash(run/'config.json')==binding['config_sha256'] and file_hash(run/'data_manifest.json')==binding['manifest_sha256']
    for p,digest in binding['code'].items():assert file_hash(p)==digest,p
    manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'))
    set_reproducible(cfg['seed']);trainer=Training(root,cfg,manifest)
    if args.mode=='preflight':
        out=run/'preflight';out.mkdir(exist_ok=False)
        tests=[trainer.update(native=True),trainer.update(native=False)]
        torch.save(trainer.state(binding),out/'resume.pt')
        reference=trainer.update(native=False);expected={k:v.detach().cpu().clone() for k,v in trainer.net.state_dict().items()}
        del trainer;gc.collect();torch.cuda.empty_cache()
        trainer=Training(root,cfg,manifest);trainer.restore(torch.load(out/'resume.pt',map_location='cpu',weights_only=True),binding)
        replay=trainer.update(native=False)
        error=max(float((v.cpu()-expected[k]).abs().max()) for k,v in trainer.net.state_dict().items())
        assert error<2e-6 and reference['samples']==replay['samples']
        assert all(p.dtype==torch.float32 for p in trainer.net.parameters())
        write_json(out/'result.json',dict(status='PASS',tests=tests,replay_error=error,reference=reference,replay=replay,binding=binding))
        print('PREFLIGHT_PASS',error,flush=True);return
    pre=json.loads((run/'preflight/result.json').read_text());assert pre['status']=='PASS' and pre['binding']==binding
    out=run/'train';out.mkdir(exist_ok=True)
    if (out/'metrics.jsonl').exists() and not args.resume:raise FileExistsError('Explicit resume required')
    if args.resume:
        trainer.restore(torch.load(args.resume,map_location='cpu',weights_only=True),binding)
        logged=[json.loads(line)['step'] for line in (out/'metrics.jsonl').read_text().splitlines()]
        assert logged==list(range(1,trainer.step+1))
    else:validate(trainer.net,manifest,out/'validation_000000')
    write_json(out/f'launch_{trainer.step:06d}.json',dict(resume_from=str(args.resume),initial_step=trainer.step,binding=binding))
    with (out/'metrics.jsonl').open('a',encoding='utf-8',buffering=1) as log:
        while trainer.step<min(args.stop_after,cfg['max_steps']):
            row=trainer.update();log.write(json.dumps(row,ensure_ascii=False)+'\n')
            if trainer.step%25==0:
                write_json(out/'progress.json',row)
                print(json.dumps({k:row[k] for k in ('step','loss','paired','edge','temporal','seconds','peak_mib')}),flush=True)
            if trainer.step%500==0:
                torch.save(trainer.state(binding),out/f'checkpoint_{trainer.step:06d}.pt')
                validate(trainer.net,manifest,out/f'validation_{trainer.step:06d}')
    if trainer.step%500:torch.save(trainer.state(binding),out/f'checkpoint_{trainer.step:06d}.pt')
    write_json(out/'status.json',dict(status='COMPLETE' if trainer.step==cfg['max_steps'] else 'BOUNDED_SEGMENT_COMPLETE',step=trainer.step))
    if trainer.step==cfg['max_steps']:
        final=run/'final';final.mkdir(exist_ok=False)
        weight=final/'fullres_c8_fp32.pt'
        torch.save(dict(schema='fullres-c8-multiscene-fp32-v1',model={k:v.detach().cpu() for k,v in trainer.net.state_dict().items()},
                        steps=trainer.step,binding=binding,production_candidate=False),weight)
        reload=FullresC8().cuda().eval();reload.load_state_dict(torch.load(weight,weights_only=True)['model'])
        with torch.inference_mode():
            x=as_tensor(read_image(manifest['validation'][0]['source']))[None].cuda()
            error=float((reload(x)-trainer.net(x)).abs().max())
        assert error==0
        write_json(final/'export.json',dict(status='FP32_RELOAD_PASS_NOT_INT8',weights=str(weight),sha256=file_hash(weight),reload_error=error))
        print('EXPORTED',weight,flush=True)


if __name__=='__main__':main()
