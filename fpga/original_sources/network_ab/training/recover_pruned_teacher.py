"""Native720 FP32 recovery of physically pruned pretrained teacher weights."""
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
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from .prune_animation_teacher import RUN, build_pruned
from .train_b1_distillation import as_tensor, to_rgb, read_image, read_flow, edges
from .coherent_shared_model import warp
from .losses import PerceptualLoss
from .runtime import configure_workspace, file_hash, write_json, contract_guard
from .engine import set_reproducible


class Recovery:
    def __init__(self, root, run, cfg, manifest):
        self.config=cfg; self.step=0; self.rng=random.Random(cfg['seed']);self.groups={}
        for row in manifest['pairs']:
            self.groups.setdefault(row['source'],[]).append(row)
        initial=run/cfg['candidate']/'initial_fp32.pt'
        assert file_hash(initial)==cfg['initial_sha256']
        state=torch.load(initial,weights_only=True,map_location='cpu')
        self.indices=state['indices']
        self.net=build_pruned(root,self.indices).cuda().eval();self.net.load_state_dict(state['generator'])
        self.opt=torch.optim.Adam(self.net.parameters(),lr=cfg['lr'])
        self.vgg=PerceptualLoss(root).cuda();self.vgg.features=self.vgg.features[:9]

    def data(self):
        row=self.rng.choice(self.groups[self.rng.choice(sorted(self.groups))])
        arrays=[as_tensor(read_image(row[k]))[None].cuda() for k in ('current','next','current_target','next_target')]
        flow,mask=read_flow(row['flow'])
        return arrays,torch.from_numpy(flow).permute(2,0,1)[None].cuda(),torch.from_numpy(mask)[None,None].cuda(),row

    def terms(self,p,t):
        p,t=p[:,:,8:-8,8:-8],t[:,:,8:-8,8:-8]
        with torch.no_grad():tf=self.vgg.extract(F.avg_pool2d(t,2))['relu2_2']
        pf=self.vgg.extract(F.avg_pool2d(p,2))['relu2_2']
        return dict(rgb=F.l1_loss(p,t),edge=sum(F.l1_loss(a,b) for a,b in zip(edges(p),edges(t))),
                    perceptual=F.l1_loss(pf,tf),planes=F.l1_loss(F.avg_pool2d(p,16),F.avg_pool2d(t,16)))

    def update(self):
        begin=time.perf_counter();self.net.eval()
        (a,b,ta,tb),flow,mask,row=self.data()
        progress=min(1.,self.step/(self.config['max_steps']-1))
        lr=(self.config['lr_floor']+(self.config['lr']-self.config['lr_floor'])*.5*(1+math.cos(math.pi*progress)))*min(1.,(self.step+1)/10)
        for group in self.opt.param_groups:group['lr']=lr
        self.opt.zero_grad(set_to_none=True)
        # Accumulate two real full720 frames without retaining two full graphs.
        pa=self.net(a);terms_a=self.terms(pa,ta)
        loss_a=sum(v*self.config['weights'][k] for k,v in terms_a.items())*.5
        loss_a.backward();previous=pa.detach();del pa
        pb=self.net(b);terms_b=self.terms(pb,tb)
        delta=(pb-b)-warp(previous-a,flow)
        temporal=(delta.abs()*mask).sum()/(3*mask.sum()+1e-6)
        loss_b=sum(v*self.config['weights'][k] for k,v in terms_b.items())*.5+self.config['temporal']*temporal
        assert torch.isfinite(loss_a) and torch.isfinite(loss_b)
        loss_b.backward()
        for name,p in self.net.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(),name
        norm=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.,error_if_nonfinite=True);assert norm>0
        self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss_a.detach()+loss_b.detach()),
            **{k:float((v.detach()+terms_b[k].detach())*.5) for k,v in terms_a.items()},
            temporal=float(temporal.detach()),grad_norm=float(norm),lr=lr,
            source=row['source'],indices=[row['current_index'],row['next_index']],native_frames=2,
            seconds=time.perf_counter()-begin,peak_mib=torch.cuda.max_memory_allocated()/2**20)

    def state(self,binding):
        return dict(generator=self.net.state_dict(),optimizer=self.opt.state_dict(),step=self.step,indices=self.indices,
                    sampler=self.rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),config=self.config,binding=binding)

    def restore(self,state,binding):
        assert state['binding']==binding and state['config']==self.config and state['indices']==self.indices
        self.net.load_state_dict(state['generator']);self.opt.load_state_dict(state['optimizer']);self.step=state['step']
        self.rng.setstate(state['sampler']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])


def evaluate(net,manifest,out):
    out.mkdir(exist_ok=False);records=[]
    from .b1_fit_controls import score
    with torch.inference_mode():
        for row in manifest['validation']:
            source=read_image(row['source']);target=read_image(row['target'])
            p=to_rgb(net(as_tensor(source)[None].cuda())[0])
            Image.fromarray(p).save(out/(row['id']+'.png'))
            records.append(dict(id=row['id'],teacher=score(p,target),source=score(p,source)))
            if row['id'] not in ('val_stairs_090','val_plaza_090','val_new_hallway_120'):continue
            xx,yy=(424,320) if 'stairs' in row['id'] else (80,400) if 'plaza' in row['id'] else (560,150)
            panel=Image.new('RGB',(960,348),'#191e28');draw=ImageDraw.Draw(panel)
            for i,(v,label) in enumerate([(source,'Source'),(target,'Teacher + accepted guard'),(p,'Pruned, no guard')]):
                panel.paste(Image.fromarray(v[yy:yy+320,xx:xx+320]),(320*i,28));draw.text((320*i+8,8),label,fill='white')
            panel.save(out/(row['id']+'_comparison.png'))
    write_json(out/'scores.json',dict(records=records,student_postprocess=False,teacher_target_has_fixed_surface_guard=True))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','preflight','train']);ap.add_argument('--candidate',choices=['keep50','keep25'],required=True)
    ap.add_argument('--resume',type=Path);ap.add_argument('--stop-after',type=int,default=300);args=ap.parse_args()
    root=configure_workspace();run=root/'runs'/RUN;contract_guard();set_reproducible(202610039);cv2.setNumThreads(2)
    out=run/(args.candidate+'_recovery')
    manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'))
    if args.mode=='prepare':
        out.mkdir(exist_ok=False)
        cfg=dict(candidate=args.candidate,seed=202610039,max_steps=300,lr=.0006,lr_floor=.00015,
                 weights=dict(rgb=1.,edge=2.,perceptual=.1,planes=.5),temporal=.2,
                 initial_sha256=file_hash(run/args.candidate/'initial_fp32.pt'),
                 train_policy='Two complete720 frames accumulated sequentially; no spatial crop, resize, AMP or TF32',
                 target='Existing whole720 teacher with accepted R48 guard; student learns target directly, no inference guard')
        write_json(out/'config.json',cfg)
        paths=[Path(__file__),Path(__file__).with_name('prune_animation_teacher.py'),Path(__file__).with_name('trainable_animegan.py'),
               Path(__file__).with_name('train_b1_distillation.py'),Path(__file__).with_name('coherent_shared_model.py'),Path(__file__).with_name('losses.py')]
        binding=dict(config_sha256=file_hash(out/'config.json'),manifest_sha256=file_hash(run/'data_manifest.json'),code={str(p):file_hash(p) for p in paths})
        write_json(out/'binding.json',binding);source=out/'source';source.mkdir()
        for p in paths:shutil.copy2(p,source/p.name)
        print('PREPARED',args.candidate,flush=True);return
    cfg=json.loads((out/'config.json').read_text());binding=json.loads((out/'binding.json').read_text())
    assert file_hash(out/'config.json')==binding['config_sha256'] and file_hash(run/'data_manifest.json')==binding['manifest_sha256']
    for p,h in binding['code'].items():assert file_hash(p)==h,p
    trainer=Recovery(root,run,cfg,manifest)
    if args.mode=='preflight':
        p=out/'preflight';p.mkdir(exist_ok=False)
        full=trainer.update()
        # A real batch2 GPU forward/backward/optimizer step, using preflight only.
        (a,b,ta,tb),_,_,_=trainer.data();x=F.interpolate(torch.cat([a,b]),size=(360,640),mode='area');t=F.interpolate(torch.cat([ta,tb]),size=(360,640),mode='area')
        trainer.opt.zero_grad(set_to_none=True);loss=F.l1_loss(trainer.net(x),t);loss.backward();trainer.opt.step()
        torch.save(trainer.state(binding),p/'resume.pt');reference=trainer.update()
        expected={k:v.detach().cpu().clone() for k,v in trainer.net.state_dict().items()}
        del trainer;gc.collect();torch.cuda.empty_cache();trainer=Recovery(root,run,cfg,manifest)
        trainer.restore(torch.load(p/'resume.pt',weights_only=True,map_location='cpu'),binding);replayed=trainer.update()
        error=max(float((v.cpu()-expected[k]).abs().max()) for k,v in trainer.net.state_dict().items() if v.is_floating_point())
        assert reference['source']==replayed['source'] and reference['indices']==replayed['indices'] and error<1e-5,error
        write_json(p/'result.json',dict(status='PASS',binding=binding,full720_two_frame_update=full,batch2_360_loss=float(loss.detach()),restore_error=error))
        print('PREFLIGHT_PASS',args.candidate,error,flush=True);return
    pre=json.loads((out/'preflight/result.json').read_text());assert pre['status']=='PASS' and pre['binding']==binding
    if args.resume:
        trainer.restore(torch.load(args.resume,weights_only=True,map_location='cpu'),binding)
        assert [json.loads(s)['step'] for s in (out/'metrics.jsonl').read_text().splitlines()]==list(range(1,trainer.step+1))
    else:
        if (out/'metrics.jsonl').exists():raise FileExistsError('Explicit resume required')
        evaluate(trainer.net,manifest,out/'validation_000000')
    write_json(out/f'launch_{trainer.step:06d}.json',dict(step=trainer.step,resume=str(args.resume),binding=binding))
    with (out/'metrics.jsonl').open('a',encoding='utf-8',buffering=1) as log:
        while trainer.step<min(args.stop_after,cfg['max_steps']):
            row=trainer.update();log.write(json.dumps(row,ensure_ascii=False)+'\n')
            if trainer.step%10==0:
                write_json(out/'progress.json',row);print('RECOVER',args.candidate,trainer.step,round(row['loss'],5),round(row['seconds'],3),flush=True)
            if trainer.step%100==0 or trainer.step==min(args.stop_after,cfg['max_steps']):
                torch.save(trainer.state(binding),out/f'checkpoint_{trainer.step:06d}.pt');evaluate(trainer.net,manifest,out/f'validation_{trainer.step:06d}')
    complete=trainer.step==cfg['max_steps'];write_json(out/'status.json',dict(status='COMPLETE' if complete else 'BOUNDED_SEGMENT_COMPLETE',step=trainer.step))
    if complete:
        path=out/'pruned_teacher_fp32.pt'
        torch.save(dict(schema='structured-pruned-teacher-recovered-fp32-v1',indices=trainer.indices,
                        generator={k:v.cpu() for k,v in trainer.net.state_dict().items()},steps=trainer.step,
                        binding=binding,production=False,inference_surface_guard=False),path)
        net=build_pruned(root,trainer.indices).cuda().eval();net.load_state_dict(torch.load(path,weights_only=True)['generator'])
        x=as_tensor(read_image(manifest['validation'][0]['source']))[None].cuda()
        with torch.inference_mode():error=float((net(x)-trainer.net(x)).abs().max())
        assert error==0;write_json(out/'export.json',dict(path=str(path),sha256=file_hash(path),reload_error=error,steps=trainer.step))
    print('SEGMENT_COMPLETE',args.candidate,trainer.step,flush=True)


if __name__=='__main__':main()
