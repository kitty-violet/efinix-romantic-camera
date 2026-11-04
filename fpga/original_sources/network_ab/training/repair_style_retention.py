"""Paired FP32 recovery trial: preserve teacher changes at fixed network size."""
import argparse
import gc
import json
import math
from pathlib import Path
import random
import shutil
import time
import cv2
import torch
from torch.nn import functional as F
from .importance_pruned_teacher import build_group_pruned
from .recover_pruned_teacher import Recovery,evaluate
from .train_b1_distillation import as_tensor,read_image,edges
from .coherent_shared_model import warp
from .losses import PerceptualLoss
from .engine import set_reproducible
from .runtime import configure_workspace,file_hash,write_json,contract_guard

RUN='style-retention-repair-20261003'
PREVIOUS='importance-pruning-20261003'
INITIAL_SHA='ed41df3091962a387782568265e11751b840c2c9f60db0490ecf5dfb89d50a8f'


def gray(x):return x[:,0:1]*.299+x[:,1:2]*.587+x[:,2:3]*.114


def bands(x,scale):
    x=gray(x)
    if scale>1:x=F.avg_pool2d(x,scale)
    return x-F.avg_pool2d(F.pad(x,(4,4,4,4),mode='replicate'),9,1)


def change_loss(pred,target,source):
    # Plain L1((pred-source)-(target-source)) equals ordinary L1(pred,target).
    # This instead normalizes each tile by the teacher's actual band change,
    # so small style-active regions are not swamped by unchanged image areas.
    loss=pred.new_zeros(());projection=[]
    for scale in (1,2,4):
        actual=bands(pred,scale)-bands(source,scale)
        wanted=(bands(target,scale)-bands(source,scale)).detach()
        k=max(4,32//scale)
        energy=F.avg_pool2d(wanted.square(),k,k)
        error=F.avg_pool2d((actual-wanted).square(),k,k)
        active=(energy>.015**2).to(pred.dtype)
        loss=loss+((error+1e-8).sqrt()/(energy+.03**2).sqrt()*active).sum()/active.sum().clamp_min(1)
        projection.append(((actual*wanted).sum()/(wanted.square().sum()+1e-8)).detach())
    return loss/3,torch.stack(projection).mean()


class Trainer(Recovery):
    def __init__(self,root,cfg,manifest):
        self.config=cfg;self.step=0;self.rng=random.Random(cfg['seed']);self.groups={}
        for row in manifest['pairs']:
            assert manifest['source_splits'][row['source']]['split']=='train'
            self.groups.setdefault(row['source'],[]).append(row)
        path=Path(cfg['initial']);assert file_hash(path)==INITIAL_SHA
        state=torch.load(path,weights_only=True,map_location='cpu');self.indices=state['indices']
        self.net=build_group_pruned(root,self.indices).cuda().eval();self.net.load_state_dict(state['generator'])
        self.opt=torch.optim.Adam(self.net.parameters(),lr=cfg['lr'])
        self.vgg=PerceptualLoss(root).cuda();self.vgg.features=self.vgg.features[:9]

    def spatial(self,p,t,s):
        terms=self.terms(p,t)
        if self.config['mode']=='repair':
            change,projection=change_loss(p[:,:,8:-8,8:-8],t[:,:,8:-8,8:-8],s[:,:,8:-8,8:-8])
        else:
            with torch.no_grad():change,projection=change_loss(p[:,:,8:-8,8:-8],t[:,:,8:-8,8:-8],s[:,:,8:-8,8:-8])
        loss=sum(terms[k]*v for k,v in self.config['weights'].items())
        if self.config['mode']=='repair':loss=loss+self.config['change_weight']*change
        return loss,dict(**terms,change=change,projection=projection)

    def update(self):
        begin=time.perf_counter();self.net.eval()
        (a,b,ta,tb),flow,mask,row=self.data()
        q=min(1.,self.step/(self.config['max_steps']-1))
        lr=(self.config['lr_floor']+(self.config['lr']-self.config['lr_floor'])*.5*(1+math.cos(math.pi*q)))*min(1.,(self.step+1)/10)
        for group in self.opt.param_groups:group['lr']=lr
        self.opt.zero_grad(set_to_none=True)
        pa=self.net(a);la,aa=self.spatial(pa,ta,a);la=la*.5;la.backward();previous=pa.detach();del pa
        pb=self.net(b);lb,bb=self.spatial(pb,tb,b)
        delta=(pb-b)-warp(previous-a,flow)
        old_temporal=(delta.abs()*mask).sum()/(3*mask.sum()+1e-6)
        # Match the temporal evolution of the teacher, rather than rewarding
        # an identity output. Teacher flicker is not assumed to be zero.
        target_delta=(tb-b)-warp(ta-a,flow)
        teacher_temporal=((delta-target_delta).abs()*mask).sum()/(3*mask.sum()+1e-6)
        temporal=teacher_temporal if self.config['mode']=='repair' else old_temporal
        lb=lb*.5+self.config['temporal']*temporal
        assert torch.isfinite(la) and torch.isfinite(lb);lb.backward()
        for name,p in self.net.named_parameters():assert p.grad is not None and torch.isfinite(p.grad).all(),name
        norm=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.,error_if_nonfinite=True);assert norm>0
        self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(la.detach()+lb.detach()),
            **{k:float((v.detach()+bb[k].detach())*.5) for k,v in aa.items()},
            temporal=float(temporal.detach()),old_temporal=float(old_temporal.detach()),teacher_temporal=float(teacher_temporal.detach()),
            grad_norm=float(norm),lr=lr,source=row['source'],indices=[row['current_index'],row['next_index']],native_frames=2,
            seconds=time.perf_counter()-begin,peak_mib=torch.cuda.max_memory_allocated()/2**20)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','preflight','train'])
    ap.add_argument('--mode',choices=['control','repair']);ap.add_argument('--resume',type=Path);ap.add_argument('--stop-after',type=int,default=240);args=ap.parse_args()
    root=configure_workspace();run=root/'runs'/RUN;contract_guard();set_reproducible(2026100313);cv2.setNumThreads(2)
    if args.action=='prepare':
        run.mkdir(exist_ok=False);shutil.copy2(root/'runs'/PREVIOUS/'data_manifest.json',run/'data_manifest.json')
        paths=[Path(__file__)]+[Path(__file__).with_name(n) for n in ['importance_pruned_teacher.py','recover_pruned_teacher.py','prune_animation_teacher.py','trainable_animegan.py','train_b1_distillation.py','coherent_shared_model.py','losses.py']]
        (run/'source').mkdir()
        for p in paths:shutil.copy2(p,run/'source'/p.name)
        binding=dict(code={str(p):file_hash(p) for p in paths},manifest_sha256=file_hash(run/'data_manifest.json'))
        write_json(run/'binding.json',binding)
        for mode in ['control','repair']:
            out=run/mode;out.mkdir()
            cfg=dict(mode=mode,seed=2026100313,max_steps=240,lr=.00015,lr_floor=.00003,
                weights=dict(rgb=1.,edge=2.,perceptual=.1,planes=.5),temporal=.2,change_weight=.15,
                initial=str(root/'runs'/PREVIOUS/'cost14g_recovery/pruned_teacher_fp32.pt'),initial_sha256=INITIAL_SHA,
                pipeline='Same123394 parameter native720 graph; FP32, no extra output postprocessing',
                intervention='repair adds normalized multiscale teacher-change tiles and anchors temporal error to teacher; control keeps previous objective')
            write_json(out/'config.json',cfg)
        print('PREPARED_PAIRED_TRIAL',flush=True);return
    assert args.mode
    out=run/args.mode;cfg=json.loads((out/'config.json').read_text());binding=json.loads((run/'binding.json').read_text())
    binding['config_sha256']=file_hash(out/'config.json')
    assert all(file_hash(p)==h for p,h in binding['code'].items())
    assert file_hash(run/'data_manifest.json')==binding['manifest_sha256']
    manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'));trainer=Trainer(root,cfg,manifest)
    if args.action=='preflight':
        folder=out/'preflight';folder.mkdir(exist_ok=False);full=trainer.update()
        (a,b,ta,tb),_,_,_=trainer.data();x=F.interpolate(torch.cat([a,b]),size=(360,640),mode='area');t=F.interpolate(torch.cat([ta,tb]),size=(360,640),mode='area')
        trainer.opt.zero_grad(set_to_none=True);loss,_=trainer.spatial(trainer.net(x),t,x);loss.backward();trainer.opt.step()
        torch.save(trainer.state(binding),folder/'resume.pt');reference=trainer.update()
        expected={k:v.detach().cpu().clone() for k,v in trainer.net.state_dict().items()}
        del trainer;gc.collect();torch.cuda.empty_cache();trainer=Trainer(root,cfg,manifest)
        trainer.restore(torch.load(folder/'resume.pt',map_location='cpu',weights_only=True),binding);replayed=trainer.update()
        error=max(float((v.cpu()-expected[k]).abs().max()) for k,v in trainer.net.state_dict().items() if v.is_floating_point())
        assert reference['source']==replayed['source'] and reference['indices']==replayed['indices'] and error<1e-5
        # Loss should distinguish weakened teacher changes from the full target.
        with torch.no_grad():
            at_target=float(change_loss(ta,ta,a)[0]);at_identity=float(change_loss(a,ta,a)[0]);at_half=float(change_loss((ta+a)*.5,ta,a)[0])
        assert at_target<at_half<at_identity
        write_json(folder/'result.json',dict(status='PASS',binding=binding,full720_update=full,batch2_loss=float(loss.detach()),restore_error=error,
            change_term_sanity=dict(target=at_target,half=at_half,identity=at_identity),scope='Synthetic blends used only to test loss, never as deliverable output'))
        print('PREFLIGHT_PASS',args.mode,error,flush=True);return
    pre=json.loads((out/'preflight/result.json').read_text());assert pre['status']=='PASS' and pre['binding']==binding
    if args.resume:
        trainer.restore(torch.load(args.resume,map_location='cpu',weights_only=True),binding)
        assert [json.loads(s)['step'] for s in (out/'metrics.jsonl').read_text().splitlines()]==list(range(1,trainer.step+1))
    else:
        assert not (out/'metrics.jsonl').exists();evaluate(trainer.net,manifest,out/'validation_000000')
    write_json(out/f'launch_{trainer.step:06d}.json',dict(step=trainer.step,resume=str(args.resume),binding=binding))
    with (out/'metrics.jsonl').open('a',encoding='utf-8',buffering=1) as log:
        while trainer.step<min(args.stop_after,cfg['max_steps']):
            row=trainer.update();log.write(json.dumps(row,ensure_ascii=False)+'\n')
            if trainer.step%20==0:write_json(out/'progress.json',row);print('TRIAL',args.mode,trainer.step,round(row['loss'],5),round(row['projection'],4),flush=True)
            if trainer.step%120==0 or trainer.step==min(args.stop_after,cfg['max_steps']):
                torch.save(trainer.state(binding),out/f'checkpoint_{trainer.step:06d}.pt');evaluate(trainer.net,manifest,out/f'validation_{trainer.step:06d}')
    complete=trainer.step==cfg['max_steps'];write_json(out/'status.json',dict(status='COMPLETE' if complete else 'BOUNDED_SEGMENT_COMPLETE',step=trainer.step))
    if complete:
        path=out/'pruned_teacher_fp32.pt';torch.save(dict(schema='teacher-change-repair-fp32-v1',indices=trainer.indices,
            generator={k:v.cpu() for k,v in trainer.net.state_dict().items()},steps=trainer.step,binding=binding,config=cfg,production=False),path)
        net=build_group_pruned(root,trainer.indices).cuda();net.load_state_dict(torch.load(path,weights_only=True)['generator'])
        x=as_tensor(read_image(manifest['validation'][0]['source']))[None].cuda()
        with torch.inference_mode():error=float((net(x)-trainer.net(x)).abs().max())
        assert error==0;write_json(out/'export.json',dict(path=str(path),sha256=file_hash(path),reload_error=error,steps=trainer.step))
    print('SEGMENT_COMPLETE',args.mode,trainer.step,flush=True)


if __name__=='__main__':main()
