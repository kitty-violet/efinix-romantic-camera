"""Cartoon distillation with an EXACT, unchanged deployed B1 generator.

All critics, representations, optical flow and phase comparisons are training
only. No filter, critic, teacher, time state or RGB shortcut at inference.
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
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F

from network_ab.contract_torch import build_fp32_model
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .engine import set_reproducible
from .train_b1_distillation import as_tensor, to_rgb, read_image, read_flow, edges
from .coherent_shared_model import warp
from .repair_style_retention import gray, change_loss
from .losses import PerceptualLoss

RUN = 'b1-cartoon-budget-20261004'
INITIAL_SHA = 'b6de9e4ea43d9515819a9f60240484632d796acb328dd2360a895ae8db546296'


def box(x, k):
    return F.avg_pool2d(F.pad(x, (k//2,)*4, mode='replicate'), k, 1)


class TextureJudge(nn.Module):
    """Training-only, conditional gray/local-contrast PatchGAN; not deployed."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3,24,3,2,1), nn.LeakyReLU(.2),
            nn.Conv2d(24,48,3,2,1), nn.LeakyReLU(.2),
            nn.Conv2d(48,64,3,2,1), nn.LeakyReLU(.2), nn.Conv2d(64,1,3,1,1))

    def forward(self, output, source):
        g, s = gray(output), gray(source)
        return self.layers(torch.cat([s, g, (g-box(g,9))*3], 1))


def prepare(root, run):
    from .media import decode_frames
    from .fpga_video_resize import area_3to2
    from .prepare_b1_distillation import rgb_output, TEACHER_SHA
    from .prepare_coherent_shared_data import pair_flow, contact
    from .trainable_animegan import TrainableAnimeGAN
    run.mkdir(exist_ok=False)
    folder = run/'adjacent_data'; folder.mkdir()
    old_path = root/'runs/b1-distillation-20261003/data/manifest.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))
    groups = {}
    for row in old['pairs']:
        assert old['source_splits'][row['source']]['split']=='train'
        groups.setdefault(row['source'], []).append(row)
    initial = root/'runs/b1-distillation-20261003/final-2000/b1_style_fp32.pt'
    assert file_hash(initial)==INITIAL_SHA
    teacher = Path(old['teacher']); assert file_hash(teacher)==TEACHER_SHA
    net = TrainableAnimeGAN(root).cuda().eval()
    net.load_state_dict(torch.load(teacher, map_location='cpu', weights_only=True)['generator'])
    pairs=[]; views=[]
    for source, rows in sorted(groups.items()):
        assert file_hash(source)==rows[0]['source_sha256']
        wanted = {r['current_index']+1:r for r in rows}
        for index,pts,rgb in decode_frames(source,indices=sorted(wanted),hlg_exposure_stops=-1.):
            row = wanted[index]
            assert file_hash(row['current'])==row['current_sha256']
            assert file_hash(row['current_target'])==row['current_target_sha256']
            x=area_3to2(rgb); target=rgb_output(net,x)
            prefix=folder/(row['source_sha256'][:12]+f'_{index:06d}')
            ip=prefix.with_name(prefix.name+'_source.png');tp=prefix.with_name(prefix.name+'_target.png')
            Image.fromarray(x).save(ip);Image.fromarray(target).save(tp)
            flow, mask = pair_flow(read_image(row['current']),x)
            fp=prefix.with_name(prefix.name+'_flow.npz');np.savez_compressed(fp,backward=flow,mask=mask)
            q={k:row[k] for k in ('source','source_sha256','current','current_sha256','current_target','current_target_sha256','current_index','current_pts')}
            q.update(next=str(ip),next_sha256=file_hash(ip),next_target=str(tp),next_target_sha256=file_hash(tp),next_index=index,next_pts=pts,
                flow=str(fp),flow_sha256=file_hash(fp),split='train',valid_fraction=float(mask.mean()))
            assert 0 < q['next_pts']-q['current_pts'] < .05, q
            pairs.append(q)
            views.append(dict(path=str(tp),scene=Path(source).name+f' f{index}'))
        print('ADJACENT_SOURCE',Path(source).name,len(rows),flush=True)
    contact(views,folder,'targets','path')
    del net;gc.collect();torch.cuda.empty_cache()
    manifest={**old,'schema':'b1-real-consecutive-pairs-v1','pairs':pairs,'parent':str(old_path),'parent_sha256':file_hash(old_path),
        'status':'PREPARED_NATIVE_720_ADJACENT_FRAMES','note':'95 pairs now gap1, exact old source split. 6 previously seen regressions remain excluded.'}
    write_json(run/'data_manifest.json',manifest)
    cfg=dict(seed=2026100407,max_steps=2000,lr=.001,lr_floor=.0001,critic_lr=.0001,crop=400,margin=40,
        pairs_per_step=2,native_interval=50,initial=str(initial),initial_sha256=INITIAL_SHA,
        weights=dict(rgb=.5,planes=2.,edge=2.,sharp=.6,flat=.4,perceptual=.06,change=.45,adversarial=.07,temporal=.7,phase=1.5),
        teacher_temporal_gain=.5,critic_warmup=100,stability_ramp=300,
        generator='EXACT unchanged B1 8/16/32 R3, 24 convolutions; no extra inference ops',
        precision='FP32; no AMP/TF32/EMA',deployed_postprocess='none',
        references=['https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/train_code/train.py'],
        distinction='Inspired by separate texture/surface training, not a reproduction. No guided filter in generator output. Conditional teacher targets, blur negatives and phase equivariance added.')
    write_json(run/'config.json',cfg)
    paths=[Path(__file__)]+[Path(__file__).with_name(n) for n in ['runtime.py','engine.py','train_b1_distillation.py','coherent_shared_model.py','repair_style_retention.py','losses.py','prepare_b1_distillation.py','prepare_coherent_shared_data.py','media.py','fpga_video_resize.py','smooth_surface_guard.py','trainable_animegan.py']]
    paths += [Path(__file__).parents[1]/n for n in ['contract_torch.py','model_contract.py','model_b_contract.json']]
    frozen=run/'source';frozen.mkdir()
    for p in paths:shutil.copy2(p,frozen/p.name)
    binding=dict(code={str(p):file_hash(p) for p in paths},config_sha256=file_hash(run/'config.json'),manifest_sha256=file_hash(run/'data_manifest.json'))
    write_json(run/'binding.json',binding)
    _,plan=contract_guard()
    write_json(run/'hardware_boundary.json',dict(contract_file_sha256=file_hash(Path(__file__).parents[1]/'model_b_contract.json'),
        contract_canonical_sha256=plan['contract_sha256'],nodes=len(plan['nodes']),convolutions=plan['quantized_operator_count'],
        trainable_parameters=plan['trainable_parameter_count'],macs_per_frame=plan['macs_per_frame'],
        inference_graph_changed=False,hardware_files_changed=False,extra_deployed_state=False,
        int8_verified=False,physical_timing_verified=False))
    write_json(run/'data_checks.json',dict(pairs=len(pairs),sources=len(groups),all_gap1=all(p['next_index']==p['current_index']+1 for p in pairs),
        pts_delta_range=[min(p['next_pts']-p['current_pts'] for p in pairs),max(p['next_pts']-p['current_pts'] for p in pairs)],
        source_splits_preserved=True,previously_seen_regression_not_blind=True))
    print('PREPARED',len(pairs),flush=True)


class Training:
    def __init__(self, root, cfg, manifest):
        self.cfg=cfg;self.manifest=manifest;self.step=0;self.rng=random.Random(cfg['seed']);self.groups={}
        for row in manifest['pairs']:self.groups.setdefault(row['source'],[]).append(row)
        self.net=build_fp32_model().cuda().eval()
        assert file_hash(cfg['initial'])==cfg['initial_sha256']
        self.net.load_state_dict(torch.load(cfg['initial'],map_location='cpu',weights_only=True)['model'])
        self.judge=TextureJudge().cuda();self.opt=torch.optim.Adam(self.net.parameters(),lr=cfg['lr'])
        self.dopt=torch.optim.Adam(self.judge.parameters(),lr=cfg['critic_lr'],betas=(.5,.99))
        self.vgg=PerceptualLoss(root).cuda();self.vgg.features=self.vgg.features[:16]

    def batch(self, native):
        src=[];tgt=[];flows=[];masks=[];spec=[]
        for _ in range(1 if native else self.cfg['pairs_per_step']):
            row=self.rng.choice(self.groups[self.rng.choice(sorted(self.groups))])
            w,h=(1280,720) if native else (self.cfg['crop'],)*2
            # All 16 sampling phases, unlike the old crops aligned to multiples of4.
            x=self.rng.randrange(1280-w+1);y=self.rng.randrange(720-h+1)
            src.append([as_tensor(read_image(row[k])[y:y+h,x:x+w]) for k in ('current','next')])
            tgt.append([as_tensor(read_image(row[k])[y:y+h,x:x+w]) for k in ('current_target','next_target')])
            flow,mask=read_flow(row['flow']);f=flow[y:y+h,x:x+w].copy();m=mask[y:y+h,x:x+w].copy()
            yy,xx=np.mgrid[:h,:w];margin=self.cfg['margin']
            m*=((xx+f[:,:,0]>=margin)&(xx+f[:,:,0]<w-margin)&(yy+f[:,:,1]>=margin)&(yy+f[:,:,1]<h-margin))
            flows.append(torch.from_numpy(f).permute(2,0,1));masks.append(torch.from_numpy(m)[None])
            spec.append(dict(source=row['source'],indices=[row['current_index'],row['next_index']],xy=[x,y],wh=[w,h]))
        # Flatten as all current followed by all next.
        a=torch.stack([r[0] for r in src]+[r[1] for r in src]).cuda()
        t=torch.stack([r[0] for r in tgt]+[r[1] for r in tgt]).cuda()
        return a,t,torch.stack(flows).cuda(),torch.stack(masks).cuda(),spec

    def spatial(self,p,t,x,native):
        m=self.cfg['margin'];p,t,x=[v[:,:,m:-m,m:-m] for v in (p,t,x)]
        ep,et=edges(gray(p)),edges(gray(t))
        flat=[];sharp=[];edge=[]
        for scale in (1,2,4):
            pp,tt=[F.avg_pool2d(gray(v),scale) if scale>1 else gray(v) for v in (p,t)]
            for gp,gt in zip(edges(pp),edges(tt)):
                significant=(gt.abs()>.04).float()
                smooth=(gt.abs()<.008).float()
                edge.append((gp-gt).abs().mean())
                sharp.append((F.relu(gt.abs()-gp.abs()-.002)*significant).sum()/significant.sum().clamp_min(1))
                flat.append((F.relu(gp.abs()-gt.abs()-.004)*smooth).sum()/smooth.sum().clamp_min(1))
        pv,tv=[F.avg_pool2d(v,2) if native else v for v in (p,t)]
        with torch.no_grad():tf=self.vgg.extract(tv)
        pf=self.vgg.extract(pv)
        change,projection=change_loss(p,t,x)
        losses=dict(rgb=F.l1_loss(p,t),planes=F.l1_loss(F.avg_pool2d(p,16),F.avg_pool2d(t,16)),
            edge=torch.stack(edge).mean(),sharp=torch.stack(sharp).mean(),flat=torch.stack(flat).mean(),
            perceptual=(F.l1_loss(pf['relu2_2'],tf['relu2_2'])+F.l1_loss(pf['relu3_3'],tf['relu3_3']))*.5,
            change=change)
        return losses,projection,(p,t,x)

    def update(self,native=None):
        begin=time.perf_counter();self.net.eval()
        if native is None:native=(self.step+1)%self.cfg['native_interval']==0
        x,t,flow,mask,spec=self.batch(native);n=len(spec);p=self.net(x)
        losses,projection,(pc,tc,xc)=self.spatial(p,t,x,native)
        # Judge sees native detail even on full frames; limit its crop rather than resize.
        dp,dt,dx=[v[:,:,:320,:320] if native else v for v in (pc,tc,xc)]
        self.judge.requires_grad_(True);self.dopt.zero_grad(set_to_none=True)
        real=self.judge(dt,dx);fake=self.judge(dp.detach(),dx)
        blurred=self.judge(box(dt,9),dx)
        dl=F.relu(1-real).mean()+.5*F.relu(1+fake).mean()+.5*F.relu(1+blurred).mean()
        assert torch.isfinite(dl);dl.backward();torch.nn.utils.clip_grad_norm_(self.judge.parameters(),5.,error_if_nonfinite=True);self.dopt.step()
        self.judge.requires_grad_(False)
        losses['adversarial']=-self.judge(dp,dx).mean()
        valid=mask*( (x[n:]-warp(x[:n],flow)).abs().mean(1,keepdim=True)<(8/128)).float()
        delta=(p[n:]-x[n:])-warp(p[:n]-x[:n],flow)
        with torch.no_grad():td=(t[n:]-x[n:])-warp(t[:n]-x[:n],flow)
        m=self.cfg['margin'];cut=lambda v:v[:,:,m:-m,m:-m]
        losses['temporal']=(cut(delta-self.cfg['teacher_temporal_gain']*td).abs()*cut(valid)).sum()/(3*cut(valid).sum()+1e-6)
        sx,sy=self.rng.choice([(1,0),(2,0),(3,0),(0,1),(0,2),(0,3),(1,1),(3,2)])
        # A true translated input and corresponding output crop; no roll seam in measured area.
        shifted=F.pad(x[:1],(sx,0,sy,0),mode='replicate')[:,:,:x.shape[2],:x.shape[3]]
        ps=self.net(shifted)
        losses['phase']=F.l1_loss(ps[:,:,m+sy:-m,m+sx:-m],p[:1,:,m:-m-sy if sy else -m,m:-m-sx if sx else -m])
        ramp=min(1.,(self.step+1)/self.cfg['stability_ramp'])
        adv=min(1.,max(0.,(self.step+1-self.cfg['critic_warmup'])/200))
        loss=sum(v*self.cfg['weights'][k]*(adv if k=='adversarial' else ramp if k in ('phase','temporal') else 1.) for k,v in losses.items())
        assert torch.isfinite(loss)
        q=self.step/max(1,self.cfg['max_steps']-1)
        lr=(self.cfg['lr_floor']+(self.cfg['lr']-self.cfg['lr_floor'])*.5*(1+math.cos(math.pi*q)))*min(1.,(self.step+1)/25)
        for group in self.opt.param_groups:group['lr']=lr
        self.opt.zero_grad(set_to_none=True);loss.backward()
        for name,v in self.net.named_parameters():assert v.grad is not None and torch.isfinite(v.grad).all(),name
        norm=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.,error_if_nonfinite=True);assert norm>0
        self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),**{k:float(v.detach()) for k,v in losses.items()},
            dloss=float(dl.detach()),projection=float(projection),grad_norm=float(norm),lr=lr,native=native,samples=spec,
            phase_xy=[sx,sy],seconds=time.perf_counter()-begin,peak_mib=torch.cuda.max_memory_allocated()/2**20)

    def state(self,binding):
        return dict(model=self.net.state_dict(),judge=self.judge.state_dict(),optimizer=self.opt.state_dict(),doptimizer=self.dopt.state_dict(),step=self.step,
            sampler=self.rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),config=self.cfg,binding=binding)

    def restore(self,s,binding):
        assert s['binding']==binding and s['config']==self.cfg
        self.net.load_state_dict(s['model']);self.judge.load_state_dict(s['judge']);self.opt.load_state_dict(s['optimizer']);self.dopt.load_state_dict(s['doptimizer'])
        self.step=s['step'];self.rng.setstate(s['sampler']);torch.set_rng_state(s['torch_rng']);torch.cuda.set_rng_state_all(s['cuda_rng'])


def evaluate(net,manifest,out):
    from .b1_fit_controls import score
    out.mkdir(exist_ok=False);records=[];panels=[]
    with torch.inference_mode():
        for row in manifest['validation']:
            source,target=[read_image(row[k]) for k in ('source','target')]
            x=as_tensor(source)[None].cuda();p=net(x);rgb=to_rgb(p[0]);t=as_tensor(target)[None].cuda()
            Image.fromarray(rgb).save(out/(row['id']+'.png'))
            phase=[]
            for shift in (1,2,3,4):
                z=net(F.pad(x,(shift,0,0,0),mode='replicate')[:,:,:,:1280])
                phase.append(float((z[:,:,48:-48,48+shift:-48]-p[:,:,48:-48,48:-48-shift]).abs().mean()*128))
            _,proj=change_loss(p,t,x)
            records.append(dict(id=row['id'],teacher=score(rgb,target),source=score(rgb,source),phase_mean_rgb255=phase,band_projection=float(proj),
                new_white_fraction=float(((rgb.min(2)>245)&(source.max(2)<220)).mean())))
            panel=Image.new('RGB',(1280,388),'#18202a');d=ImageDraw.Draw(panel)
            for i,(a,label) in enumerate([(source,'SOURCE'),(target,'TEACHER'),(rgb,'EXACT B1 NEW')]):
                panel.paste(Image.fromarray(a).resize((426,240)),(i*426,28));d.text((i*426+6,6),label,fill='white')
            panels.append(panel.crop((0,0,1280,268)))
    board=Image.new('RGB',(1280,268*len(panels)))
    for i,p in enumerate(panels):board.paste(p,(0,268*i))
    board.save(out/'comparison.jpg',quality=95)
    write_json(out/'scores.json',dict(records=records,not_blind=True))
    return records


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','preflight','train']);ap.add_argument('--stop-after',type=int,default=2000);ap.add_argument('--resume',type=Path);a=ap.parse_args()
    root=configure_workspace();set_reproducible(2026100407);cv2.setNumThreads(2);contract_guard();run=root/'runs'/RUN
    if a.action=='prepare':prepare(root,run);return
    cfg=json.loads((run/'config.json').read_text());manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'));binding=json.loads((run/'binding.json').read_text())
    assert all(file_hash(p)==v for p,v in binding['code'].items())
    assert file_hash(run/'config.json')==binding['config_sha256'] and file_hash(run/'data_manifest.json')==binding['manifest_sha256']
    for row in manifest['pairs']:
        for k in ('current','next','current_target','next_target','flow'):assert file_hash(row[k])==row[k+'_sha256']
    tr=Training(root,cfg,manifest)
    if a.action=='preflight':
        out=run/'preflight';out.mkdir(exist_ok=False);tests=[tr.update(True),tr.update(False)]
        torch.save(tr.state(binding),out/'resume.pt');ref=tr.update(False)
        expected={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};de={k:v.cpu().clone() for k,v in tr.judge.state_dict().items()}
        del tr;gc.collect();torch.cuda.empty_cache();tr=Training(root,cfg,manifest);tr.restore(torch.load(out/'resume.pt',map_location='cpu',weights_only=True),binding);rep=tr.update(False)
        err=max(float((v.cpu()-expected[k]).abs().max()) for k,v in tr.net.state_dict().items() if v.is_floating_point())
        derr=max(float((v.cpu()-de[k]).abs().max()) for k,v in tr.judge.state_dict().items())
        assert err<2e-6 and derr<2e-6 and ref['samples']==rep['samples'] and ref['phase_xy']==rep['phase_xy']
        assert all(v.dtype==torch.float32 for v in tr.net.parameters())
        write_json(out/'result.json',dict(status='PASS',binding=binding,tests=tests,model_restore_error=err,judge_restore_error=derr,
            full720_backward=True,batch4_crop_backward=True,includes_optimizer_and_random_state=True))
        print('PREFLIGHT_PASS',err,derr,flush=True);return
    pre=json.loads((run/'preflight/result.json').read_text());assert pre['status']=='PASS' and pre['binding']==binding
    out=run/'train';out.mkdir(exist_ok=True)
    if a.resume:
        tr.restore(torch.load(a.resume,map_location='cpu',weights_only=True),binding)
        assert [json.loads(s)['step'] for s in (out/'metrics.jsonl').read_text().splitlines()]==list(range(1,tr.step+1))
    else:assert not (out/'metrics.jsonl').exists();evaluate(tr.net,manifest,out/'validation_000000')
    write_json(out/f'launch_{tr.step:06d}.json',dict(step=tr.step,resume=str(a.resume),binding=binding))
    with (out/'metrics.jsonl').open('a',encoding='utf-8',buffering=1) as log:
        while tr.step<min(a.stop_after,cfg['max_steps']):
            row=tr.update();log.write(json.dumps(row,ensure_ascii=False)+'\n')
            if tr.step%25==0:write_json(out/'progress.json',row);print('B1_CARTOON',tr.step,round(row['loss'],4),round(row['projection'],3),round(row['phase']*128,3),round(row['seconds'],2),flush=True)
            if tr.step%250==0 or tr.step==min(a.stop_after,cfg['max_steps']):
                torch.save(tr.state(binding),out/f'checkpoint_{tr.step:06d}.pt');evaluate(tr.net,manifest,out/f'validation_{tr.step:06d}')
    write_json(out/'status.json',dict(status='COMPLETE' if tr.step==cfg['max_steps'] else 'SEGMENT_COMPLETE',step=tr.step))
    print('SEGMENT_COMPLETE',tr.step,flush=True)


if __name__=='__main__':main()
