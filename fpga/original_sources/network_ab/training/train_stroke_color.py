"""Paired FP32 teacher RGB learning from the archived native contour backbone."""
import argparse
from collections import OrderedDict
import copy
import json
import math
from pathlib import Path
import random
import shutil
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch.nn import functional as F
from .stroke_color_net import StrokeColorNet
from .train_detail_rgb import Training as PairData
from .train_direct_teacher_rgb import spatial
from .train_teacher_strokes import criterion as line_loss,strokes
from .train_b1_distillation import as_tensor,to_rgb,read_image
from .coherent_shared_model import warp
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible

ROOT=Path('E:/fpga/model-training-workspace/runs/region-detail-deploy-20261004')
SOURCE=ROOT/'native-teacher-strokes-v1'


def visible_strokes(x):
    gray=(x*x.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    dilated=F.max_pool2d(F.pad(gray,(1,1,1,1),mode='replicate'),3,1)
    closed=-F.max_pool2d(-F.pad(dilated,(1,1,1,1),mode='replicate'),3,1)
    return ((closed-gray-3/128)/(29/128)).clamp(0,1)


def amplitude(p,t,m=24):
    p,t=[v[...,m:-m,m:-m] for v in (p,t)]
    a=(t>.05).float();b=1-a;e=F.smooth_l1_loss(p,t,beta=.1,reduction='none')
    return (e*a).sum()/a.sum().clamp_min(1)+1.5*(e*b).sum()/b.sum().clamp_min(1)


class Training:
    cached=PairData.cached
    batch=PairData.batch
    def __init__(self,out):
        self.out=out;self.cfg=json.loads((out/'config.json').read_text(encoding='utf-8'))
        self.data=json.loads((out/'data.json').read_text(encoding='utf-8'));self.binding=json.loads((out/'binding.json').read_text())
        assert all(file_hash(p)==h for p,h in self.binding['code'].items())
        assert file_hash(out/'config.json')==self.binding['config'] and file_hash(out/'data.json')==self.binding['data']
        self.net=StrokeColorNet().cuda().train()
        s=torch.load(self.cfg['initial'],map_location='cpu',weights_only=True)
        assert file_hash(self.cfg['initial'])==self.cfg['initial_sha256']
        missing=self.net.load_state_dict(s['model'],strict=False)
        assert not missing.unexpected_keys and all(k.startswith(('color_context.','ink_features.','render.','rgb.')) for k in missing.missing_keys)
        heads=[];backbone=[]
        for name,p in self.net.named_parameters():
            (heads if name.startswith(('color_context.','ink_features.','render.','rgb.')) else backbone).append(p)
        self.opt=torch.optim.Adam([dict(params=backbone,lr=.0002,mult=.15),dict(params=heads,lr=.0015,mult=1.)])
        self.rng=random.Random(self.cfg['seed']);self.cache=OrderedDict();self.groups={};self.step=0
        for i,r in enumerate(self.data['pairs']):self.groups.setdefault(r['source'],[]).append(i)

    def state(self):
        return dict(model=self.net.state_dict(),optimizer=self.opt.state_dict(),step=self.step,binding=self.binding,config=self.cfg,
            rng=self.rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())

    def restore(self,s):
        assert s['binding']==self.binding and s['config']==self.cfg
        self.net.load_state_dict(s['model']);self.opt.load_state_dict(s['optimizer']);self.step=s['step']
        self.rng.setstate(s['rng']);torch.set_rng_state(s['torch_rng']);torch.cuda.set_rng_state_all(s['cuda_rng'])

    def update(self,native=None):
        start=time.perf_counter();native=(self.step+1)%100==0 if native is None else native
        x,t,flow,mask,spec=self.batch(native);m=self.cfg['margin'];cut=lambda z:z[...,m:-m,m:-m]
        y,line=self.net(x,aux=True)
        with torch.no_grad():target_line=visible_strokes(t)
        terms=spatial(y,t,m)
        terms['line']=line_loss(line,target_line,margin=m)[0]
        terms['visible']=amplitude(visible_strokes(y),target_line,m)
        with torch.no_grad():
            xd=(x[1:]-warp(x[:1],flow)).abs().mean(1,keepdim=True)
            valid=mask*(xd<8/128);allowed=xd.clamp(max=3/128)+1/128
        delta=(y[1:]-warp(y[:1],flow)).abs().mean(1,keepdim=True)
        terms['temporal']=(cut(F.relu(delta-allowed))*cut(valid)).sum()/cut(valid).sum().clamp_min(1)
        sx,sy=self.rng.choice([(1,0),(0,1),(1,1)])
        shifted=F.pad(x[:1],(sx,0,sy,0),mode='replicate')[...,:x.shape[-2],:x.shape[-1]]
        q=self.net(shifted)
        terms['phase']=(q[...,m+sy:-m,m+sx:-m]-y[:1,...,m:-m-sy if sy else -m,m:-m-sx if sx else -m]).abs().mean()
        ramp=min(1,max(0,(self.step-100)/400))
        loss=sum(v*self.cfg['weights'][k]*(ramp if k in ('visible','phase','temporal') else 1) for k,v in terms.items())
        lr=.0015*(.2+.8*.5*(1+math.cos(math.pi*self.step/self.cfg['steps'])))
        for group in self.opt.param_groups:group['lr']=lr*group['mult']
        self.opt.zero_grad(set_to_none=True);assert torch.isfinite(loss);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.,error_if_nonfinite=True)
        self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),**{k:float(v.detach()) for k,v in terms.items()},
            native=native,sample=spec,grad=float(norm),lr=lr,seconds=time.perf_counter()-start)

    @torch.no_grad()
    def evaluate(self):
        out=self.out/f'eval-{self.step:06d}';out.mkdir(exist_ok=False);rows=[]
        for r in self.data['validation']:
            x=as_tensor(read_image(r['source']))[None].cuda();t=as_tensor(read_image(r['target']))[None].cuda()
            y,ink=self.net(x,aux=True);ablated=self.net(x,ablate_ink=True)
            Image.fromarray(to_rgb(y[0])).save(out/(r['id']+'.png'))
            Image.fromarray(np.uint8((1-ink.sigmoid()[0,0].cpu().numpy())*255)).save(out/(r['id']+'_line.png'))
            a=spatial(y,t,24);tl=visible_strokes(t)
            rows.append(dict(id=r['id'],**{k:float(v) for k,v in a.items()},visible_stroke=float(amplitude(visible_strokes(y),tl)),
                source_visible_stroke=float(amplitude(visible_strokes(x),tl)),line_loss=float(line_loss(ink,tl,margin=24)[0]),
                edge_cue_ablation_rgb_mae=float((y-ablated).abs().mean()*128)))
        write_json(out/'report.json',dict(step=self.step,records=rows,validation_repeatedly_reviewed_not_blind=True,style_accepted=False))
        cases=[('val_stairs_090',(420,180,740,460)),('val_stairs_090',(140,380,460,660)),('val_plaza_090',(80,400,400,680)),('val_new_hallway_120',(680,240,1000,520))]
        board=Image.new('RGB',(1280,len(cases)*316),'#202732');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
        for j,(key,box) in enumerate(cases):
            r=next(r for r in self.data['validation'] if r['id']==key)
            for i,(name,p) in enumerate([('原片',r['source']),('教师目标',r['target']),('学习轮廓＋颜色',out/(key+'.png')),('同期轮廓支路',out/(key+'_line.png'))]):
                d.text((i*320+4,j*316+4),name,font=font,fill='white');board.paste(Image.open(p).convert('RGB').crop(box),(i*320,j*316+34))
        board.save(out/'comparison.png');print('EVAL',self.step,round(sum(r['center'] for r in rows)/len(rows),4),flush=True)


def prepare(out,steps):
    out.mkdir(exist_ok=False);shutil.copy2(SOURCE/'data.json',out/'data.json')
    data=json.loads((out/'data.json').read_text(encoding='utf-8'));hashes={}
    for r in data['pairs']:
        assert r['next_index']==r['current_index']+1
        for k in ('current','next','current_target','next_target','flow'):
            assert file_hash(r[k])==r[k+'_sha256'];hashes[r[k]]=r[k+'_sha256']
    assert set(data['training_sources']).isdisjoint(r['raw_source'] for r in data['validation'])
    for r in data['validation']:
        for k in ('source','target'):assert file_hash(r[k])==r[k+'_sha256'];hashes[r[k]]=r[k+'_sha256']
    initial=SOURCE/'checkpoint_001600.pt'
    cfg=dict(seed=2026100487,initial=str(initial),initial_sha256=file_hash(initial),steps=steps,crop=256,margin=24,
        weights=dict(center=6,boundary=3,flat=2,clean=8,density=.5,line=1.,visible=1.,temporal=3,phase=2),
        precision='FP32; no AMP, TF32, EMA',renderer='native contour features + learned quarter color context + predicted contour cue',
        teacher='original refined AnimeGANv3 Hayao RGB targets',board_validated=False)
    write_json(out/'config.json',cfg)
    names=['train_stroke_color.py','stroke_color_net.py','native_contour_net.py','train_context_contours.py','train_detail_rgb.py',
        'train_direct_teacher_rgb.py','audit_capacity_texture.py','train_teacher_strokes.py','train_b1_distillation.py',
        'coherent_shared_model.py','runtime.py','engine.py','fpga_video_resize.py']
    paths=[Path(__file__).with_name(n) for n in names];(out/'source').mkdir()
    for p in paths:shutil.copy2(p,out/'source'/p.name)
    write_json(out/'binding.json',dict(code={str(p):file_hash(p) for p in paths},config=file_hash(out/'config.json'),data=file_hash(out/'data.json'),images=hashes))
    print('PREPARED',out,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','preflight','train','evaluate']);p.add_argument('--name',default='native-stroke-color-v1')
    p.add_argument('--steps',type=int,default=2400);p.add_argument('--resume',type=Path);a=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100487);torch.set_num_threads(4);cv2.setNumThreads(2)
    out=ROOT/a.name
    if a.mode=='prepare':prepare(out,a.steps);return
    t=Training(out)
    if a.mode=='preflight':
        torch.cuda.reset_peak_memory_stats();rows=[t.update(native=True),t.update(native=False)]
        torch.save(t.state(),out/'preflight_resume.pt');ref=t.update(native=False);expected=copy.deepcopy(t.net.state_dict())
        t.restore(torch.load(out/'preflight_resume.pt',map_location='cpu',weights_only=True));replay=t.update(native=False)
        error=max(float((v-expected[k]).abs().max()) for k,v in t.net.state_dict().items())
        assert error<=1e-7 and ref['sample']==replay['sample']
        x,y,_,_,_=t.batch(True)
        with torch.no_grad():fold=float((t.net(x[:1])-t.net.folded()(x[:1])).abs().max())
        assert fold<=1e-6
        b=t.net.budget();assert b['convolution_weights']<=10256 and b['ideal_macs']<=966758400
        losses=spatial(y,y,24);assert all(float(v)==0 for v in losses.values())
        write_json(out/'preflight.json',dict(status='PASS',binding=t.binding,updates=rows,reference=ref,replay=replay,resume_max_error=error,
            fold_max_error=fold,budget=b,peak_gpu_bytes=torch.cuda.max_memory_allocated(),perfect_teacher_loss={k:float(v) for k,v in losses.items()}))
        print('PREFLIGHT_PASS',error,fold,b['parameters'],b['ideal_macs'],flush=True);return
    pre=json.loads((out/'preflight.json').read_text());assert pre['status']=='PASS' and pre['binding']==t.binding
    if a.resume:t.restore(torch.load(a.resume,map_location='cpu',weights_only=True))
    if a.mode=='evaluate':t.evaluate();return
    assert a.resume or not (out/'metrics.jsonl').exists()
    write_json(out/'status.json',dict(status='RUNNING',step=t.step))
    with (out/'metrics.jsonl').open('a',encoding='utf-8',buffering=1) as f:
        while t.step<min(a.steps,t.cfg['steps']):
            row=t.update();f.write(json.dumps(row)+'\n')
            if t.step%50==0:write_json(out/'progress.json',row);print(json.dumps(row),flush=True)
            if t.step%400==0 or t.step==min(a.steps,t.cfg['steps']):
                torch.save(t.state(),out/f'checkpoint_{t.step:06d}.pt');t.evaluate()
    torch.save(dict(model=t.net.folded().state_dict(),folded=True,step=t.step,binding=t.binding,config=t.cfg,board_validated=False),out/'generator_fp32.pt')
    write_json(out/'status.json',dict(status='COMPLETED_BOUNDED_RUN',step=t.step,weight_sha256=file_hash(out/'generator_fp32.pt'),style_accepted=False))
    print('COLOR_TRAIN_DONE',t.step,flush=True)


if __name__=='__main__':main()
