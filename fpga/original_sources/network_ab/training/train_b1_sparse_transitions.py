"""Teach sparse, steep colour transitions without pixel-locking fine texture.

The objective borrows the sparse-gradient motivation of Xu et al., SIGGRAPH
Asia 2011; this differentiable training loss is not their image solver.
All inference is the existing B1 conv/ReLU graph, including learned plateaus.
"""
import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from . import train_b1_learned_colour as colour
from .seed_b1_rank_blocks import OUT as SEEDS
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible

OUT=SEEDS/'sparse-transitions'


def quantiles(g,side=8):
    a=g.abs().amax(1,keepdim=True)
    h,w=a.shape[-2:];a=a[:,:,:h//side*side,:w//side*side]
    a=a.reshape(len(a),1,h//side,side,w//side,side).permute(0,2,4,1,3,5).flatten(-3)
    return a.sort(-1).values


def objective(p,t):
    values={'rgb':F.l1_loss(p,t),'planes':F.l1_loss(F.avg_pool2d(p,8),F.avg_pool2d(t,8))}
    hist=[];sparse=[];structure=[]
    for dim in (-1,-2):
        gp,gt=torch.diff(p,dim=dim),torch.diff(t,dim=dim)
        hist.append(F.l1_loss(quantiles(gp),quantiles(gt)))
        mag=gp.square().sum(1);sparse.append((mag/(mag+.02**2)).mean())
        for scale in (4,8):
            a,b=F.avg_pool2d(p,scale),F.avg_pool2d(t,scale)
            structure.append(F.l1_loss(torch.diff(a,dim=dim),torch.diff(b,dim=dim)))
    values.update(gradient_distribution=torch.stack(hist).mean(),sparse=torch.stack(sparse).mean(),
                  structure=torch.stack(structure).mean())
    return .2*values['rgb']+values['planes']+2*values['gradient_distribution']+.06*values['sparse']+values['structure'],values


class Trainer(colour.Trainer):
    def __init__(self,cache,selected,steps):
        super().__init__(cache,selected,'sparse',steps)
        self.net=colour.ColourB1(revive=False,source=SEEDS/'blocks/coefficients-padded.bin').cuda().eval()
        groups=[]
        for kind,parameters in [('w',self.net.weight),('b',self.net.bias)]:
            for key,p in parameters.items():
                rate=.0002 if kind=='w' else .01
                groups.append(dict(params=[p],lr=rate,initial_lr=rate))
        self.opt=torch.optim.Adam(groups)

    def update(self,native=False):
        start=time.perf_counter();s=self.selected[self.step%len(self.selected)]
        item=next(v for v in self.cached if v['row']['id']==s['id'])
        x0,y0,x1,y1=s['box']
        if not native:
            x0=max(0,min(960,x0+self.rng.randrange(-8,9)*4));y0=max(0,min(400,y0+self.rng.randrange(-8,9)*4))
            x1=x0+320;y1=y0+320
        else:x0=y0=0;x1=1280;y1=720
        x=item['source'][None,:,y0:y1,x0:x1];t=item['target'][None,:,y0:y1,x0:x1]
        p,_=self.net(x);cut=lambda a:a[:,:,40:-40,40:-40]
        loss,values=objective(cut(p),cut(t));phase=p.new_zeros(())
        if self.step%4==0 and not native:
            shifted=F.pad(x,(1,0,0,0),mode='replicate')[:,:,:,:x.shape[-1]]
            ps,_=self.net(shifted)
            phase=F.l1_loss(ps[:,:,40:-40,41:-40],p[:,:,40:-40,40:-41])
            loss=loss+.1*phase
        for g in self.opt.param_groups:g['lr']=g['initial_lr']*(.4+.6*.5*(1+math.cos(math.pi*self.step/max(self.steps,1))))
        self.opt.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5,error_if_nonfinite=True);assert grad>0 and torch.isfinite(loss)
        self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),values={k:float(v.detach()) for k,v in values.items()},phase=float(phase.detach()),
                    sample=s['id'],crop=[x0,y0,x1,y1],grad=float(grad),seconds=time.perf_counter()-start)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=1000);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100520);contract_guard();OUT.mkdir(exist_ok=False)
    data,selected,cache=colour.load_data();tr=Trainer(cache,selected,a.steps)
    write_json(OUT/'binding.json',dict(seed=str(SEEDS/'blocks/coefficients-padded.bin'),seed_sha256=file_hash(SEEDS/'blocks/coefficients-padded.bin'),
        steps=a.steps,selected=selected,code_sha256=file_hash(__file__),colour_code_sha256=file_hash(colour.__file__),
        primary_reference='https://www.cse.cuhk.edu.hk/~leojia/papers/L0smooth_Siggraph_Asia2011.pdf',
        inference='Same B1, no oracle, no external filter',objective='Coarse geometry + local gradient quantiles + sparse transitions'))
    for src in (__file__,colour.__file__):shutil.copy2(src,OUT/Path(src).name)
    native=tr.update(True);crop=tr.update();tr.save(OUT/'resume_probe.pt');ref=tr.update()
    expected={k:v.detach().clone() for k,v in tr.net.state_dict().items()}
    fresh=Trainer(cache,selected,a.steps);fresh.restore(OUT/'resume_probe.pt');got=fresh.update()
    error=max(float((v-expected[k]).abs().max()) for k,v in fresh.net.state_dict().items());assert error<1e-5 and ref['crop']==got['crop']
    write_json(OUT/'preflight.json',dict(status='PASS',native=native,crop=crop,resume_error=error))
    print('SPARSE_PREFLIGHT_PASS',error,flush=True)
    del tr,fresh;torch.cuda.empty_cache();tr=Trainer(cache,selected,a.steps)
    colour.review(tr,data,selected,OUT/'initial',oracle_comparison=False)
    with (OUT/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as f:
        while tr.step<a.steps:
            row=tr.update();f.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('SPARSE',tr.step,round(row['loss'],5),row['values'],flush=True)
            if tr.step%500==0 or tr.step==a.steps:
                tr.save(OUT/f'checkpoint_{tr.step:06d}.pt');colour.review(tr,data,selected,OUT/f'step_{tr.step:06d}',oracle_comparison=False)
    print('SPARSE_COMPLETE',flush=True)


if __name__=='__main__':main()
