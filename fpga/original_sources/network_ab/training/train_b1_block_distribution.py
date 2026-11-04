"""Controlled loss correction: coarse correspondence, unaligned block texture.

No pixelwise fine-edge matching. Only existing B1 operators execute inference.
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
from . import train_b1_colour_texture as texture
from . import train_b1_live_rank_features as live
from .train_b1_sparse_transitions import quantiles
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible

OUT=live.OUT.parent/'block-distribution'
PARENT=live.OUT/'checkpoint_003000.pt'


class Trainer(texture.Trainer):
    def __init__(self,cache,selected,steps):
        super().__init__(cache,selected,steps)
        s=torch.load(PARENT,map_location='cpu',weights_only=True)
        self.net.load_state_dict(s['model']);self.critic.load_state_dict(s['critic'])
        self.contour_weight=0.
        groups=[]
        for kind,parameters in [('w',self.net.weight),('b',self.net.bias)]:
            for key,p in parameters.items():
                rate=.0002 if kind=='w' else .008
                groups.append(dict(params=[p],lr=rate,initial_lr=rate))
        self.opt=torch.optim.Adam(groups,betas=(.5,.99))
        self.dopt=torch.optim.Adam(self.critic.parameters(),lr=.00005,betas=(.5,.99))

    def update(self,native=False):
        start=time.perf_counter();s=self.selected[self.step%len(self.selected)]
        item=next(v for v in self.cached if v['row']['id']==s['id']);x0,y0,x1,y1=s['box']
        if not native:
            x0=max(0,min(960,x0+self.rng.randrange(-8,9)*4));y0=max(0,min(400,y0+self.rng.randrange(-8,9)*4))
            x1=x0+320;y1=y0+320
        else:x0=y0=0;x1=1280;y1=720
        x=item['source'][None,:,y0:y1,x0:x1];t=item['target'][None,:,y0:y1,x0:x1]
        p,_=self.net(x);p0=p[:,:,40:-40,40:-40];t0=t[:,:,40:-40,40:-40]
        for v in self.critic.parameters():v.requires_grad_(True)
        self.dopt.zero_grad(set_to_none=True)
        real=self.critic(t0)[-1];fake=self.critic(p0.detach())[-1]
        blur=F.avg_pool2d(F.pad(t0,(2,2,2,2),mode='replicate'),5,1)
        dl=.5*(real-1).square().mean()+.25*(fake.square().mean()+self.critic(blur)[-1].square().mean())
        dl.backward();torch.nn.utils.clip_grad_norm_(self.critic.parameters(),5,error_if_nonfinite=True);self.dopt.step()
        for v in self.critic.parameters():v.requires_grad_(False)
        adv=(self.critic(p0)[-1]-1).square().mean()
        planes=torch.stack([F.l1_loss(F.avg_pool2d(p0,k),F.avg_pool2d(t0,k)) for k in (8,16)]).mean()
        hist=torch.stack([F.l1_loss(quantiles(torch.diff(p0,dim=d),16),quantiles(torch.diff(t0,dim=d),16)) for d in (-1,-2)]).mean()
        loss=planes+.05*F.l1_loss(p0,t0)+4*hist+.4*adv
        phase=p.new_zeros(())
        if self.step%4==0 and not native:
            shifted=F.pad(x,(1,0,0,0),mode='replicate')[:,:,:,:x.shape[-1]]
            ps,_=self.net(shifted);phase=F.l1_loss(ps[:,:,40:-40,41:-40],p[:,:,40:-40,40:-41])
            loss=loss+.2*phase
        for g in self.opt.param_groups:g['lr']=g['initial_lr']*(.5+.5*.5*(1+math.cos(math.pi*self.step/max(self.steps,1))))
        self.opt.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5,error_if_nonfinite=True);assert grad>0 and torch.isfinite(loss)
        self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),planes=float(planes.detach()),hist=float(hist.detach()),
                    adv=float(adv.detach()),discriminator=float(dl.detach()),phase=float(phase.detach()),grad=float(grad),
                    crop=[x0,y0,x1,y1],sample=s['id'],seconds=time.perf_counter()-start)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=2000);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100522);contract_guard();OUT.mkdir(exist_ok=False)
    data,selected,cache=colour.load_data();tr=Trainer(cache,selected,a.steps)
    write_json(OUT/'binding.json',dict(parent=str(PARENT),sha256=file_hash(PARENT),steps=a.steps,selected=selected,
        code={str(p):file_hash(p) for p in [Path(__file__),Path(colour.__file__),Path(texture.__file__)]},
        rationale='Parameter-gradient conflict measured; remove fine pixel edge lock, retain coarse correspondence',
        actual_B1=True,external_inference=False))
    for src in (__file__,colour.__file__,texture.__file__):shutil.copy2(src,OUT/Path(src).name)
    native=tr.update(True);crop=tr.update();tr.save(OUT/'resume_probe.pt');ref=tr.update()
    expected={k:v.detach().clone() for k,v in tr.net.state_dict().items()};expected_d={k:v.detach().clone() for k,v in tr.critic.state_dict().items()}
    fresh=Trainer(cache,selected,a.steps);fresh.restore(OUT/'resume_probe.pt');got=fresh.update()
    error=max(float((v-expected[k]).abs().max()) for k,v in fresh.net.state_dict().items())
    de=max(float((v-expected_d[k]).abs().max()) for k,v in fresh.critic.state_dict().items());assert error<1e-5 and de<1e-5 and ref['crop']==got['crop']
    write_json(OUT/'preflight.json',dict(status='PASS',native=native,crop=crop,resume_error=error,critic_resume_error=de))
    print('BLOCK_DISTRIBUTION_PREFLIGHT_PASS',error,de,flush=True)
    del tr,fresh;torch.cuda.empty_cache();tr=Trainer(cache,selected,a.steps)
    with (OUT/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as f:
        while tr.step<a.steps:
            row=tr.update();f.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('BLOCK_DISTRIBUTION',tr.step,round(row['loss'],4),round(row['planes'],4),round(row['hist'],4),flush=True)
            if tr.step%500==0 or tr.step==a.steps:
                tr.save(OUT/f'checkpoint_{tr.step:06d}.pt');colour.review(tr,data,selected,OUT/f'step_{tr.step:06d}',oracle_comparison=False)
    print('BLOCK_DISTRIBUTION_COMPLETE',flush=True)


if __name__=='__main__':main()
