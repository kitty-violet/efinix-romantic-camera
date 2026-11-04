"""Add a training-only texture critic to the unchanged learned-colour B1.

Motivation: author's white-box cartoon train_code/train.py separates surface /
texture adversaries from structure reconstruction. This is a new PyTorch
experiment, not a reproduction of that paper and not an extra deployed stage.
"""
import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from . import train_b1_learned_colour as colour
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .engine import set_reproducible

OUT=colour.OUT/'texture-v2'
PARENT=colour.OUT/'oracle_full/checkpoint_001200.pt'


class TextureCritic(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        for a,b,stride in [(4,24,1),(24,32,2),(32,48,2),(48,64,1),(64,1,1)]:
            layers.append(nn.Conv2d(a,b,3,stride,1))
        self.layers=nn.ModuleList(layers)

    def forward(self,x):
        # Gray intensities and signed local colour differences make the critic
        # inspect transition shape as well as colour, not colour alone.
        gray=(x*torch.tensor([.299,.587,.114],device=x.device)[None,:,None,None]).sum(1,keepdim=True)
        a=torch.cat((gray,x-F.avg_pool2d(F.pad(x,(2,2,2,2),mode='replicate'),5,1)),1)
        features=[]
        for i,layer in enumerate(self.layers):
            a=layer(a)
            if i<len(self.layers)-1:a=F.leaky_relu(a,.2)
            features.append(a)
        return features


class Trainer(colour.Trainer):
    def __init__(self,cached,selected,steps):
        super().__init__(cached,selected,'texture',steps)
        self.critic=TextureCritic().cuda().train()
        self.dopt=torch.optim.Adam(self.critic.parameters(),lr=.00015,betas=(.5,.99))
        self.contour_weight=.04
        for group in self.opt.param_groups:
            if group['initial_lr']<.01:
                group['lr']*=1.6;group['initial_lr']*=1.6

    def update(self,native=False):
        start=time.perf_counter();selected=self.selected[self.step%len(self.selected)]
        item=next(v for v in self.cached if v['row']['id']==selected['id'])
        x0,y0,x1,y1=selected['box']
        # Jitter only within each of the four training scenes. No validation
        # or additional source video is used to fit this controlled experiment.
        if not native:
            dx,dy=self.rng.randrange(-8,9)*4,self.rng.randrange(-8,9)*4
            x0=max(0,min(960,x0+dx));y0=max(0,min(400,y0+dy));x1=x0+320;y1=y0+320
        else:x0=y0=0;x1=1280;y1=720
        x=item['source'][None,:,y0:y1,x0:x1];t=item['target'][None,:,y0:y1,x0:x1]
        gate=item['gate'][None,:,y0:y1:4,x0:x1:4]
        p,predicted=self.net(x)
        m=40;p0=p[:,:,m:-m,m:-m];t0=t[:,:,m:-m,m:-m]
        for v in self.critic.parameters():v.requires_grad_(True)
        self.dopt.zero_grad(set_to_none=True)
        real=self.critic(t0)[-1];fake=self.critic(p0.detach())[-1]
        # Deliberately blurred target is an additional negative: the critic
        # must distinguish loss of block boundaries from clean flat interiors.
        blur=F.avg_pool2d(F.pad(t0,(2,2,2,2),mode='replicate'),5,1)
        soft=self.critic(blur)[-1]
        dl=.5*(real-1).square().mean()+.25*(fake.square().mean()+soft.square().mean())
        dl.backward();torch.nn.utils.clip_grad_norm_(self.critic.parameters(),5,error_if_nonfinite=True);self.dopt.step()
        for v in self.critic.parameters():v.requires_grad_(False)
        pf=self.critic(p0)
        with torch.no_grad():tf=self.critic(t0)
        adv=(pf[-1]-1).square().mean()
        feature=torch.stack([F.l1_loss(a,b) for a,b in zip(pf[:-1],tf[:-1])]).mean()
        reconstruction,values=colour.loss_colour(p0,t0)
        contour=(F.smooth_l1_loss(predicted[:,:,10:-10,10:-10]/24,gate[:,:,10:-10,10:-10]/24)
                 if self.contour_weight else p.new_zeros(()))
        loss=reconstruction+.08*adv+.3*feature+self.contour_weight*contour
        for group in self.opt.param_groups:
            group['lr']=group['initial_lr']*(.4+.6*.5*(1+math.cos(math.pi*self.step/max(self.steps,1))))
        self.opt.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5,error_if_nonfinite=True)
        assert torch.isfinite(loss) and grad>0
        self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),discriminator=float(dl.detach()),adversarial=float(adv.detach()),
                    feature=float(feature.detach()),contour=float(contour.detach()),values={k:float(v.detach()) for k,v in values.items()},
                    sample=selected['id'],crop=[x0,y0,x1,y1],seconds=time.perf_counter()-start,grad=float(grad))

    def save(self,path):
        super().save(path)
        s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(critic=self.critic.state_dict(),dopt=self.dopt.state_dict())
        torch.save(s,path)

    def restore(self,path):
        super().restore(path)
        s=torch.load(path,map_location='cpu',weights_only=True)
        self.critic.load_state_dict(s['critic']);self.dopt.load_state_dict(s['dopt'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=1200);args=ap.parse_args()
    configure_workspace();set_reproducible(colour.SEED);contract_guard();OUT.mkdir(exist_ok=False)
    data,selected,cached=colour.load_data()
    tr=Trainer(cached,selected,args.steps)
    initial=torch.load(PARENT,map_location='cpu',weights_only=True)['model'];tr.net.load_state_dict(initial)
    write_json(OUT/'binding.json',dict(parent=str(PARENT),parent_sha256=file_hash(PARENT),selected=selected,steps=args.steps,
        code_sha256=file_hash(__file__),colour_code_sha256=file_hash(colour.__file__),
        primary_reference='https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/train_code/train.py',
        differences='Own paired RGB/boundary objective and local texture critic; no claim of paper reproduction',
        deployed='Same B1 only. Critic and target interventions are not inference operators.'))
    for src in (__file__,colour.__file__):shutil.copy2(src,OUT/Path(src).name)
    a=tr.update(True);b=tr.update();tr.save(OUT/'resume_probe.pt');reference=tr.update()
    expected={k:v.detach().clone() for k,v in tr.net.state_dict().items()}
    expected_d={k:v.detach().clone() for k,v in tr.critic.state_dict().items()}
    fresh=Trainer(cached,selected,args.steps);fresh.restore(OUT/'resume_probe.pt');got=fresh.update()
    error=max(float((v-expected[k]).abs().max()) for k,v in fresh.net.state_dict().items())
    derror=max(float((v-expected_d[k]).abs().max()) for k,v in fresh.critic.state_dict().items())
    assert error<1e-5 and derror<1e-5 and reference['crop']==got['crop']
    write_json(OUT/'preflight.json',dict(status='PASS',native=a,crop=b,resume_error=error,critic_resume_error=derror))
    print('TEXTURE_PREFLIGHT_PASS',error,derror,flush=True)
    del tr,fresh;torch.cuda.empty_cache()
    # Inter-op thread count can only be configured once per process.
    torch.manual_seed(colour.SEED);torch.cuda.manual_seed_all(colour.SEED)
    tr=Trainer(cached,selected,args.steps);tr.net.load_state_dict(initial)
    colour.review(tr,data,selected,OUT/'initial')
    with (OUT/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as f:
        while tr.step<args.steps:
            row=tr.update();f.write(json.dumps(row)+'\n')
            if tr.step%50==0:print('TEXTURE',tr.step,round(row['loss'],4),round(row['discriminator'],4),round(row['adversarial'],4),flush=True)
            if tr.step%300==0 or tr.step==args.steps:
                tr.save(OUT/f'checkpoint_{tr.step:06d}.pt')
                colour.review(tr,data,selected,OUT/f'step_{tr.step:06d}')
    print('TEXTURE_COMPLETE',flush=True)


if __name__=='__main__':main()
