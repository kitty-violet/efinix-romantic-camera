"""Learn cumulative colour-region masks; integer requantization selects blocks.

The unchanged final DW/RQ rounds eight nonnegative bounded feature maps to
0/1. The ordinary RGB 1x1 weights map them to YCoCg levels. No external mask,
palette lookup, threshold operator or postfilter runs at inference.
"""
import argparse
import copy
import json
import math
import random
import shutil
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch.nn import functional as F
from . import train_b1_learned_colour as colour
from . import train_b1_live_rank_features as live
from .seed_b1_rank_blocks import OUT as SEEDS
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible

OUT=SEEDS/'colour-regions'


def target_masks(t):
    v=(t+1)*64;r,g,b=v[:,0:1],v[:,1:2],v[:,2:3]
    y=(r+2*g+b)/4;co=(r-b)/2+64;cg=(-r+2*g-b)/4+64
    return torch.cat([y>v for v in (24,48,72,96)]+[co>v for v in (57,71)]+[cg>v for v in (57,71)],1).float()


@torch.no_grad()
def initialize(net):
    live.activate(net)
    # Features up to the last PW retain the nonlinear median/spatial seed.
    w,b=net.weight['up2_low_pw'],net.bias['up2_low_pw'];w.zero_();b.zero_()
    for c,(src,threshold) in enumerate([(0,v) for v in (24,48,72,96)]+[(1,57),(1,71),(2,57),(2,71)]):
        w[c,src,0,0]=1;b[c]=63.5-threshold
    # Positive normalized kernel bounds the pre-rounded feature to [0,127].
    w,b=net.weight['up2_high_dw'],net.bias['up2_high_dw'];w.zero_();b.zero_()
    w[:,0]=torch.tensor([[0,1,0],[1,4,1],[0,1,0]],device=w.device)/8
    w,b=net.weight['rgb_head'],net.bias['rgb_head'];w.zero_()
    matrix=torch.tensor([[1,1,-1],[1,0,1],[1,-1,-1]],device=w.device,dtype=w.dtype)
    matrix[:,1:]*=1.08
    for c in range(3):
        w[c,:4,0,0]=2*24*matrix[c,0]/127
        w[c,4:6,0,0]=2*14*matrix[c,1]/127
        w[c,6:8,0,0]=2*14*matrix[c,2]/127
        b[c]=2*(12*matrix[c,0]-14*matrix[c,1]-14*matrix[c,2])-128
    for k in ('up2_high_dw','rgb_head'):
        net.weight[k].requires_grad_(False);net.bias[k].requires_grad_(False)


@torch.no_grad()
def integer_version(net):
    q=copy.deepcopy(net)
    q.weight['up2_high_dw'].div_(127);q.bias['up2_high_dw'].div_(127)
    q.weight['rgb_head'].mul_(127)
    q.calibrate();q.quantized=True
    # Exactly normalized integer last-DW: kernel denominator 8*127=1016.
    # Choose weights [0,16,0;16,64,16;0,16,0], m=129 approximates 1/1016
    # after division by 16; calibrated row also remains nonnegative and bounded.
    return q


class Trainer(colour.Trainer):
    def __init__(self,cache,selected,steps):
        super().__init__(cache,selected,'regions',steps)
        self.net=colour.ColourB1(revive=False,source=SEEDS/'blocks/coefficients-padded.bin').cuda().eval()
        initialize(self.net);self.rng=random.Random(2026100523)
        self.masks={v['row']['id']:target_masks(v['target'][None]) for v in cache}
        groups=[]
        for kind,params in [('w',self.net.weight),('b',self.net.bias)]:
            for key,p in params.items():
                if p.requires_grad:
                    lr=.00025 if kind=='w' else .02
                    groups.append(dict(params=[p],lr=lr,initial_lr=lr))
        self.opt=torch.optim.Adam(groups)

    def update(self,native=False):
        begin=time.perf_counter();s=self.selected[self.step%len(self.selected)]
        row=next(v for v in self.cached if v['row']['id']==s['id']);x0,y0,x1,y1=s['box']
        if not native:
            x0=max(0,min(960,x0+self.rng.randrange(-8,9)*4));y0=max(0,min(400,y0+self.rng.randrange(-8,9)*4));x1=x0+320;y1=y0+320
        else:x0=y0=0;x1=1280;y1=720
        x=row['source'][None,:,y0:y1,x0:x1];t=self.masks[s['id']][:,:,y0:y1,x0:x1]
        nodes=self.net(x,all_nodes=True);feature=nodes['up2_high_dw'];cut=lambda v:v[:,:,40:-40,40:-40]
        z=(cut(feature)-63.5)/6;label=cut(t)
        # Separate positive/negative support per mask; uniform regions cannot
        # dominate by predicting all background. No source-detail reconstruction.
        count=label.mean((0,2,3)).clamp(.03,.97)[None,:,None,None]
        weight=torch.where(label>0,.5/count,.5/(1-count)).clamp_max(8)
        bce=(F.binary_cross_entropy_with_logits(z,label,reduction='none')*weight).mean()
        prob=z.sigmoid();transition=[]
        for d in (-1,-2):
            dp,dt=torch.diff(prob,dim=d),torch.diff(label,dim=d)
            transition.append((dp.abs()*(dt==0)).mean())
        region=torch.stack(transition).mean()
        loss=bce+.8*region
        phase=loss.new_zeros(())
        if self.step%4==0 and not native:
            shifted=F.pad(x,(1,0,0,0),mode='replicate')[:,:,:,:x.shape[-1]]
            fs=self.net(shifted,all_nodes=True)['up2_high_dw']
            phase=F.l1_loss(fs[:,:,40:-40,41:-40]/127,feature[:,:,40:-40,40:-41]/127)
            loss+=.25*phase
        for g in self.opt.param_groups:g['lr']=g['initial_lr']*(.4+.6*.5*(1+math.cos(math.pi*self.step/max(1,self.steps))))
        self.opt.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5,error_if_nonfinite=True);assert grad>0 and torch.isfinite(loss)
        self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),classification=float(bce.detach()),region=float(region.detach()),
                    phase=float(phase.detach()),grad=float(grad),crop=[x0,y0,x1,y1],sample=s['id'],seconds=time.perf_counter()-begin)


@torch.inference_mode()
def review(tr,data,selected,dest):
    saved=tr.net;tr.net=integer_version(saved)
    colour.review(tr,data,selected,dest,integer=True,oracle_comparison=False)
    # Also retain the exact target palette and the predicted region masks.
    board=Image.new('RGB',(1260,4*312),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    records=[]
    for i,(key,box,label) in enumerate(colour.CROPS):
        row=next(v for v in tr.cached if v['row']['id']==key);bits=tr.masks[key]
        target=F.conv2d(bits,tr.net.weight['rgb_head'],tr.net.bias['rgb_head']).clamp(-128,127)/128
        nodes=tr.net(row['source'][None],all_nodes=True);p=nodes['rgb_head']/128
        gate=nodes['up2_high_dw'];assert gate.min()>=0 and gate.max()<=1,(float(gate.min()),float(gate.max()))
        for j,(name,img) in enumerate([('冻结教师',colour.rgb_image(row['target'])),('本轮颜色分区目标',colour.rgb_image(target[0])),('实际20KiB整数包',colour.rgb_image(p[0]))]):
            board.paste(img.crop(box),(420*j,312*i+32));draw.text((420*j+4,312*i+4),name,font=font,fill='white')
        records.append(dict(id=key,mask_accuracy=float(((gate>0)==(bits>0)).float().mean()),gate_min=float(gate.min()),gate_max=float(gate.max())))
    board.save(dest/'region_targets.png');write_json(dest/'regions.json',dict(teacher_reduced_to_45_colour_combinations=True,records=records,accepted=False))
    tr.net=saved


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=3000);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100523);contract_guard();OUT.mkdir(exist_ok=False)
    data,selected,cache=colour.load_data();tr=Trainer(cache,selected,a.steps)
    write_json(OUT/'binding.json',dict(steps=a.steps,selected=selected,seed=str(SEEDS/'blocks/coefficients-padded.bin'),
        codes={str(p):file_hash(p) for p in [Path(__file__),Path(colour.__file__),Path(live.__file__)]},
        objective='Cumulative YCoCg teacher colour masks; balanced BCE, within-region smoothness, phase consistency',
        deployment='Unchanged B1, final DW requantization to binary and ordinary fixed RGB weights',actual_board=False))
    for src in (__file__,colour.__file__,live.__file__):shutil.copy2(src,OUT/Path(src).name)
    native=tr.update(True);crop=tr.update();tr.save(OUT/'resume_probe.pt');ref=tr.update();expected={k:v.detach().clone() for k,v in tr.net.state_dict().items()}
    fresh=Trainer(cache,selected,a.steps);fresh.restore(OUT/'resume_probe.pt');got=fresh.update()
    error=max(float((v-expected[k]).abs().max()) for k,v in fresh.net.state_dict().items());assert error<1e-5 and ref['crop']==got['crop']
    write_json(OUT/'preflight.json',dict(status='PASS',native=native,crop=crop,disk_resume_error=error));print('REGIONS_PREFLIGHT_PASS',error,flush=True)
    del tr,fresh;torch.cuda.empty_cache();tr=Trainer(cache,selected,a.steps);review(tr,data,selected,OUT/'initial')
    with (OUT/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as f:
        while tr.step<a.steps:
            row=tr.update();f.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('REGIONS',tr.step,round(row['loss'],4),round(row['classification'],4),flush=True)
            if tr.step%750==0 or tr.step==a.steps:
                tr.save(OUT/f'checkpoint_{tr.step:06d}.pt');review(tr,data,selected,OUT/f'step_{tr.step:06d}')
    print('REGIONS_COMPLETE',flush=True)


if __name__=='__main__':main()
