"""Learn luminance regions while preserving teacher chroma; unchanged B1.

Six cumulative masks define seven value planes; the other two final channels
carry coarse YCoCg chroma. This avoids reducing all hues to 45 fixed colours.
The last DW/RQ and RGB PW implement the complete deployment conversion.
"""
import argparse
import copy
import json
import math
import random
import shutil
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_learned_colour as colour
from . import train_b1_live_rank_features as live
from .seed_b1_rank_blocks import OUT as SEEDS
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible

OUT=SEEDS/'luma-regions'
THRESHOLDS=(16,32,48,64,80,104)
INCREMENTS=(16,16,16,16,20,24)


def target_features(t):
    v=(t+1)*64;r,g,b=v[:,0:1],v[:,1:2],v[:,2:3]
    y=(r+2*g+b)/4;co=(r-b)/2+64;cg=(-r+2*g-b)/4+64
    return torch.cat([(y>v).float()*127 for v in THRESHOLDS]+[co,cg],1)


@torch.no_grad()
def initialize(net):
    live.activate(net)
    w,b=net.weight['up2_low_pw'],net.bias['up2_low_pw'];w.zero_();b.zero_()
    for c,threshold in enumerate(THRESHOLDS):w[c,0,0,0]=1;b[c]=63.5-threshold
    w[6,1,0,0]=1;w[7,2,0,0]=1
    w,b=net.weight['up2_high_dw'],net.bias['up2_high_dw'];w.zero_();b.zero_()
    w[:,0]=torch.tensor([[0,1,0],[1,4,1],[0,1,0]],device=w.device)/8
    w,b=net.weight['rgb_head'],net.bias['rgb_head'];w.zero_()
    matrix=torch.tensor([[1,1,-1],[1,0,1],[1,-1,-1]],device=w.device,dtype=w.dtype)
    for c in range(3):
        w[c,:6,0,0]=torch.tensor(INCREMENTS,device=w.device)*2/127
        w[c,6,0,0]=2*matrix[c,1];w[c,7,0,0]=2*matrix[c,2]
        b[c]=16-128-128*(matrix[c,1]+matrix[c,2])
    for k in ('up2_high_dw','rgb_head'):
        net.weight[k].requires_grad_(False);net.bias[k].requires_grad_(False)


@torch.no_grad()
def integer_version(net):
    if net.quantized:return copy.deepcopy(net)
    q=copy.deepcopy(net)
    q.weight['up2_high_dw'][:6].div_(127);q.weight['up2_high_dw'][6:].div_(2)
    q.weight['rgb_head'][:,:6].mul_(127);q.weight['rgb_head'][:,6:].mul_(2)
    q.calibrate();q.quantized=True
    return q


def soft_features(net,nodes):
    if not net.quantized:return nodes['up2_high_dw']
    kernel=torch.tensor([[0,1,0],[1,4,1],[0,1,0]],device='cuda',dtype=torch.float32)[None,None].repeat(8,1,1,1)/8
    return F.conv2d(F.pad(nodes['up2_nn'],(1,1,1,1),mode='replicate'),kernel,groups=8)


class Trainer(colour.Trainer):
    def __init__(self,cache,selected,steps):
        super().__init__(cache,selected,'luma_regions',steps)
        self.net=colour.ColourB1(revive=False,source=SEEDS/'blocks/coefficients-padded.bin').cuda().eval();initialize(self.net)
        self.rng=random.Random(2026100524);self.training=[v for v in cache if v['row']['split']=='train']
        self.targets={v['row']['id']:target_features(v['target'][None]) for v in cache}
        self.new_optimizer()

    def new_optimizer(self):
        groups=[]
        for kind,params in [('w',self.net.weight),('b',self.net.bias)]:
            for key,p in params.items():
                if p.requires_grad:
                    lr=(.0002 if kind=='w' else .015)*(.25 if self.net.quantized else 1)
                    groups.append(dict(params=[p],lr=lr,initial_lr=lr))
        self.opt=torch.optim.Adam(groups)

    def update(self,native=False):
        start=time.perf_counter();row=self.rng.choice(self.training)
        if native:x0=y0=0;x1=1280;y1=720
        else:
            x0=self.rng.randrange(241)*4;y0=self.rng.randrange(101)*4;x1=x0+320;y1=y0+320
        x=row['source'][None,:,y0:y1,x0:x1];t=self.targets[row['row']['id']][:,:,y0:y1,x0:x1]
        nodes=self.net(x,all_nodes=True);feature=soft_features(self.net,nodes);cut=lambda v:v[:,:,40:-40,40:-40]
        feature0=cut(feature);label=cut(t);logits=(feature0[:,:6]-63.5)/6
        # Unweighted BCE keeps the correct class prior/0.5 decision threshold.
        bce=F.binary_cross_entropy_with_logits(logits,label[:,:6]/127)
        chroma=F.smooth_l1_loss(feature0[:,6:]/32,label[:,6:]/32)
        prob=logits.sigmoid();inside=[]
        for d in (-1,-2):
            dp,dt=torch.diff(prob,dim=d),torch.diff(label[:,:6],dim=d)
            inside.append((dp.abs()*(dt==0)).mean())
        region=torch.stack(inside).mean();loss=bce+2*chroma+region
        phase=loss.new_zeros(())
        if self.step%4==0 and not native:
            shifted=F.pad(x,(1,0,0,0),mode='replicate')[:,:,:,:x.shape[-1]]
            ns=self.net(shifted,all_nodes=True);fs=soft_features(self.net,ns)
            phase=F.l1_loss(fs[:,:,40:-40,41:-40]/127,feature[:,:,40:-40,40:-41]/127);loss+=.3*phase
        for g in self.opt.param_groups:g['lr']=g['initial_lr']*(.5+.5*.5*(1+math.cos(math.pi*min(self.step,self.steps)/max(1,self.steps))))
        self.opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5,error_if_nonfinite=True)
        assert grad>0 and torch.isfinite(loss);self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),bce=float(bce.detach()),chroma=float(chroma.detach()),
                    region=float(region.detach()),phase=float(phase.detach()),grad=float(grad),crop=[x0,y0,x1,y1],sample=row['row']['id'],
                    integer=self.net.quantized,seconds=time.perf_counter()-start)


@torch.inference_mode()
def review(tr,data,selected,dest):
    saved=tr.net;tr.net=integer_version(saved);colour.review(tr,data,selected,dest,integer=True,oracle_comparison=False)
    board=Image.new('RGB',(1260,1248),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    records=[]
    for i,(key,box,label) in enumerate(colour.CROPS):
        row=next(v for v in tr.cached if v['row']['id']==key);t=tr.targets[key].clone();t[:,:6]/=127;t[:,6:]=(t[:,6:]/2).round()
        target=F.conv2d(t,tr.net.weight['rgb_head'],tr.net.bias['rgb_head']).clamp(-128,127)/128
        nodes=tr.net(row['source'][None],all_nodes=True);p=nodes['rgb_head']/128;gate=nodes['up2_high_dw'][:,:6]
        assert gate.min()>=0 and gate.max()<=1
        for j,(name,img) in enumerate([('冻结教师',colour.rgb_image(row['target'])),('本轮明度分区目标',colour.rgb_image(target[0])),('实际20KiB整数包',colour.rgb_image(p[0]))]):
            board.paste(img.crop(box),(420*j,312*i+32));draw.text((420*j+4,312*i+4),name,font=font,fill='white')
        records.append(dict(id=key,mask_accuracy=float(((gate>0)==(t[:,:6]>0)).float().mean())))
    board.save(dest/'region_targets.png');write_json(dest/'regions.json',dict(records=records,visual_accepted=False));tr.net=saved


def preflight(tr,cache,selected,steps,path):
    native=tr.update(True);crop=tr.update();tr.save(path/'resume_probe.pt');ref=tr.update();expected={k:v.detach().clone() for k,v in tr.net.state_dict().items()}
    fresh=Trainer(cache,selected,steps)
    if tr.net.quantized:fresh.net=integer_version(fresh.net);fresh.new_optimizer()
    fresh.restore(path/'resume_probe.pt');got=fresh.update()
    error=max(float((v-expected[k]).abs().max()) for k,v in fresh.net.state_dict().items());assert error<1e-5 and ref['crop']==got['crop'] and ref['sample']==got['sample']
    write_json(path/'preflight.json',dict(status='PASS',native=native,crop=crop,resume_error=error));print('LUMA_REGIONS_PREFLIGHT_PASS',tr.net.quantized,error,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=3000);ap.add_argument('--qat',type=int,default=500);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100524);contract_guard();OUT.mkdir(exist_ok=False)
    data,selected,cache=colour.load_data();tr=Trainer(cache,selected,a.steps)
    write_json(OUT/'binding.json',dict(steps=a.steps,qat=a.qat,training_ids=[v['row']['id'] for v in tr.training],
        codes={str(p):file_hash(p) for p in [Path(__file__),Path(colour.__file__),Path(live.__file__)]},
        actual_B1=True,external_inference=False,loss='Teacher cumulative luma regions + teacher chroma; no source RGB loss',
        changes='Remove rare-class reweighting; preserve chroma; use 42 train frames instead of four fixed crops'))
    for src in (__file__,colour.__file__,live.__file__):shutil.copy2(src,OUT/Path(src).name)
    preflight(tr,cache,selected,a.steps,OUT);del tr;torch.cuda.empty_cache();tr=Trainer(cache,selected,a.steps)
    with (OUT/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as f:
        while tr.step<a.steps:
            row=tr.update();f.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('LUMA_REGIONS',tr.step,round(row['loss'],4),round(row['bce'],4),flush=True)
            if tr.step%750==0 or tr.step==a.steps:
                tr.save(OUT/f'checkpoint_{tr.step:06d}.pt');review(tr,data,selected,OUT/f'step_{tr.step:06d}')
        parent=copy.deepcopy(tr.net.state_dict());sampler=tr.rng.getstate();qdir=OUT/'qat';qdir.mkdir()
        tr.net=integer_version(tr.net);tr.new_optimizer();qstate=copy.deepcopy(tr.net.state_dict())
        preflight(tr,cache,selected,a.steps,qdir)
        tr=Trainer(cache,selected,a.steps);tr.net=integer_version(tr.net);tr.net.load_state_dict(qstate);tr.new_optimizer();tr.step=a.steps;tr.rng.setstate(sampler)
        while tr.step<a.steps+a.qat:
            row=tr.update();f.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('LUMA_QAT',tr.step,round(row['loss'],4),flush=True)
        tr.save(qdir/f'checkpoint_{tr.step:06d}.pt');review(tr,data,selected,qdir/'package')
    print('LUMA_REGIONS_COMPLETE',flush=True)


if __name__=='__main__':main()
