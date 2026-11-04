"""Learn early contour features inside B1, preserving the RGB surface lanes."""
import argparse
import copy
import json
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import cv2
from . import train_b1_structural_gate as gate
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .engine import set_reproducible

OUT=gate.ROOT/'b1-learned-structural-features-20261005-v2'
PARENT=gate.OUT


class FeatureStudent(gate.Student):
    def __init__(self):
        super().__init__()
        ranges={'stem':(3,8),'down1__dw':(3,8),'down1__pw':(6,16),
                'down2__dw':(6,16),'down2__pw':(6,24)}
        with torch.no_grad():
            # Five nonzero seed features; signed differences are centred in the
            # ReLU range. Initial RGB carrier remains exact and frozen.
            w=self.weight['stem'];b=self.bias['stem'];m=self.stem_m
            y=torch.tensor([1.,2.,1.],device=w.device)
            for c in range(3,8):w[c].zero_();b[c]=512;m[0,c,0,0]=16384
            w[3,:,1,0]=-y;w[3,:,1,2]=y
            w[4,:,0,1]=-y;w[4,:,2,1]=y
            w[5,:,0,0]=-y;w[5,:,2,2]=y
            w[6,:,1,1]=y
            w[7,:,0,1]=-y;w[7,:,2,1]=-y;w[7,:,1,1]=2*y
            # The original colour-only initializer zeroed these depthwise
            # rows. A zero filter followed by ReLU disconnects all five new
            # stem features and has zero gradient at its zero output.
            self.weight['down1__dw'][3:8].zero_()
            self.weight['down1__dw'][3:8,0,1,1]=2
            self.bias['down1__dw'][3:8].zero_()
            self.down1__dw_m[:,3:8]=65536
            for key,(lo,hi) in ranges.items():
                self.weight[key].mul_(8);self.bias[key].mul_(8)
                mult=getattr(self,key+'_m');mult.div_(8)
                assert bool((mult>=1).all()) and self.weight[key].abs().max()<=127
                self.package['parameters'][key.replace('__','.')]['multipliers']=mult.flatten().to(torch.int64).tolist()
                for kind,params in [('w',self.weight),('b',self.bias)]:
                    value=params[key];value.requires_grad_(True);mask=torch.zeros_like(value);mask[lo:hi]=1
                    self.register_buffer(kind+'_'+key+'_mask',mask)
                    self.register_buffer(kind+'_'+key+'_initial',value.detach().clone())
                    self.masks[(kind,key)]=True


class FeatureTraining(gate.Training):
    def __init__(self,data):
        super().__init__(data)
        self.net=FeatureStudent().cuda().eval()
        self.opt=torch.optim.Adam([p for p in self.net.parameters() if p.requires_grad],lr=.035)

    def update(self,native=False):
        started=time.perf_counter();r,rgb,label=self.rng.choice(self.cache);h,w=rgb.shape[:2];size=320
        y=0 if native else self.rng.randrange(0,h-size+1);x=0 if native else self.rng.randrange(0,w-size+1)
        hh,ww=(h,w) if native else (size,size)
        source=torch.from_numpy(rgb[y:y+hh,x:x+ww].copy().transpose(2,0,1)).float()[None].cuda()/128-1
        target=torch.from_numpy(label[y:y+hh,x:x+ww].copy())[None,None].cuda()[:,:,::4,::4]
        p=self.net(source)[:,:,6:-6,6:-6];t=target[:,:,6:-6,6:-6];pos=t>.12;neg=~pos
        desired=t*32-8
        error=F.smooth_l1_loss(p/16,desired/16,reduction='none')
        positive=(error*pos).sum()/pos.sum().clamp_min(1);negative=(error*neg).sum()/neg.sum().clamp_min(1)
        false_response=(F.relu(p+2)/16*neg).sum()/neg.sum().clamp_min(1)
        loss=positive+1.5*negative+.4*false_response
        self.opt.zero_grad(set_to_none=True);loss.backward()
        for kind,key in self.net.masks:
            value=(self.net.weight if kind=='w' else self.net.bias)[key]
            if value.grad is not None:value.grad.mul_(getattr(self.net,kind+'_'+key+'_mask'))
        norm=torch.nn.utils.clip_grad_norm_([p for p in self.net.parameters() if p.requires_grad],10,error_if_nonfinite=True)
        assert torch.isfinite(loss) and norm>0
        self.opt.step();self.net.project();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),positive=float(positive.detach()),negative=float(negative.detach()),
            false_response=float(false_response.detach()),grad=float(norm),sample=r['id'],crop=[x,y,ww,hh],seconds=time.perf_counter()-started,quantized=self.net.quantized)

    def save(self,path):
        super().save(path)
        state=torch.load(path,map_location='cpu',weights_only=True);state['feature_code_sha256']=file_hash(__file__);torch.save(state,path)

    def restore(self,path):
        state=torch.load(path,map_location='cpu',weights_only=True);assert state['feature_code_sha256']==file_hash(__file__)
        super().restore(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['preflight','train']);args=ap.parse_args()
    configure_workspace();set_reproducible(2026100506);cv2.setNumThreads(2);contract_guard();gate.OUT=OUT
    if args.mode=='preflight':
        OUT.mkdir(exist_ok=False);shutil.copy2(PARENT/'data.json',OUT/'data.json')
        (OUT/'labels').mkdir()
        for r in json.loads((OUT/'data.json').read_text()):
            if r['split']!='train':shutil.copy2(PARENT/'labels'/(r['id']+'_thin.png'),OUT/'labels'/(r['id']+'_thin.png'))
    data=json.loads((OUT/'data.json').read_text(encoding='utf-8'))
    for r in data:
        for key in ('source','target','label'):assert file_hash(r[key])==r[key+'_sha256']
    tr=FeatureTraining(data)
    if args.mode=='preflight':
        dest=OUT/'preflight';dest.mkdir();a=tr.update(True);b=tr.update();tr.save(dest/'resume.pt');ref=tr.update()
        state={k:v.clone() for k,v in tr.net.state_dict().items()};tr.restore(dest/'resume.pt');got=tr.update()
        error=max(float((v-state[k]).abs().max()) for k,v in tr.net.state_dict().items());assert error<1e-6 and ref['crop']==got['crop']
        tr.net.quantized=True;q=tr.update();gate.evaluate(tr,data,'preflight-export')
        write_json(dest/'result.json',dict(status='PASS',native=a,crop=b,qat=q,resume_error=error,
            code_sha256=file_hash(__file__),gate_code_sha256=file_hash(gate.__file__)))
        print('FEATURE_PREFLIGHT_PASS',error,flush=True);return
    pre=json.loads((OUT/'preflight/result.json').read_text());assert pre['status']=='PASS' and pre['code_sha256']==file_hash(__file__) and pre['gate_code_sha256']==file_hash(gate.__file__)
    for path in [__file__,gate.__file__]:shutil.copy2(path,OUT/Path(path).name)
    write_json(OUT/'config.json',dict(steps=2400,fp32_steps=1600,qat_steps=800,lr=.035,qat_lr=.018,
        trainable='Early reserved feature channels and res1 gate, original RGB lanes frozen',
        deployed='Same B1 graph; direct learned gate controls res2/res3; no extra inference operators'))
    with (OUT/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        while tr.step<2400:
            tr.net.quantized=tr.step>=1600
            lr=.035 if not tr.net.quantized else .018
            for group in tr.opt.param_groups:group['lr']=lr
            row=tr.update((tr.step+1)%100==0);log.write(json.dumps(row)+'\n')
            if tr.step%50==0:print('LEARN_FEATURES',tr.step,round(row['loss'],4),round(row['seconds'],3),'QAT',tr.net.quantized,flush=True)
            if tr.step%800==0:tr.save(OUT/f'checkpoint_{tr.step:06d}.pt');gate.evaluate(tr,data,f'step_{tr.step:06d}')


if __name__=='__main__':main()
