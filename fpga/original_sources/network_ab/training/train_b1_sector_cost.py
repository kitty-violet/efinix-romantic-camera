"""Learn texture selection inside a fixed, hand-designed block-forming B1.

Only paired zero-DC texture filters are trained. RGB averages, finite colours,
spatial voting, the decoder and all topology remain fixed. Teacher regions
provide training-only mean source colours to rank the four candidate sectors.
"""
import argparse
import copy
import json
import random
import shutil
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from . import train_b1_region_formation as formation
from .runtime import configure_workspace,write_json,file_hash,contract_guard
from .engine import set_reproducible


@lru_cache(maxsize=16)
def sample(path,region):
    rgb=m.t.r.read(path);labels=m.load_region(region);n=int(labels.max())+1
    count=np.bincount(labels.ravel(),minlength=n).clip(1)
    means=np.stack([np.bincount(labels.ravel(),weights=rgb[...,c].ravel(),minlength=n)/count for c in range(3)],1)
    return rgb,(means[labels]/2).astype(np.float32)


class Trainer:
    def __init__(self,args,data):
        self.args=args;self.data=data;self.rng=random.Random(args.seed);self.step=0
        self.parent=torch.load(args.warm,map_location='cpu',weights_only=True);self.net=m.t.r.initialize()
        self.net.load_state_dict(self.parent['model']);self.net.package['residual_encoding']=self.parent['residual_encoding'];self.net.quantized=True
        self.net.requires_grad_(False);self.net.weight['stem'].requires_grad_(True)
        with torch.no_grad():
            # A second zero-DC opponent filter provides chromatic boundaries.
            kernel=self.net.weight['stem'].new_tensor([[0,-1,0],[-1,4,-1],[0,-1,0]])/8
            self.net.weight['stem'][5]=kernel[None]*self.net.weight['stem'].new_tensor([.5,-1,.5])[:,None,None]
            self.net.weight['stem'][6]=-self.net.weight['stem'][5]
            self.net.stem_m[:,3:7]=512
            blur=kernel.new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
            self.net.weight['down1__dw'][5:7,0]=blur;self.net.down1__dw_m[:,5:7]=512
            for sector in range(4):self.net.weight['down1__pw'][sector*4+3,5:7,0,0]=1
        self.fixed={k:v.detach().clone() for k,v in self.net.state_dict().items()}
        self.opt=torch.optim.Adam([self.net.weight['stem']],lr=.0003)

    def save(self,path):
        state=dict(self.parent);state.update(model=self.net.state_dict(),sector_cost_trained=True,step=self.step,
            optimizer=self.opt.state_dict(),rng=self.rng.getstate(),parent=self.args.warm,qat=True)
        torch.save(state,path)

    def restore(self,path):
        state=torch.load(path,map_location='cpu',weights_only=True);self.net.load_state_dict(state['model'])
        self.opt.load_state_dict(state['optimizer']);self.rng.setstate(state['rng']);self.step=state['step']

    def update(self,native=False):
        rows=[self.rng.choice(self.data['training']) for _ in range(1 if native else 2)];xs=[];ts=[];specs=[]
        for row in rows:
            rgb,target=sample(row['path'],row['region']);w,h=(1280,720) if native else (320,320)
            x,y=(0,0) if native else (self.rng.randrange((1280-w)//4+1)*4,self.rng.randrange((720-h)//4+1)*4)
            xs.append(m.t.r.tensor(rgb[y:y+h,x:x+w]));ts.append(torch.from_numpy(target[y:y+h,x:x+w].copy()).permute(2,0,1).cuda())
            specs.append((row['id'],x,y,w,h))
        x=torch.stack(xs);truth=F.avg_pool2d(torch.stack(ts),4)
        nodes=self.net(x,all_nodes=True);z=nodes['res1_conv1.dw'];colours=torch.stack([z[:,4*s:4*s+3] for s in range(4)],1)
        costs=z[:,3:16:4];margin=10 if not native else 4
        colours=colours[...,margin:-margin,margin:-margin];costs=costs[...,margin:-margin,margin:-margin];truth=truth[...,margin:-margin,margin:-margin]
        errors=(colours-truth[:,None]).square().mean(2).detach()
        wanted=F.softmax(-errors/8,1);logp=F.log_softmax(-costs/2,1)
        loss=-(wanted*logp).sum(1).mean()
        self.opt.zero_grad(set_to_none=True);loss.backward();grad=self.net.weight['stem'].grad
        grad[:3]=0;grad[7:]=0
        assert torch.isfinite(loss) and grad.norm()>0
        torch.nn.utils.clip_grad_norm_([self.net.weight['stem']],1,error_if_nonfinite=True);self.opt.step()
        with torch.no_grad():
            w=self.net.weight['stem']
            for a,b in ((3,4),(5,6)):
                k=(w[a]-w[b])/2;k-=k.mean((-1,-2),keepdim=True);k.clamp_(-.45,.45)
                w[a]=k;w[b]=-k
            assert torch.equal(w[:3],self.fixed['weight.stem'][:3]) and torch.equal(w[7:],self.fixed['weight.stem'][7:])
        self.step+=1
        return dict(step=self.step,loss=float(loss.detach()),specs=specs,native720=native,actual_selection_error=float(errors.gather(1,costs.argmin(1,keepdim=True)).mean()),
            best_available_error=float(errors.min(1).values.mean()))

    def check_fixed(self):
        for key,value in self.net.state_dict().items():
            if key!='weight.stem':assert torch.equal(value,self.fixed[key]),key


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True);ap.add_argument('--steps',type=int,default=1000)
    ap.add_argument('--resume-from',default='');ap.add_argument('--seed',type=int,default=20261006149);a=ap.parse_args();configure_workspace();contract_guard();set_reproducible(a.seed);torch.set_num_threads(2)
    dest=m.OUT/a.name;dest.mkdir(exist_ok=False);d=m.load_data();tr=Trainer(a,d)
    native=tr.update(True);crop=tr.update();tr.save(dest/'preflight.pt');expected=tr.update();state=copy.deepcopy(tr.net.state_dict())
    other=Trainer(a,d);other.restore(dest/'preflight.pt');actual=other.update()
    error=max(float((state[k]-v).abs().max()) for k,v in other.net.state_dict().items());assert error<1e-6 and actual['specs']==expected['specs']
    tr.check_fixed();write_json(dest/'preflight.json',dict(status='PASS',native=native,crop=crop,resume_error=error));print('SECTOR_TRAIN_PREFLIGHT',error,flush=True)
    del tr,other;torch.cuda.empty_cache();tr=Trainer(a,d)
    if a.resume_from:tr.restore(a.resume_from)
    started=time.perf_counter();history=[];best=float('inf');stop=None
    paths=[Path(__file__),Path(m.t.r.c.__file__)];write_json(dest/'config.json',dict(args=vars(a),source_hashes={str(p):file_hash(p) for p in paths},
        data_sha256=file_hash(m.OUT/'data.json'),warm_sha256=file_hash(a.warm),precision='FP32_integer_forward_no_AMP_TF32',
        learned='two paired zero-DC3x3 texture filters only',added_operators=0,postfilter=False,visual_acceptance=False))
    def assess():
        nonlocal best
        e=formation.evaluate(tr.net,d['validation']);history.append(dict(step=tr.step,validation=e));tr.check_fixed()
        tr.save(dest/f'candidate_{tr.step:06d}.pt');m.t.preview(tr.net,d['validation'],dest/f'preview_{tr.step:06d}')
        if e['mean']['score']<best:best=e['mean']['score'];tr.save(dest/'best.pt')
        write_json(dest/'history.json',history);print('SECTOR_TRAIN_EVAL',tr.step,e['mean'],flush=True)
    assess()
    with (dest/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as f:
        while tr.step<a.steps:
            row=tr.update(native=(tr.step+1)%200==0);f.write(json.dumps(row)+'\n')
            request=dest/'STOP_REQUEST.json'
            if request.exists():
                stop=json.loads(request.read_text(encoding='utf-8-sig'));assert stop.get('reason')
                tr.save(dest/'last.pt');assess();break
            if tr.step%100==0:print('SECTOR_TRAIN_STEP',row,flush=True)
            if tr.step%200==0 or tr.step==a.steps:
                assess()
    for p in paths:shutil.copy2(p,dest/p.name)
    write_json(dest/'completion.json',dict(executed_steps=tr.step,elapsed_seconds=time.perf_counter()-started,stop_evidence=stop,visual_acceptance=False))
    print('SECTOR_TRAIN_COMPLETE',tr.step,flush=True)


if __name__=='__main__':main()
