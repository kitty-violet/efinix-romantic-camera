"""Keep nineteen levels, train actual RGB edges and conditional chroma fidelity.

Everything added here is a training loss or a coefficient initialization. The
exported inference graph remains B1; it has no output sharpening/filter stage.
"""
import argparse
import copy
import json
import shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from . import train_b1_19_levels as n
from . import train_b1_learned_colour as c
from . import train_b1_sampling_colour_v2 as base
from .probe_b1_internal_blur import modify
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible

OUT=n.OUT/'crisp-edges'
PARENT=n.OUT/'chroma-calibration/checkpoint_calibrated.pt'


def target_rgb(t):
    """Frozen nineteen-value colour representation, in signed half-RGB units."""
    y=8+sum(t[:,k:k+1]*(inc/127) for k,inc in enumerate(n.INCREMENTS))
    co,cg=t[:,6:7]-64,t[:,7:8]-64
    return torch.cat((2*(y+co-cg)-128,2*(y+cg)-128,2*(y-co-cg)-128),1).clamp(-128,127)/128


class Trainer(n.Trainer):
    def __init__(self,cache,selected,steps,qat=False):
        super().__init__(cache,selected,steps,True)
        self.net.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=True)['model'])
        with torch.no_grad():
            modify(self.net,('up1_high_dw','up2_low_dw','up2_high_dw'),.7)
            self.net.weight['up2_high_dw'][:6]*=127/3
            self.net.weight['up2_high_dw'][6:]*=2
            self.net.weight['rgb_head'][:,:6]/=127/3
            self.net.weight['rgb_head'][:,6:]/=2
        self.net.quantized=False
        if qat:self.net=n.quantize(self.net)
        self.new_optimizer()

    def colour_loss(self,f,t):
        # Neutral teacher regions must not acquire green/magenta. This adds
        # supervised weight, rather than suppressing genuine foliage greens.
        err=(f[:,6:]-t[:,6:])/16
        neutral=((t[:,6:]-64).square().sum(1,keepdim=True)<25).float()
        weight=1+2*neutral
        point=(F.smooth_l1_loss(err,torch.zeros_like(err),beta=.25,reduction='none')*weight).mean()
        # A signed low-frequency error catches a colour cast even if positive
        # and negative local edges have similar average absolute loss.
        mean=F.avg_pool2d(err,16,16)
        return point+.5*mean.abs().mean()

    def appearance_loss(self,nodes,target):
        p=nodes['rgb_head'][:,:,40:-40,40:-40]/128
        t=target_rgb(target)[:,:,40:-40,40:-40]
        lp=(p[:,0:1]+2*p[:,1:2]+p[:,2:3])/4
        lt=(t[:,0:1]+2*t[:,1:2]+t[:,2:3])/4
        edges=[]
        # Teacher edge within two pixels: sharpen the actual output edge, but
        # do not require a subpixel-perfect alignment after two stride-2 layers.
        for dim in (-1,-2):
            dp,dt=torch.diff(lp,dim=dim),torch.diff(lt,dim=dim)
            for sign in (-1,1):
                gp,gt=F.relu(sign*dp),F.relu(sign*dt)
                near_p=F.max_pool2d(gp,5,1,2)
                near_t=F.max_pool2d(gt,5,1,2)
                mask=(gt>.055).float()
                recall=(F.relu(gt-near_p)*mask).sum()/mask.sum().clamp_min(1)
                # No reward for sharpening grain that the teacher removed.
                precision=F.relu(gp-near_t-.02).mean()
                edges.append(recall+2*precision)
        chroma_p=torch.cat(((p[:,0:1]-p[:,2:3])/4,(-p[:,0:1]+2*p[:,1:2]-p[:,2:3])/8),1)
        chroma_t=torch.cat(((t[:,0:1]-t[:,2:3])/4,(-t[:,0:1]+2*t[:,1:2]-t[:,2:3])/8),1)
        return 3*torch.stack(edges).mean()+.3*F.l1_loss(p,t)+F.l1_loss(chroma_p,chroma_t)


def preflight(cache,selected,steps,path,qat=False,state=None):
    tr=Trainer(cache,selected,steps,qat)
    if state is not None:tr.net.load_state_dict(state)
    native=tr.update(True);crop=tr.update();tr.save(path/'resume_probe.pt');ref=tr.update()
    expected={k:v.detach().clone() for k,v in tr.net.state_dict().items()}
    fresh=Trainer(cache,selected,steps,qat);fresh.restore(path/'resume_probe.pt');got=fresh.update()
    error=max(float((v-expected[k]).abs().max()) for k,v in fresh.net.state_dict().items())
    assert error<1e-5 and ref['crop']==got['crop'] and ref['samples']==got['samples']
    write_json(path/'preflight.json',dict(status='PASS',native=native,pair_crop=crop,resume_error=error,quantized=qat))
    print('CRISP_PREFLIGHT_PASS',qat,error,flush=True)
    del tr,fresh;torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=800);ap.add_argument('--qat',type=int,default=1200);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100533);contract_guard();OUT.mkdir(exist_ok=False)
    data,selected,cache=c.load_data()
    write_json(OUT/'binding.json',dict(parent=str(PARENT),parent_sha256=file_hash(PARENT),parent_cumulative_steps=13700,
        new_fp32_steps=a.steps,new_qat_steps=a.qat,luminance_levels=19,actual_B1=True,external_inference=False,
        changes='decoder low-pass initialization .7; actual RGB signed-edge recall within2px; penalty for invented edges; neutral and low-frequency chroma loss; preserve temporal training',
        codes={str(p):file_hash(p) for p in [Path(__file__),Path(n.__file__),Path(base.__file__)]}))
    for p in (__file__,n.__file__,base.__file__):shutil.copy2(p,OUT/Path(p).name)
    preflight(cache,selected,a.steps,OUT);tr=Trainer(cache,selected,a.steps)
    with (OUT/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        while tr.step<a.steps:
            row=tr.update();log.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('CRISP_FP32',tr.step,round(row['loss'],4),round(row['appearance'],4),flush=True)
        tr.save(OUT/f'checkpoint_{tr.step:06d}.pt');n.review(tr,data,selected,OUT/f'step_{tr.step:06d}')
        q=n.quantize(tr.net);state=copy.deepcopy(q.state_dict());del tr,q;torch.cuda.empty_cache()
        qdir=OUT/'qat';qdir.mkdir();preflight(cache,selected,a.qat,qdir,True,state)
        tr=Trainer(cache,selected,a.qat,True);tr.net.load_state_dict(state)
        while tr.step<a.qat:
            row=tr.update();log.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('CRISP_QAT',tr.step,round(row['loss'],4),round(row['appearance'],4),flush=True)
        tr.save(qdir/f'checkpoint_{tr.step:06d}.pt');n.review(tr,data,selected,qdir/'package')
    print('CRISP_COMPLETE',flush=True)


if __name__=='__main__':main()
