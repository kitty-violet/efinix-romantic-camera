"""Recover decoder headroom and directly supervise its real 360p sample grid."""
import argparse
import copy
import json
import shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from . import train_b1_sampling_colour_v2 as base
from . import train_b1_learned_colour as c
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible

PARENT=base.OUT/'learned/qat/checkpoint_000800.pt'
OUT=base.OUT/'lattice-recovery'


class Trainer(base.Trainer):
    def __init__(self,cache,selected,steps,qat=False):
        super().__init__(cache,selected,steps,True)
        self.net.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=True)['model'])
        with torch.no_grad():
            self.net.weight['up2_high_dw'][:6]*=127/2
            self.net.weight['up2_high_dw'][6:]*=2
            self.net.weight['rgb_head'][:,:6]/=127/2
            self.net.weight['rgb_head'][:,6:]/=2
            # Homogeneous reparameterization of the three hidden decoder layers.
            # It preserves the unclipped FP32 mapping but opens formerly clipped
            # activations; therefore this intervention requires actual retraining.
            self.net.weight['up1_low_pw']*=.5;self.net.bias['up1_low_pw']*=.5
            self.net.bias['up1_high_dw']*=.5;self.net.bias['up2_low_dw']*=.5
            self.net.weight['up2_low_pw']*=2
            # Match stride-2 sample coordinates: even output pixels correspond
            # to the low-grid sample, odd pixels interpolate adjacent samples.
            self.net.weight['up2_high_dw'][:,0]=self.net.weight['up2_high_dw'].new_tensor([[0,0,0],[0,.25,.25],[0,.25,.25]])
        self.net.quantized=False
        if qat:self.net=base.quantize(self.net)
        self.new_optimizer()
        for g in self.opt.param_groups:g['lr']*=1.5;g['initial_lr']*=1.5

    def colour_loss(self,f,t):
        # Old /32 beta=1 Huber loss nearly ignored a persistent +3/-1 chroma bias.
        return F.smooth_l1_loss(f[:,6:]/16,t[:,6:]/16,beta=.25)

    def lattice_loss(self,nodes,t):
        f=nodes['up2_low_pw'][:,:,20:-20,20:-20]
        label=t[:,:,::2,::2][:,:,20:-20,20:-20]
        truth=torch.stack([(label[:,:6]>0).float(),(label[:,:6]>64).float()],2).flatten(1,2)
        return .5*F.binary_cross_entropy_with_logits(base.ordinal(f),truth)+.5*self.colour_loss(f,label)


def preflight(cache,selected,steps,path,qat=False,state=None):
    tr=Trainer(cache,selected,steps,qat)
    if state is not None:tr.net.load_state_dict(state)
    native=tr.update(True);crop=tr.update();tr.save(path/'resume_probe.pt');ref=tr.update()
    expected={k:v.detach().clone() for k,v in tr.net.state_dict().items()}
    fresh=Trainer(cache,selected,steps,qat);fresh.restore(path/'resume_probe.pt');got=fresh.update()
    error=max(float((v-expected[k]).abs().max()) for k,v in fresh.net.state_dict().items())
    assert error<1e-5 and ref['crop']==got['crop'] and ref['samples']==got['samples']
    write_json(path/'preflight.json',dict(status='PASS',native=native,real_pair_crop=crop,resume_error=error,quantized=qat))
    print('LATTICE_PREFLIGHT_PASS',qat,error,flush=True)
    del tr,fresh;torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=1600);ap.add_argument('--qat',type=int,default=800);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100529);contract_guard();OUT.mkdir(exist_ok=False)
    data,selected,cache=c.load_data()
    write_json(OUT/'binding.json',dict(parent=str(PARENT),parent_sha256=file_hash(PARENT),parent_cumulative_steps=8100,
        new_fp32_steps=a.steps,new_qat_steps=a.qat,actual_B1=True,external_inference=False,
        changes='Decoder hidden activation headroom; 360-grid ordinal supervision; phase-aligned last kernel; stronger chroma Huber',
        codes={str(p):file_hash(p) for p in [Path(__file__),Path(base.__file__),Path(c.__file__)]}))
    for p in (__file__,base.__file__):shutil.copy2(p,OUT/Path(p).name)
    preflight(cache,selected,a.steps,OUT)
    tr=Trainer(cache,selected,a.steps)
    with (OUT/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        while tr.step<a.steps:
            row=tr.update();log.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('LATTICE_FP32',tr.step,round(row['loss'],4),round(row['chroma'],5),flush=True)
            if tr.step==a.steps:
                tr.save(OUT/f'checkpoint_{tr.step:06d}.pt');base.review(tr,data,selected,OUT/f'step_{tr.step:06d}')
        q=base.quantize(tr.net);state=copy.deepcopy(q.state_dict());del q,tr;torch.cuda.empty_cache()
        qdir=OUT/'qat';qdir.mkdir();preflight(cache,selected,a.qat,qdir,True,state)
        tr=Trainer(cache,selected,a.qat,True);tr.net.load_state_dict(state)
        while tr.step<a.qat:
            row=tr.update();log.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('LATTICE_QAT',tr.step,round(row['loss'],4),flush=True)
        tr.save(qdir/f'checkpoint_{tr.step:06d}.pt');base.review(tr,data,selected,qdir/'package')
    print('LATTICE_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()
