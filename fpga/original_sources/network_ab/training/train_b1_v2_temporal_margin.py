"""Stabilize the extra region levels at the actual integer decision thresholds."""
import argparse
import json
import shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from . import train_b1_lattice_recovery as recovery
from . import train_b1_sampling_colour_v2 as base
from . import train_b1_learned_colour as c
from .coherent_shared_model import warp
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible

PARENT=recovery.OUT/'qat/checkpoint_000800.pt'
OUT=base.OUT/'temporal-margin'


class Trainer(recovery.Trainer):
    temporal_weight=5
    def __init__(self,cache,selected,steps):
        super().__init__(cache,selected,steps,True)
        self.net.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=True)['model'])
        self.new_optimizer()

    def stability_margin(self,logits,truth):
        maximum=F.max_pool2d(truth,7,1,3);minimum=-F.max_pool2d(-truth,7,1,3)
        interior=(maximum==minimum).float()
        return .3*(F.relu(1.5-(truth*2-1)*logits)/1.5*interior).mean()

    def rgb_temporal(self,nodes,flow,valid):
        rgb=nodes['rgb_head']/128
        error=(rgb[1:]-warp(rgb[:1],flow)).abs()[:,:,40:-40,40:-40]
        valid=valid[:,:,40:-40,40:-40]
        return .7*(error*valid).sum()/(valid.sum()*3).clamp_min(1)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=1200);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100530);contract_guard();OUT.mkdir(exist_ok=False)
    data,selected,cache=c.load_data()
    write_json(OUT/'binding.json',dict(parent=str(PARENT),parent_sha256=file_hash(PARENT),parent_cumulative_steps=10500,
        new_qat_steps=a.steps,actual_B1=True,external_inference=False,
        code={str(p):file_hash(p) for p in [Path(__file__),Path(base.__file__),Path(recovery.__file__)]},
        changes='Ordinal threshold margin in teacher interiors; temporal probability weight5; actual RGB flow consistency .7'))
    for p in (__file__,base.__file__,recovery.__file__):shutil.copy2(p,OUT/Path(p).name)
    tr=Trainer(cache,selected,a.steps);native=tr.update(True);crop=tr.update();tr.save(OUT/'resume_probe.pt');ref=tr.update()
    expected={k:v.detach().clone() for k,v in tr.net.state_dict().items()}
    fresh=Trainer(cache,selected,a.steps);fresh.restore(OUT/'resume_probe.pt');got=fresh.update()
    error=max(float((v-expected[k]).abs().max()) for k,v in fresh.net.state_dict().items())
    assert error<1e-5 and ref['crop']==got['crop'] and ref['samples']==got['samples']
    write_json(OUT/'preflight.json',dict(status='PASS',native=native,pair_crop=crop,resume_error=error))
    print('TEMPORAL_V2_PREFLIGHT_PASS',error,flush=True)
    del tr,fresh;torch.cuda.empty_cache();tr=Trainer(cache,selected,a.steps)
    with (OUT/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        while tr.step<a.steps:
            row=tr.update();log.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('TEMPORAL_V2',tr.step,round(row['loss'],4),round(row['rgb_temporal'],4),flush=True)
        tr.save(OUT/f'checkpoint_{tr.step:06d}.pt');base.review(tr,data,selected,OUT/'package')
    print('TEMPORAL_V2_COMPLETE',flush=True)


if __name__=='__main__':main()
