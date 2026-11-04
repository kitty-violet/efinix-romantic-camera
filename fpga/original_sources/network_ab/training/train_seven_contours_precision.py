"""Precision-weighted contour classification with explicit input-scale control.

The fixed 3x3 input average is part of this experimental contour graph, not a
filter applied to output to hide false positives. RGB filling uses other input.
"""
import json
from pathlib import Path
import shutil
import sys
import torch
from torch.nn import functional as F
from . import train_seven_contours as base
from .runtime import file_hash,write_json


class SevenScaleContourNet(base.SevenContourNet):
    def forward(self,x):
        return super().forward(F.avg_pool2d(F.pad(x,(1,1,1,1),mode='replicate'),3,1))


def criterion(z,t,tracing=.35,margin=16):
    z=z[...,margin:-margin,margin:-margin];t=t[...,margin:-margin,margin:-margin]
    ce=F.binary_cross_entropy_with_logits(z,t,reduction='none');p=z.sigmoid()
    dims=(-1,-2)
    pos=(ce*t).sum(dims)/t.sum(dims).clamp_min(1)
    neg=(ce*(1-t)).sum(dims)/(1-t).sum(dims).clamp_min(1)
    # Raw-logit CE keeps a nonzero restoring gradient even for missing edges.
    # Increased background precision replaces tracing that rewarded clutter.
    loss=pos.mean()+4*neg.mean()
    return loss,dict(positive=float(pos.mean().detach()),negative=float(neg.mean().detach()),
                     response_mean=float(p.mean().detach()))


def main():
    base.SevenContourNet=SevenScaleContourNet;base.criterion=criterion
    if '--name' not in sys.argv:sys.argv.extend(['--name','seven-contour-precision-v2'])
    base.main()
    if sys.argv[1]=='prepare':
        out=base.runner.base.ROOT/sys.argv[sys.argv.index('--name')+1]
        cfg=json.loads((out/'config.json').read_text(encoding='utf-8'))
        cfg.update(lr=.0003,objective='seven binary contour maps; precision-weighted logit CE 1:4; no tracing; fixed input average3',
                   graph='context90_seven_contour_input_average3',input_average3=True)
        write_json(out/'config.json',cfg);b=json.loads((out/'binding.json').read_text(encoding='utf-8'))
        p=Path(__file__);b['code'][str(p)]=file_hash(p);shutil.copy2(p,out/'source'/p.name)
        b['config']=file_hash(out/'config.json');write_json(out/'binding.json',b)


if __name__=='__main__':main()
