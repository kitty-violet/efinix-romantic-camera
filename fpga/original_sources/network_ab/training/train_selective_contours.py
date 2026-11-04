"""Same native contour graph, explicit supervision for teacher-removed texture.

All source-derived edge masks exist only in training. They mark hard negative
examples, never a feature input or an inference postprocessor. One-pixel teacher
neighborhoods are excluded from these extra negatives to avoid imposing the
source geometry on slightly displaced teacher edges.
"""
import json
import shutil
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
from . import train_native_contours as base
from .runtime import file_hash,write_json

base_loss=base.criterion
BaseSamples=base.Samples


class SelectiveSamples(BaseSamples):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.source_edges={}

    def batch(self,native=False):
        x,t,spec=super().batch(native);neg=[]
        for i,nxt,px,py,size in spec:
            key=(i,nxt)
            if key not in self.source_edges:self.source_edges[key]=base.edges(self.image(i,nxt)[0])
            edge=self.source_edges[key]
            if size:edge=edge[py:py+size,px:px+size]
            neg.append(torch.from_numpy(edge.copy())[None])
        return x,torch.cat([t,torch.stack(neg).to(t.device)],1),spec


def criterion(logit,packed,tracing=.35,margin=16):
    target=packed[:,:1];loss,terms=base_loss(logit,target,tracing,margin)
    hard=logit.new_zeros(())
    if packed.shape[1]==2:
        mask=packed[:,1:2]*(1-F.max_pool2d(target,3,1,1))
        mask=mask[...,margin:-margin,margin:-margin]
        z=logit[...,margin:-margin,margin:-margin]
        hard=(F.softplus(z)*mask).sum()/mask.sum().clamp_min(1)
    terms['teacher_removed_source_edges']=float(hard.detach())
    return loss+.8*hard,terms


def main():
    base.Samples=SelectiveSamples;base.criterion=criterion
    if '--name' not in sys.argv:sys.argv.extend(['--name','native-contour-selective-v1'])
    if sys.argv[1]=='prepare':
        base.main()
        name=sys.argv[sys.argv.index('--name')+1];out=base.ROOT/name
        cfg=json.loads((out/'config.json').read_text(encoding='utf-8'))
        cfg.update(objective='native contours with teacher-removed source-edge hard negatives',hard_negative_weight=.8)
        write_json(out/'config.json',cfg)
        b=json.loads((out/'binding.json').read_text(encoding='utf-8'))
        b['code'][str(Path(__file__))]=file_hash(__file__);b['config']=file_hash(out/'config.json')
        write_json(out/'binding.json',b);shutil.copy2(__file__,out/'source'/Path(__file__).name)
        return
    base.main()


if __name__=='__main__':main()
