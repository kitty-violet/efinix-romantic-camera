"""Expand context for boundary selection while retaining native-pixel output.

The extra stage uses only stride2 DW3/PW, nearest2 and residual additions.
Nominal arithmetic/parameter checks are not a physical allocation/FPGA proof.
"""
import json
import shutil
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .native_contour_net import NativeContourNet,Block
from . import train_native_contours as base
from .train_selective_contours import SelectiveSamples,criterion as selective_loss,base_loss
from .runtime import file_hash,write_json


class ContextContourNet(NativeContourNet):
    def __init__(self,difference=True):
        super().__init__(difference)
        self.down3=Block(32,32,2,difference)
        self.context.append(Block(32,32,difference=difference))

    def forward(self,x):
        native=self.native(F.relu(self.stem(x)))
        half=self.half_block(self.down1(native));quarter=self.down2(half)
        c=self.down3(quarter)
        for block in self.context:c=c+block(c)
        c=F.interpolate(c,scale_factor=2,mode='nearest')+quarter
        h=self.decode1(F.interpolate(F.relu(self.reduce(c)),scale_factor=2,mode='nearest')+half)
        n=F.interpolate(h,scale_factor=2,mode='nearest')+native
        return self.edge(F.relu(self.finish(n)))


def criterion(logit,packed,tracing=.35,margin=16):
    full,terms=selective_loss(logit,packed,tracing,margin)
    plain,_=base_loss(logit,packed[:,:1],tracing,margin)
    return .375*full+.625*plain,terms  # .3 hard-negative weight, other terms unchanged.


def main():
    base.NativeContourNet=ContextContourNet;base.Samples=SelectiveSamples;base.criterion=criterion
    if '--name' not in sys.argv:sys.argv.extend(['--name','native-contour-context-v1'])
    if sys.argv[1]=='prepare':
        base.main();out=base.ROOT/sys.argv[sys.argv.index('--name')+1]
        cfg=json.loads((out/'config.json').read_text(encoding='utf-8'))
        source=cfg['initial'];state=torch.load(source,map_location='cpu',weights_only=True)
        torch.manual_seed(cfg['seed']);net=ContextContourNet();keys=net.load_state_dict(state['model'],strict=False)
        assert not keys.unexpected_keys and all(k.startswith(('down3.','context.3.')) for k in keys.missing_keys)
        initial=out/'initial_context.pt';torch.save(dict(model=net.state_dict(),source=source,source_sha256=file_hash(source)),initial)
        cfg.update(initial=str(initial),initial_sha256=file_hash(initial),graph='context90_native_contour',
            objective='native contours with wider context and teacher-removed source edges',hard_negative_weight=.3)
        write_json(out/'config.json',cfg);b=json.loads((out/'binding.json').read_text(encoding='utf-8'))
        for p in [Path(__file__),Path(__file__).with_name('train_selective_contours.py')]:
            b['code'][str(p)]=file_hash(p);shutil.copy2(p,out/'source'/p.name)
        b['config']=file_hash(out/'config.json');write_json(out/'binding.json',b)
        return
    base.main()


if __name__=='__main__':main()
