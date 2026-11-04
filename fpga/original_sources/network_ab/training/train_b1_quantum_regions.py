"""Respect each row's INT8 weight quantum during region learning.

An effective-weight LR shared across rows with scales1/4096 and1/4 can
overdrive one row while never changing an integer coefficient in another.
Only the optimizer changes here; exported graph and arithmetic are unchanged.
"""
from pathlib import Path
import torch
from .train_b1_region_capacity import RegionCapacityTraining
from . import train_b1_late_region_supervision as driver
from . import train_b1_region_formation as formation


class RowScaledAdam(torch.optim.Adam):
    @torch.no_grad()
    def step(self,closure=None):
        saved=[(p,p.clone(),group['row_gain']) for group in self.param_groups if 'row_gain' in group
               for p in group['params'] if p.grad is not None]
        result=super().step(closure)
        for p,old,gain in saved:p.copy_(old+(p-old)*gain.to(p.device))
        return result


class QuantumRegionTraining(RegionCapacityTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        # Biases use wide integer accumulators and do not have the same127-code
        # range restriction. Keep their effective-value LR independently.
        groups=[]
        for key,p in self.net.weight.items():
            if not p.requires_grad:continue
            scale=(getattr(self.net,key+'_m')/131072).float().flatten()
            groups.append(dict(params=[p],lr=args.lr,row_gain=(scale/(1/256))[:,None,None,None]))
        groups.append(dict(params=[p for p in self.net.bias.values() if p.requires_grad],lr=.002))
        self.opt=RowScaledAdam(groups);self.base_lrs=[g['lr'] for g in groups]
        self.start_integer={key:self.codes(key).clone() for key,p in self.net.weight.items() if p.requires_grad}

    @torch.no_grad()
    def codes(self,key):
        s=(getattr(self.net,key+'_m')/131072).float().flatten()[:,None,None,None]
        return self.net.round(self.net.weight[key]/s).clamp(-127,127)

    def update(self,native=None):
        row=super().update(native)
        if self.step%100==0:
            changes={k:int((self.codes(k)!=v).sum()) for k,v in self.start_integer.items()}
            row['integer_weight_changes']=changes;print('INTEGER_WEIGHT_CHANGES',self.step,changes,flush=True)
        return row


if __name__=='__main__':
    here=Path(__file__).parent
    driver.main(training_factory=QuantumRegionTraining,
                extra_sources=[Path(__file__),Path(formation.__file__),here/'train_b1_region_capacity.py',here/'train_b1_explicit_regions.py'],
                evaluation=formation.evaluate)
