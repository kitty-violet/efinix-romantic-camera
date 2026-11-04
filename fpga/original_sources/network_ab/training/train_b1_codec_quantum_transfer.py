"""Controlled optimizer revision of painting-code transfer, same B1 graph.

Use an existing per-row integer-step optimizer so large-scale DW rows receive
comparable integer-code updates to PW rows. Not a new inference operator.
"""
from pathlib import Path
import torch
from .train_b1_painting_codec import PaintingCodeTransferTraining
from .train_b1_quantum_regions import RowScaledAdam
from .train_b1_joint_region_colour import evaluate,preview
from . import train_b1_late_region_supervision as driver


class CodecQuantumTransferTraining(PaintingCodeTransferTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        groups=[]
        for key,p in self.net.weight.items():
            if not p.requires_grad:continue
            scale=(getattr(self.net,key+'_m')/131072).float().flatten()
            groups.append(dict(params=[p],lr=args.lr,row_gain=(scale*256)[:,None,None,None]))
        groups.append(dict(params=[p for p in self.net.bias.values() if p.requires_grad],lr=.002))
        self.opt=RowScaledAdam(groups);self.base_lrs=[g['lr'] for g in groups]

    def save(self,path):
        super().save(path)
        state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(row_scaled_updates=True,visual_acceptance=False)
        torch.save(state,path)


if __name__=='__main__':
    driver.main(training_factory=CodecQuantumTransferTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
