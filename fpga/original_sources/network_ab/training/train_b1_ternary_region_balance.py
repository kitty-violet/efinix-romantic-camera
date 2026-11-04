"""Keep ternary colour capacity, prioritize constant connected-region interiors.

The first ternary stage learns count labels but validation flatness worsens
after500 steps while colour/count fit continues. This controlled second stage
keeps the graph, targets, decoder and count objective, and raises the measured
region-flatness and adjacent-region contrast weights. No new inference code.
"""
from pathlib import Path
import torch
from .train_b1_ternary_regions import TernaryRegionTraining
from .train_b1_joint_region_colour import evaluate,preview
from . import train_b1_late_region_supervision as driver


class TernaryRegionBalanceTraining(TernaryRegionTraining):
    def preflight_extras(self):
        rows=[]
        while self.step%4:rows.append(self.update(True))
        row=self.update(False)
        assert row['temporal_pair'] and 'tiny_input_stability' in row['values']
        rows.append(row)
        return rows

    def additional_loss(self,p,target,native):
        loss,values=super().additional_loss(p,target,native)
        extra=4*values['region_flatness']+values['adjacent_region_contrast']
        values['region_balance_extra']=extra
        return loss+extra,values

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(region_balance_extra=dict(flatness=4,adjacent_contrast=1),visual_acceptance=False)
        torch.save(s,path)


if __name__=='__main__':
    driver.main(training_factory=TernaryRegionBalanceTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
