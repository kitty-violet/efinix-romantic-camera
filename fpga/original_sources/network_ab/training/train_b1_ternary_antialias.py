"""Refit the ternary predictor around fixed, existing stride kernels.

Controlled reuse of antialiasing, now in the ternary colour decoder branch.
Freeze only the two replaced stride DW kernels, not their neighbouring PW
mixing layers. No added inference layer, buffer, pass or external filtering.
"""
from pathlib import Path
import torch
from .train_b1_joint_rgb_edges import JointRGBEdgeTraining
from .train_b1_joint_region_colour import evaluate,preview
from . import train_b1_late_region_supervision as driver


class TernaryAntialiasTraining(JointRGBEdgeTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        warm=torch.load(args.warm,map_location='cpu',weights_only=True)
        self.aa_keys=warm['encoder_antialias_keys']
        assert self.aa_keys==['down1__dw','down2__dw']
        for key in self.aa_keys:
            w,b=self.net.weight[key],self.net.bias[key]
            w.requires_grad_(False);b.requires_grad_(False)
            self.fixed[key]=(w.detach().clone(),b.detach().clone())

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(encoder_antialias_keys=self.aa_keys,encoder_antialias_frozen=True,visual_acceptance=False)
        torch.save(s,path)


if __name__=='__main__':
    driver.main(training_factory=TernaryAntialiasTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
