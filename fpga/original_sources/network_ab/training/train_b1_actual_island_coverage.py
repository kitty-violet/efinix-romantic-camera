"""Actual720 wrong-island supervision across every TRAIN source and flow pair."""
from pathlib import Path
import cv2
import numpy as np
import torch
from .train_b1_native_coverage_counts import NativeCoverageCountTraining, schedule_regression
from .train_b1_actual_island_regions import actual_region_losses, regression
from .train_b1_count_regions import count_targets, count_loss
from .minimum_area_blocks import components, boundary_mask
from .train_b1_ordered_regions import evaluate, preview
from . import train_b1_late_region_supervision as driver


class ActualIslandCoverageTraining(NativeCoverageCountTraining):
    preflight_native_resume=True

    def count_supervision(self,logits,truth):
        return count_loss(logits,truth),{}

    def count_logits(self):
        return (self.nodes['up1_high_dw'][:,:15]-63.5)/4

    def preflight_extras(self):
        # The shared native + three-crop preflight lands on step4; its native
        # resume check would otherwise skip the actual adjacent-frame branch.
        rows=[]
        while self.step%4:rows.append(self.update(True))
        row=self.update(False)
        assert row['temporal_pair'] and len(row['samples'])==2
        assert 'temporal' in row['values'] and 'tiny_input_stability' in row['values']
        rows.append(row)
        return rows

    def additional_loss(self,p,target,native):
        logits=self.count_logits()
        truth=count_targets(self.class_labels);coords=self.nodes['up2_high_dw'][:,:3]
        codes=self.class_labels
        if not native:
            logits=logits[...,20:-20,20:-20];truth=[v[...,20:-20,20:-20] for v in truth]
            coords=coords[...,40:-40,40:-40];codes=codes[...,40:-40,40:-40]
        wanted=torch.stack((codes//25,(codes//5)%5,codes%5),1).float()
        labs=[];valid=[]
        for code in codes.detach().cpu().numpy():
            labs.append(torch.tensor(components(code).astype(np.int64),device=p.device))
            valid.append(torch.tensor(cv2.dilate(boundary_mask(code),np.ones((7,7),np.uint8))==0,device=p.device))
        ce,extra_values=self.count_supervision(logits,truth)
        extra,values=actual_region_losses(coords,wanted,torch.stack(labs),torch.stack(valid))
        values['actual_palette_count_ce']=ce
        values.update(extra_values)
        return ce+extra,values

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(actual_integer_native_region_objective=True,area_minimum720=256,
            objective='count CE + actual720 coordinate Huber + .5 anchored region variance + wrong-component correction + existing input/flow stability',
            full_training_coverage=True,training_sources=len(self.groups),
            actual_training_frame_count=sum(map(len,self.groups.values())),
            small_island_target='immutable TRAIN truth; correct coordinates protected; no self-colour labels',
            minimum_area_inference_guarantee=False,visual_acceptance=False)
        # A warm start from a limited fitting test does not make this a
        # seven-scene-only run, and does not inherit its optimizer state.
        s.pop('seven_scene_fit_diagnostic',None);s.pop('fit_ids',None)
        torch.save(s,path)


if __name__=='__main__':
    regression();schedule_regression();here=Path(__file__).parent
    names=['train_b1_actual_island_regions.py','probe_b1_native_region_objective.py','train_b1_native_coverage_counts.py',
        'train_b1_truth_area_counts.py','train_b1_temporal_count_regions.py','train_b1_count_regions.py',
        'train_b1_ordered_regions.py','minimum_area_blocks.py','train_b1_clean_regions.py']
    driver.main(training_factory=ActualIslandCoverageTraining,extra_sources=[Path(__file__)]+[here/n for n in names],
        evaluation=evaluate,preview=preview)
