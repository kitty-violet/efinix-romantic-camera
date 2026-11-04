"""Fit contour distance as primary supervision before joint RGB fine tuning.

Controlled continuation of the same source/augmentation/temporal setup. Remove
competing RGB and coincidence terms for this stage; retain direct target-field
and rank supervision. No improvement is assumed until actual images are read.
"""
from pathlib import Path
import torch
from .train_b1_ordered_distance_rims import OrderedDistanceRimTraining,distance_objective
from .probe_b1_ordered_distance_geometry import distance_fields
from .train_b1_ordered_boundary_fields import rank_loss
from .train_b1_coincident_distance import evaluate,driver,preview


class DistanceCurriculumTraining(OrderedDistanceRimTraining):
    def additional_loss(self,p,t,native):
        truth=distance_fields(self.class_labels,8);field=self.nodes['up1_low_dw']
        self.supervision_distance_truth=truth
        if not native:field=field[...,10:-10,10:-10];truth=truth[...,10:-10,10:-10]
        full,band=distance_objective(field,truth);rank=rank_loss(field)
        return self.args.classification*(8*full+4*band)+10*rank,dict(ordered_distance_full=full,ordered_distance_band=band,ordinal_rank_violation=rank,motion_aligned_target_fraction=self.motion_fraction)

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(distance_primary_curriculum=True,rgb_training_target='No RGB target loss in this curriculum; RGB validation unchanged',
            rgb_objective_weight=0,coincident_boundary_supervision=False,rgb_priority_gradient=False,
            signed_distance_training_only=True,extra_inference_operations=0,visual_acceptance=False)
        torch.save(s,path)


if __name__=='__main__':
    torch.set_num_threads(2)
    driver.main(training_factory=DistanceCurriculumTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
