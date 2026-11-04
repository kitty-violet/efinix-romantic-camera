"""Train the actual15 pre-ramp decisions alongside weaker30-field geometry.

No additional inference operation or decoder change. This checks whether
optimizing physical final decisions improves on pure intermediate distance.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_raw_source_distance import RawSourceDistanceTraining,evaluate,driver,preview
from .probe_b1_ordered_distance_geometry import distance_fields
from .probe_b1_decoder_coverage import fragment
from .train_b1_ordered_distance_rims import distance_objective
from .b1_render_boundary_objective import rendering_decision_loss,smooth_rank_loss,regression
from .b1_distance_target_cache import DistanceTargetCache


def decision_objective(net,nodes,truth,native):
    with torch.no_grad():
        wanted=fragment(net,F.pad(net.round(truth),(0,0,0,0,0,2)),'up1_low_dw')['up1_high_dw']
    field=nodes['up1_low_dw'];pred=nodes['up1_high_dw']
    if not native:
        field=field[...,10:-10,10:-10];truth=truth[...,10:-10,10:-10]
        pred=pred[...,20:-20,20:-20];wanted=wanted[...,20:-20,20:-20]
    full,band=distance_objective(field,truth)
    decision,near=rendering_decision_loss(pred,wanted)
    return 2*full+band+8*decision+4*near,dict(ordered_distance_full=full,ordered_distance_band=band,
        rendered_decision_kl=decision,rendered_decision_band_kl=near)


class RenderBoundaryTraining(RawSourceDistanceTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        self.distance_cache=DistanceTargetCache('E:/fpga/model-training-workspace/cache/b1-native-sdf8-fp32')

    def additional_loss(self,p,t,native):
        truth=self.distance_cache(self.class_labels,8);self.supervision_distance_truth=truth
        loss,v=decision_objective(self.net,self.nodes,truth,native)
        field=self.nodes['up1_low_dw'] if native else self.nodes['up1_low_dw'][...,10:-10,10:-10]
        rank=smooth_rank_loss(field);v.update(squared_ordinal_rank_violation=rank,motion_aligned_target_fraction=self.motion_fraction)
        return self.args.classification*loss+10*rank,v

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(rendering_decision_supervision=True,rendering_decision_node='up1_high_dw',
            rendering_decision_temperature=3.,decision_target='same fixed decoder of known SDF8',
            ordinal_rank_penalty='squared_max_violation',ordinal_rank_loss_weight=10,
            native_distance_cache='lossless FP32 content-addressed; device/source/code hash',
            distance_primary_curriculum=False,extra_inference_operations=0,visual_acceptance=False)
        torch.save(s,path)


if __name__=='__main__':
    torch.set_num_threads(2);regression()
    driver.main(training_factory=RenderBoundaryTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
