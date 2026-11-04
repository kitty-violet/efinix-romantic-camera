"""Predict shared-interface targets in the unchanged B1, real source only.

Every colour predicate uses the same local interface, replacing independent
colour-union distances. The real1215 constant outputs determine training code
assignment, including a calibrated existing head. No labels at inference.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_raw_source_distance import RawSourceDistanceTraining,driver,preview
from .train_b1_coincident_distance import evaluate as old_evaluate
from .b1_joint_interface_targets import joint_interface_fields,regression
from .b1_distance_target_cache import DistanceTargetCache
from .build_b1_thirty_binary_regions import code_bits
from .probe_b1_decoder_coverage import fragment
from .train_b1_ordered_distance_rims import distance_objective
from .b1_render_boundary_objective import smooth_rank_loss
from .train_b1_ternary_regions import region_codes
from . import train_b1_minimum_area as m
import numpy as np


class JointInterfaceTraining(RawSourceDistanceTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        self.distance_cache=DistanceTargetCache('E:/fpga/model-training-workspace/cache/b1-joint-interface-fp32',compute=joint_interface_fields)
        with torch.no_grad():
            z=F.pad(code_bits(torch.arange(1215,device='cuda')[:,None,None])*127,(0,0,0,0,0,2))
            self.palette=fragment(self.net,z,'up1_low_dw')['rgb_head'][:,:,0,0]/128
            self.pk=tuple(map(tuple,(self.palette.cpu().numpy()*128+128).tolist()))

    def additional_loss(self,p,t,native):
        truth=self.distance_cache(self.class_labels,8);self.supervision_distance_truth=truth
        field=self.nodes['up1_low_dw']
        if not native:field=field[...,10:-10,10:-10];truth=truth[...,10:-10,10:-10]
        full,band=distance_objective(field,truth);rank=smooth_rank_loss(field)
        return self.args.classification*(8*full+4*band)+10*rank,dict(ordered_distance_full=full,ordered_distance_band=band,
            squared_ordinal_rank_violation=rank,motion_aligned_target_fraction=self.motion_fraction)

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        warm=torch.load(self.args.warm,map_location='cpu',weights_only=True)
        s.update(joint_colour_interface_targets=True,ordinal_target='shared nearest palette interface, bounded common native mixture then avg4',
            colour_assignment='1215 actual constant integer decoder outputs',ordinal_rank_penalty='squared_max_violation',
            independent_signed_distance_targets=False,native_distance_cache='lossless FP32 content-addressed',
            radix_colour_calibration=warm.get('radix_colour_calibration'),constant_palette_requires_runtime=True,
            extra_inference_operations=0,visual_acceptance=False)
        torch.save(s,path)


@torch.inference_mode()
def evaluate(net,rows):
    result=old_evaluate(net,rows);device=next(net.parameters()).device
    z=F.pad(code_bits(torch.arange(1215,device=device)[:,None,None])*127,(0,0,0,0,0,2))
    pk=tuple(map(tuple,(fragment(net,z,'up1_low_dw')['rgb_head'][:,:,0,0].cpu()+128).tolist()))
    for record,row in zip(result['records'],rows):
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];field=net(x,all_nodes=True)['up1_low_dw']
        code=torch.from_numpy(region_codes(row['target'],row['region'],pk).astype(np.int64))[None].to(device)
        truth=joint_interface_fields(code);full,band=distance_objective(field,truth)
        record.update(joint_interface_full=float(full),joint_interface_band=float(band))
    for key in ('joint_interface_full','joint_interface_band'):result['mean'][key]=sum(r[key] for r in result['records'])/len(rows)
    return result


if __name__=='__main__':
    torch.set_num_threads(2);regression()
    driver.main(training_factory=JointInterfaceTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
