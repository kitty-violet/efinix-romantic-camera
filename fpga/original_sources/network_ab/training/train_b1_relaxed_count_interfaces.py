"""Joint luminance choices, allowing2 native pixels of boundary-label error.

Replace the independent14-field KL emphasis; retain actual renderedY,
stylized chroma, white-shoulder and source stability objectives. No inference
probability, neighbourhood lookup or filter is added.
"""
from pathlib import Path
import torch
from .train_b1_native_luma_readout import NativeLumaReadoutTraining,driver,evaluate,preview
from .b1_relaxed_luma_counts import allowed_levels,relaxed_count_loss,regression


class RelaxedCountInterfaceTraining(NativeLumaReadoutTraining):
    def additional_loss(self,p,t,native):
        loss,v=super().additional_loss(p,t,native)
        field=self.nodes['up1_high_dw'][:,:14];allowed=allowed_levels(self.class_labels,2)
        if native:
            phase=getattr(self,'step',0)%4;field=field[...,phase//2::2,phase%2::2];allowed=allowed[...,phase//2::2,phase%2::2]
        else:field=field[...,20:-20,20:-20];allowed=allowed[...,20:-20,20:-20]
        full,band=relaxed_count_loss(field,allowed)
        previous=self.args.classification*(2*v['ordered_distance_full']+v['ordered_distance_band']+8*v['luminance_decision_kl']+4*v['luminance_decision_band_kl'])
        replacement=.5*v['ordered_distance_full']+.25*v['ordered_distance_band']+8*full+4*band
        v.update(joint_luma_count_full=full,joint_luma_count_boundary=band,relaxed_label_fraction=(allowed.sum(1)>1).float().mean())
        return loss-previous+replacement,v

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(joint_luma_count_likelihood=True,independent_luma_KL_replaced=True,
            boundary_label_relaxation720=2,relaxed_labels_not_averaged=True,
            luminance_count_weights={'full':8,'boundary':4,'SDF_full':.5,'SDF_band':.25},
            probabilities_training_only=True,extra_inference_operations=0,visual_acceptance=False)
        torch.save(s,path)

    def preflight_extras(self):
        result=super().preflight_extras();assert 'joint_luma_count_full' in result['values'];return result


if __name__=='__main__':
    torch.set_num_threads(2);regression()
    driver.main(training_factory=RelaxedCountInterfaceTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
