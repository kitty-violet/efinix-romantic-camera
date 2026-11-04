"""Align existing14Y predicates after actual360p spatial reconstruction.

The labels identify which thresholds belong to one visible boundary. Only
training uses them. Inference remains the identical B1 graph and20KiB image.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_native_luma_readout import NativeLumaReadoutTraining,driver,evaluate,preview
from .b1_active_luma_interface import active_interface_loss,regression
from .probe_b1_decoder_coverage import fragment


class SharedRenderInterfaceTraining(NativeLumaReadoutTraining):
    alignment_weight=16.

    def additional_loss(self,p,t,native):
        loss,values=super().additional_loss(p,t,native)
        with torch.no_grad():
            z=F.pad(self.net.round(self.supervision_distance_truth),(0,0,0,0,0,2))
            known=fragment(self.net,z,'up1_low_pw')['up1_high_dw'][:,:14]
        field=self.nodes['up1_high_dw'][:,:14]
        if not native:field=field[...,20:-20,20:-20];known=known[...,20:-20,20:-20]
        alignment,fraction=active_interface_loss(field,known)
        values.update(shared_actual360_interface_alignment=alignment,shared_actual360_active_fraction=fraction)
        return loss+self.alignment_weight*alignment,values

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(shared_luma_alignment360=True,shared_luma_alignment_weight=self.alignment_weight,
            shared_luma_alignment_node='up1_high_dw',shared_luma_alignment_target='same active known native luminance interface after frozen interpolation',
            inference_target_labels=False,extra_inference_operations=0,visual_acceptance=False)
        torch.save(s,path)

    def preflight_extras(self):
        result=super().preflight_extras()
        assert 'shared_actual360_interface_alignment' in result['values']
        return result


if __name__=='__main__':
    torch.set_num_threads(2);regression()
    driver.main(training_factory=SharedRenderInterfaceTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
