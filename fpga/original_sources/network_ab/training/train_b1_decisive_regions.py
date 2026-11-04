"""Make the existing B1 learn modal-colour area cleanup, not colour averaging.

Connected regions/morphology exist only in supervision. Every prediction and
export remains the same B1 INT8/Q17 graph. No minimum-area inference guarantee.
"""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .train_b1_native_distance_boundaries import NativeDistanceTraining
from .train_b1_ordered_regions import evaluate,preview
from .minimum_area_blocks import components,area_merge,boundary_mode
from . import train_b1_late_region_supervision as driver


def modal_region_cleanup(ids,palette,minimum=64,rounds=2):
    rgb=palette[ids];regions,_=area_merge(components(ids),rgb,minimum)
    regions,_=area_merge(components(boundary_mode(regions,rounds)),rgb,minimum)
    table=np.bincount((regions.astype(np.int64)*len(palette)+ids).ravel(),
        minlength=(int(regions.max())+1)*len(palette)).reshape(-1,len(palette))
    colour=table.argmax(1);clean=colour[regions]
    assert set(np.unique(clean))<=set(np.unique(ids))
    return clean,regions


class DecisiveRegionTraining(NativeDistanceTraining):
    separate_region_projection=True

    def additional_loss(self,p,target,native):
        loss,values=super().additional_loss(p,target,native)
        if native:return loss,values
        # Count area using the full predicted crop, including surrounding
        # context. Cutting the loss ROI before segmentation falsely makes
        # regions crossing that ROI boundary look small.
        with torch.no_grad():
            actual=self.nodes['up2_low_dw'][0,:15]
            ids=self.predicted_codes(actual-63.5).cpu().numpy()
            palette=(self.palette.cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)
            cleaned,_=modal_region_cleanup(ids,palette,self.args.minimum_area//4,2)
            codes=torch.from_numpy(cleaned[20:-20,20:-20].astype(np.int64)).to(p.device)[None]
            before=torch.from_numpy(ids[20:-20,20:-20]).to(p.device)[None]
            changed=(codes!=before).float()[:,None]
            wanted=self.bit_table[codes].permute(0,3,1,2)
        logits=(self.nodes['up1_high_dw'][:1,:15,20:-20,20:-20]-63.5)/4
        ce=F.binary_cross_entropy_with_logits(logits,wanted,reduction='none')
        correction=(ce*changed).sum()/(15*changed.sum()).clamp_min(1)
        values.update(modal_area_correction=correction,modal_area_changed_fraction=changed.mean(),
            area_correction_weight=loss.new_tensor(3.))
        return loss+3*correction,values

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(modal_colour_area_supervision=True,area_correction_weight=3.,
            area_correction_frequency='every non-native crop',projection_uses_full_crop_context=True,
            minimum_area_inference_guarantee=False)
        torch.save(s,path)


def regression():
    palette=np.array([[20,40,130],[80,170,50],[220,140,90]],np.uint8)
    ids=np.zeros((48,64),np.int64);ids[:,32:]=1;ids[20:23,12:15]=2
    ids[10:20:2,32]=0
    clean,regions=modal_region_cleanup(ids,palette,64,2)
    assert np.all(clean[20:23,12:15]==0) and set(np.unique(clean))=={0,1}
    assert np.bincount(regions.ravel()).min()>=64
    # The full colour-region areas remain unchanged by selecting a smaller
    # training ROI afterwards; target cleanup never invents a mixed colour.
    counts=np.bincount(components(clean).ravel());assert counts.min()>=64
    print('MODAL_AREA_REGRESSION_PASS',dict(minimum_area=int(counts.min()),no_new_colours=True),flush=True)


if __name__=='__main__':
    regression();here=Path(__file__).parent
    driver.main(training_factory=DecisiveRegionTraining,extra_sources=[Path(__file__),
        here/'train_b1_native_distance_boundaries.py',here/'probe_b1_independent_decoder.py',
        here/'train_b1_independent_region_boundaries.py',here/'train_b1_balanced_region_boundaries.py',
        here/'train_b1_binary_regions.py',here/'train_b1_ordered_regions.py',here/'build_b1_ordered_palette.py',
        here/'minimum_area_blocks.py',here/'train_b1_quantum_regions.py',here/'train_b1_region_formation.py'],
        evaluation=evaluate,preview=preview)
