"""Restore explicit signed contour contrast at the B1 sampling scales.

Count/island learning alone does not directly match colour jumps at object
boundaries. Compare vector differences over4 and8 native720 pixels, avoiding
a demand for photograph texture or adding image sharpening at inference.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_pure_footprint_regions import PureFootprintTraining
from .train_b1_ordered_regions import evaluate, preview
from . import train_b1_late_region_supervision as driver


def boundary_contrast_loss(pred,target):
    terms=[]
    for distance in (4,8):
        for axis in (-1,-2):
            length=pred.shape[axis]-distance
            delta=lambda z:z.narrow(axis,distance,length)-z.narrow(axis,0,length)
            actual,wanted=delta(pred),delta(target)
            # Includes colour direction, not merely total edge magnitude.
            mask=(wanted.abs().amax(1,keepdim=True)>8/128).float()
            error=F.smooth_l1_loss(actual,wanted,beta=8/128,reduction='none')
            terms.append((error*mask).sum()/(3*mask.sum()).clamp_min(1))
    return torch.stack(terms).mean()


class NativeContrastTraining(PureFootprintTraining):
    def additional_loss(self,p,target,native):
        loss,values=super().additional_loss(p,target,native)
        contrast=boundary_contrast_loss(p,target)
        values['native_signed_boundary_contrast']=contrast
        return loss+2*contrast,values

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(native_boundary_contrast_supervision=True,boundary_offsets720=[4,8],boundary_loss_weight=2,
                     no_source_detail_reconstruction=True,visual_acceptance=False)
        torch.save(state,path)


def regression():
    target=torch.full((1,3,64,96),-.25);target[:,:,:,48:]=.5
    blurred=F.avg_pool2d(F.pad(target,(8,8,8,8),mode='replicate'),17,1)
    contrastless=torch.full_like(target,.125)
    reversed_image=.25-target
    shifted=target.clone();shifted[:,:,:,48:49]=-.25
    cases=dict(exact=target,blurred=blurred,flat=contrastless,reversed=reversed_image,one_pixel_shift=shifted)
    scores={k:float(boundary_contrast_loss(p,target)) for k,p in cases.items()}
    assert scores['exact']==0 and scores['reversed']>scores['flat']>scores['blurred']>scores['one_pixel_shift']
    # Clean up a weakened edge; preserve the signed colour order.
    weak=(target*.5).requires_grad_();loss=boundary_contrast_loss(weak,target);loss.backward()
    assert weak.grad[:,:,:,44:48].mean()>0 and weak.grad[:,:,:,48:52].mean()<0
    constant=torch.zeros_like(target,requires_grad=True)
    loss=boundary_contrast_loss(constant,torch.zeros_like(target));loss.backward()
    assert float(loss.detach())==0 and not constant.grad.any()
    print('NATIVE_CONTRAST_REGRESSION_PASS',scores,flush=True)


if __name__=='__main__':
    regression();here=Path(__file__).parent
    names=['train_b1_pure_footprint_regions.py','train_b1_actual_island_coverage.py','train_b1_actual_island_regions.py',
        'train_b1_native_coverage_counts.py','train_b1_count_regions.py','train_b1_truth_area_counts.py',
        'train_b1_temporal_count_regions.py','probe_b1_native_region_objective.py','minimum_area_blocks.py']
    driver.main(training_factory=NativeContrastTraining,extra_sources=[Path(__file__)]+[here/n for n in names],
                evaluation=evaluate,preview=preview)
