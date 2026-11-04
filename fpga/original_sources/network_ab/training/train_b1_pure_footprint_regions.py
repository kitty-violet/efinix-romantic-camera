"""Do not fit incompatible mixtures with independent count likelihoods.

Pure2x2 footprints retain the original count CE. The actual720 colour/region
loss still supervises every pixel, including mixed boundary footprints.
No output processing is added, and this is not a hard area guarantee.
"""
from pathlib import Path
import torch
from .train_b1_actual_island_coverage import ActualIslandCoverageTraining
from .train_b1_count_regions import count_loss, count_targets
from .train_b1_ordered_regions import evaluate, preview
from . import train_b1_late_region_supervision as driver


class PureFootprintTraining(ActualIslandCoverageTraining):
    def count_supervision(self,logits,truth):
        pure=torch.stack([v.amax(1)>.999 for v in truth]).all(0)[:,None].float()
        return count_loss(logits,truth,mask=pure),dict(mixed_footprint_fraction=1-pure.mean())

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(pure_footprint_count_supervision=True,
            boundary_supervision='actual native720 labels, not independent-count mixture',
            manual_decoder_parent=self.args.warm,
            minimum_area_inference_guarantee=False)
        torch.save(state,path)


def regression():
    # This2x4 image has one pure2x2 footprint and one mixed footprint.
    codes=torch.tensor([[[0,0,0,199],[0,0,0,199]]])
    truth=count_targets(codes)
    z=torch.zeros(1,15,1,2,requires_grad=True)
    loss,values=PureFootprintTraining.count_supervision(None,z,truth)
    loss.backward()
    assert z.grad[:,:,:,0].abs().sum()>0 and z.grad[:,:,:,1].abs().sum()==0
    assert float(values['mixed_footprint_fraction'])==.5
    # Entirely mixed batches must be finite and contribute no count gradient.
    z=torch.zeros(1,15,1,1,requires_grad=True)
    loss,_=PureFootprintTraining.count_supervision(None,z,[v[:,:,:,1:] for v in truth])
    loss.backward();assert float(loss.detach())==0 and not z.grad.any()
    print('PURE_FOOTPRINT_REGRESSION_PASS protected_mixture / pure_gradient / empty',flush=True)


if __name__=='__main__':
    regression();here=Path(__file__).parent
    names=['train_b1_actual_island_coverage.py','train_b1_actual_island_regions.py',
        'train_b1_native_coverage_counts.py','train_b1_count_regions.py','train_b1_truth_area_counts.py',
        'train_b1_temporal_count_regions.py','probe_b1_native_region_objective.py','minimum_area_blocks.py']
    driver.main(training_factory=PureFootprintTraining,extra_sources=[Path(__file__)]+[here/n for n in names],
                evaluation=evaluate,preview=preview)
