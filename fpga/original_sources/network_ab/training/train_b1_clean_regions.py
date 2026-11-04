"""Train fixed majority-region B1 with explicit tiny-input-change stability."""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_coverage_classifiers import CoverageClassifierTraining,coverage_regression
from . import train_b1_late_region_supervision as driver
from . import train_b1_region_formation as formation


class CleanRegionTraining(CoverageClassifierTraining):
    def input_stability_loss(self,x,pred,native):
        if native or self.step%2:return pred.new_zeros(()),{}
        # Same geometry and colours to within sensor/encoding-sized changes.
        # No blurred target, frame blending, history or extra inference input.
        sign=1 if (self.step//2)%2 else -1
        noise=torch.randn_like(x[:1])*(.5/128)
        perturbed=(x[:1]+sign/128+noise).clamp(-1,127/128)
        other=self.net(perturbed)[0][...,40:-40,40:-40]
        anchor=pred[:1,...,40:-40,40:-40].detach()
        stable=F.smooth_l1_loss(other,anchor,beta=2/128)
        return 5*stable,dict(tiny_input_stability=stable,
                            tiny_input_output_mae_rgb=(other-anchor).abs().mean()*128)


if __name__=='__main__':
    print('COVERAGE_TARGET_REGRESSION',coverage_regression(),flush=True)
    here=Path(__file__).parent
    driver.main(training_factory=CleanRegionTraining,
                extra_sources=[Path(__file__),here/'train_b1_coverage_classifiers.py',here/'train_b1_region_classifiers.py',
                               Path(formation.__file__),here/'train_b1_quantum_regions.py',here/'train_b1_region_capacity.py',
                               here/'train_b1_explicit_regions.py'],evaluation=formation.evaluate)
