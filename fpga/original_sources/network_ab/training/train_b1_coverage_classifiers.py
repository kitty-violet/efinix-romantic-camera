"""Preserve subcell region coverage in supervision of the sampled B1.

Classifying an averaged RGB pixel invents a hard class at boundaries. Classify
native pixels first and average their masks to preserve motion phase. This is
training-only supervision; no class search or extra operation is deployed.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_region_classifiers import RegionClassifierTraining
from . import train_b1_late_region_supervision as driver
from . import train_b1_region_formation as formation


@torch.no_grad()
def native_coverage(target,palette,indices):
    flat=target.permute(0,2,3,1).reshape(-1,3);selected=[]
    squared=palette.square().sum(1)[None]
    for chunk in flat.split(8192):
        distance=chunk.square().sum(1,keepdim=True)+squared-2*chunk@palette.T
        selected.append(indices[distance.argmin(1)])
    idx=torch.cat(selected).reshape(target.shape[0],*target.shape[2:],3).permute(0,3,1,2)
    bits=torch.cat([(idx[:,c:c+1]>=torch.arange(1,n+1,device=target.device)[None,:,None,None]).float()
                    for c,n in enumerate((8,4,4))],1)
    return F.avg_pool2d(bits,4)


def coverage_regression():
    palette=torch.tensor([[-1.,-1.,-1.],[0.,0.,0.],[1.,1.,1.]])
    indices=torch.tensor([[0,0,0],[4,2,2],[8,4,4]])
    target=torch.ones(1,3,4,4);target[:,:,:,:2]=-1
    expected=torch.full((1,16,1,1),.5)
    got=native_coverage(target,palette,indices)
    assert torch.equal(got,expected)
    # The previous mean-RGB assignment selects the middle class, containing
    # eight fully on and eight fully off bits, instead of half-covered masks.
    old=torch.cat([(indices[1,c]>=torch.arange(1,n+1)).float() for c,n in enumerate((8,4,4))])
    assert float((old-.5).abs().mean())==.5
    phases=[]
    for split in range(5):
        x=torch.ones(1,3,4,4);x[:,:,:,:split]=-1
        coverage=native_coverage(x,palette,indices)
        assert torch.equal(coverage,torch.full_like(coverage,(4-split)/4))
        phases.append(float(coverage.mean()))
    return dict(status='PASS',old_mask_mae=.5,new_mask_mae=0.,one_pixel_phases=phases,
                scope='Supervision regression only; actual learned phase/shape still needs validation')


class CoverageClassifierTraining(RegionClassifierTraining):
    def coverage_truth(self,target):return native_coverage(target,self.palette,self.palette_indices)


if __name__=='__main__':
    print('COVERAGE_TARGET_REGRESSION',coverage_regression(),flush=True)
    here=Path(__file__).parent
    driver.main(training_factory=CoverageClassifierTraining,
                extra_sources=[Path(__file__),here/'train_b1_region_classifiers.py',Path(formation.__file__),
                               here/'train_b1_quantum_regions.py',here/'train_b1_region_capacity.py',here/'train_b1_explicit_regions.py'],
                evaluation=formation.evaluate)
