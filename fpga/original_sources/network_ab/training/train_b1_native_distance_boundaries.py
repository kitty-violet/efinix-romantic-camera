"""Preserve native720 boundary coverage before supervising180p fields.

Point-sampling binary regions first discards boundary phase/geometry. Compute
signed distances at native720, subtract the pixel-centre half-distance, then
average over4x4 footprint. This affects training targets only; B1 stays fixed.
"""
from pathlib import Path
import torch
from .probe_b1_independent_decoder import native_distances
from .train_b1_balanced_region_boundaries import BalancedBoundaryTraining
from .train_b1_independent_region_boundaries import distances
from .train_b1_ordered_regions import evaluate,preview
from . import train_b1_late_region_supervision as driver


class NativeDistanceTraining(BalancedBoundaryTraining):
    def distance_targets(self,bits_full):return native_distances(bits_full)

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(native720_distance_targets=True,distance_sampling='pixel-centre SDF at720; average4x4; /4; clip4')
        torch.save(state,path)


def regression():
    # A one-pixel translation inside a stride4 footprint must remain visible
    # in the supervised field instead of disappearing until the next phase.
    masks=torch.zeros(4,1,32,32)
    for i in range(4):masks[i,:,:,12+i:]=1
    native=native_distances(masks);old=distances(masks[:,:,::4,::4])
    assert native.shape==(4,1,8,8)
    assert torch.equal(old[1],old[2]) and torch.equal(old[2],old[3])
    assert all(not torch.equal(native[i],native[i+1]) for i in range(3))
    constant=native_distances(torch.cat((torch.zeros(1,1,32,32),torch.ones(1,1,32,32))))
    assert torch.all(constant[0]==-4) and torch.all(constant[1]==4)
    print('NATIVE_DISTANCE_PHASE_REGRESSION_PASS',flush=True)


if __name__=='__main__':
    regression();here=Path(__file__).parent
    driver.main(training_factory=NativeDistanceTraining,extra_sources=[Path(__file__),
        here/'probe_b1_independent_decoder.py',here/'train_b1_independent_region_boundaries.py',
        here/'train_b1_balanced_region_boundaries.py',here/'train_b1_binary_regions.py',
        here/'train_b1_ordered_regions.py',here/'build_b1_ordered_palette.py',
        here/'minimum_area_blocks.py',here/'train_b1_region_formation.py',here/'train_b1_quantum_regions.py'],
        evaluation=evaluate,preview=preview)
