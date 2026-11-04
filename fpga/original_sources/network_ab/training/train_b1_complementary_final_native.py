"""Train existing B1 layers after complementary final-basis initialization.

Initialization is inspired by positive/negative phase pairs (CReLU,
https://proceedings.mlr.press/v48/shang16.html), not the paper's architecture
or accuracy claim. Paired coefficients are untied during learning, and no
concatenation or other inference operator is added to the fixed B1 graph.
"""
from pathlib import Path
import copy,torch
from .train_b1_balanced_phase_shape import BalancedPhaseShapeTraining
from .train_b1_fine_rgb_native import evaluate
from .train_b1_late_region_supervision import main as driver
from . import train_b1_minimum_area as m


class ComplementaryFinalTraining(BalancedPhaseShapeTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        warm=torch.load(args.warm,map_location='cpu',weights_only=True)
        self.basis_initialization=copy.deepcopy(warm['complementary_final_basis'])

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(complementary_final_basis=self.basis_initialization,
            late_pairs_untied_during_training=True,warm_only=False,extra_inference_operations=0,visual_acceptance=False)
        torch.save(state,path)


if __name__=='__main__':
    torch.set_num_threads(2)
    driver(training_factory=ComplementaryFinalTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=m.t.preview)
