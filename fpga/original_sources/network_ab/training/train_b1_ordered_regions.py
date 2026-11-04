"""Learn tied ordered region levels, with sharp360p classification before RGB."""
from pathlib import Path
import torch
from torch.nn import functional as F
from . import train_b1_binary_regions as parent
from . import train_b1_late_region_supervision as driver
from . import train_b1_minimum_area as m
from . import train_b1_region_formation as formation
from .build_b1_ordered_palette import GROUPS,bits,actual_palette,project
from .train_b1_winner_surrogate import ExactValueProxyGradient


class OrderedGradientB1(m.t.r.c.ColourB1):
    def training_node_proxy(self,name,value,nodes):
        if name!='up2_low_dw':return value
        logits=(nodes['up1_high_dw'][:,:15]-63.5)/4
        proxy=F.pad(127*logits.sigmoid(),(0,0,0,0,0,1))
        return ExactValueProxyGradient.apply(value,proxy)


class OrderedTraining(parent.BinaryTraining):
    checkpoint_flag='ordered_regions'
    network_class=OrderedGradientB1
    palette_function=staticmethod(actual_palette)
    bit_function=staticmethod(bits)
    num_bits=15

    def __init__(self,args,data):
        super().__init__(args,data)
        # Share the gradient before Adam as well as the stored rows. Merely
        # averaging independently updated Adam rows gives the wrong shared-field
        # optimization when thresholds have different gradient magnitudes.
        def tied_gradient(grad):
            grad=grad.clone()
            for start,end in GROUPS:grad[start:end]=grad[start:end].mean(0,keepdim=True)
            return grad
        self.net.weight['up1_low_pw'].register_hook(tied_gradient)
        self.net.bias['up1_low_pw'].register_hook(tied_gradient)

    def predicted_codes(self,logits):
        count=torch.stack([(logits[start:end]>=0).long().sum(0) for start,end in GROUPS])
        return count[0]*25+count[1]*5+count[2]

    def update(self,native=None):
        row=super().update(native);project(self.net);return row

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True);s.pop('binary_regions',None)
        s.update(ordered_regions=True,coverage_layout='15ordered360_rgb200',monotonic_tied_thresholds=True)
        torch.save(s,path)


def evaluate(net,rows):return parent.evaluate_palette(net,rows,actual_palette)


def preview(net,rows,dest):return parent.preview_palette(net,rows,dest,actual_palette)


if __name__=='__main__':
    here=Path(__file__).parent
    driver.main(training_factory=OrderedTraining,extra_sources=[Path(__file__),Path(parent.__file__),here/'build_b1_ordered_palette.py',
        here/'build_b1_binary_palette.py',here/'fit_b1_sixteen_regions.py',here/'train_b1_sixteen_regions.py',
        here/'train_b1_winner_surrogate.py',here/'train_b1_quantum_regions.py',here/'train_b1_clean_regions.py',
        here/'minimum_area_blocks.py',Path(formation.__file__)],evaluation=evaluate,preview=preview)
