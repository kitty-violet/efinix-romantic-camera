"""Train full-precision categorical decisions with an exact B1 tournament."""
from pathlib import Path
import torch
from . import train_b1_sixteen_regions as parent
from . import train_b1_late_region_supervision as driver
from . import train_b1_minimum_area as m
from . import train_b1_region_formation as formation
from .train_b1_winner_surrogate import ExactValueProxyGradient
from .train_b1_binary_regions import preview_palette
from .probe_b1_sixteen_decoder import palette_actual


class TournamentGradientB1(m.t.r.c.ColourB1):
    def training_node_proxy(self,name,value,nodes):
        if name!='up1_low_pw':return value
        logits=nodes['res1_conv2.dw'][:,:16]/8
        return ExactValueProxyGradient.apply(value,127*logits.softmax(1))


class TournamentTraining(parent.SixteenTraining):
    network_class=TournamentGradientB1
    sampler_bound=127

    def score_logits(self,scores):return scores/8

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True);s.update(exact_tournament=True,full_precision_scores=True)
        torch.save(s,path)


def preview(net,rows,dest):return preview_palette(net,rows,dest,palette_actual)


if __name__=='__main__':
    here=Path(__file__).parent
    driver.main(training_factory=TournamentTraining,extra_sources=[Path(__file__),Path(parent.__file__),
        here/'compile_b1_tournament.py',here/'repair_b1_region_partition.py',here/'fit_b1_sixteen_regions.py',
        here/'probe_b1_sixteen_decoder.py',here/'train_b1_binary_regions.py',here/'train_b1_winner_surrogate.py',
        here/'train_b1_quantum_regions.py',here/'train_b1_clean_regions.py',here/'minimum_area_blocks.py',Path(formation.__file__)],
        evaluation=parent.evaluate,preview=preview)
