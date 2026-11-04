"""Fine-tune rank-preserving spatial decisions inside existing B1 layers."""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_exact_complement_hinges import ExactComplementTraining,evaluate,driver,preview
from .train_b1_symmetric_binary_coverage import BinaryCoverageB1
from .train_b1_winner_surrogate import ExactValueProxyGradient


class OrdinalVoteB1(BinaryCoverageB1):
    def training_node_proxy(self,name,value,nodes):
        if name!='up1_low_dw':return value
        # The signed res3 PW clips at-128, then the existing up1 DW ReLU clips
        # at0. This proxy bypasses both saturations for training only.
        score=nodes['res3_conv2.dw'][:,:30]
        proxy=127*torch.sigmoid((score-63.5)/6)
        return ExactValueProxyGradient.apply(value,F.pad(proxy,(0,0,0,0,0,2)))


class OrdinalVoteTraining(ExactComplementTraining):
    def __init__(self,args,data):
        super().__init__(args,data);assert self.setup.get('ordinal_spatial_vote')
        self.net.__class__=OrdinalVoteB1

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(ordinal_spatial_vote=True,vote_gradient_proxy_only=True,minimum_area_guarantee=False,visual_acceptance=False)
        torch.save(state,path)


def regression(checkpoint):
    from .probe_b1_encoder_gauge import load
    state,net=load(checkpoint);x=torch.rand(1,3,64,80)*1.9-1
    with torch.no_grad():before=net(x,all_nodes=True)
    raw=net.export();net.__class__=OrdinalVoteB1;net.coverage_gain=state['binary_coverage720']['gain']
    after=net(x,all_nodes=True);assert all(torch.equal(v,after[k]) for k,v in before.items()) and net.export()==raw
    after['rgb_head'].square().mean().backward()
    grad=net.weight['res3_conv1__pw'].grad;assert grad is not None and grad.norm()>0 and torch.isfinite(grad).all()
    print('ORDINAL_VOTE_TRAIN_REGRESSION_PASS forward/bytes/finite_upstream_gradient',flush=True)


if __name__=='__main__':
    import sys
    torch.set_num_threads(2)
    if '--check' in sys.argv:regression(sys.argv[sys.argv.index('--check')+1])
    else:driver.main(training_factory=OrdinalVoteTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
