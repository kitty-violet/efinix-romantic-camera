"""Recover training gradients for wrongly saturated ordinal boundary fields.

Earlier proxies covered the final score ramp and the spatial vote, but the
preceding30-field PW could already clip a wrong field to0/127 with zero gradient.
Keep exact integer forward values; use a small leaky training-only derivative
outside that interval. Separately weight fractional target coverage cells.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_ordered_boundary_fields import OrderedBoundaryTraining,driver,evaluate,preview
from .train_b1_ordinal_vote_hinges import OrdinalVoteB1
from .train_b1_winner_surrogate import ExactValueProxyGradient
from .build_b1_thirty_binary_regions import code_bits

def recover_gradient(value,pre):
    clipped=pre.clamp(0,127)
    proxy=clipped+.05*(pre-clipped)
    return ExactValueProxyGradient.apply(value,proxy)

class BoundaryGradientB1(OrdinalVoteB1):
    def training_activation_proxy(self,name,value,pre,r):
        if name=='res3_conv1.pw':return recover_gradient(value,pre)
        return super().training_activation_proxy(name,value,pre,r)

def mixed_occupancy_loss(pred,code):
    truth=F.avg_pool2d(code_bits(code),4);mixed=(truth>0)&(truth<1)
    error=F.smooth_l1_loss(pred[:,:30]/127,truth,beta=.1,reduction='none')
    return (error*mixed).sum()/mixed.sum().clamp_min(1)

class BoundaryGradientTraining(OrderedBoundaryTraining):
    def __init__(self,args,data):
        super().__init__(args,data);self.net.__class__=BoundaryGradientB1

    def additional_loss(self,p,t,native):
        loss,v=super().additional_loss(p,t,native);code=self.class_labels;after=self.nodes['up1_low_dw']
        if not native:code=code[...,40:-40,40:-40];after=after[...,10:-10,10:-10]
        mixed=mixed_occupancy_loss(after,code);v['mixed_boundary_coverage']=mixed
        return loss+4*mixed,v

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(ordinal_saturation_gradient_leak=.05,mixed_boundary_coverage_weight=4,
          gradient_proxy_inference_ops=0,visual_acceptance=False);torch.save(s,path)

def regression(checkpoint):
    from .probe_b1_encoder_gauge import load
    pre=torch.tensor([-254.,381.],requires_grad=True);value=pre.clamp(0,127)
    old=(value-63.5).square().mean();old.backward();assert pre.grad.abs().sum()==0
    pre=pre.detach().clone().requires_grad_();value=pre.clamp(0,127)
    y=recover_gradient(value,pre);assert torch.equal(y,value)
    ((y-63.5).square().mean()).backward();assert pre.grad[0]<0 and pre.grad[1]>0 and torch.isfinite(pre.grad).all()
    state,net=load(checkpoint);x=torch.rand(1,3,64,80)*1.9-1
    with torch.no_grad():before=net(x,all_nodes=True)
    raw=net.export();net.__class__=BoundaryGradientB1;net.coverage_gain=state['binary_coverage720']['gain']
    after=net(x,all_nodes=True);assert all(torch.equal(v,after[k]) for k,v in before.items()) and net.export()==raw
    after['rgb_head'].square().mean().backward();g=net.weight['res3_conv1__pw'].grad
    assert g is not None and g.norm()>0 and torch.isfinite(g).all()
    code=torch.full((1,16,24),5*81+4*9+4,dtype=torch.long);code[...,9:]=9*81+4*9+4
    truth=F.avg_pool2d(code_bits(code),4);assert mixed_occupancy_loss(truth*127,code)==0
    assert mixed_occupancy_loss((truth>=.5).float()*127,code)>0
    print('BOUNDARY_GRADIENT_REGRESSION_PASS old_saturated_gradient0/new_directional_gradient/forward_bytes_unchanged/mixed_area',flush=True)

if __name__=='__main__':
    import sys
    torch.set_num_threads(2)
    if '--check' in sys.argv:regression(sys.argv[sys.argv.index('--check')+1])
    else:driver.main(training_factory=BoundaryGradientTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
