"""Keep corrective derivatives across all three colour-decision saturations.

The forward is still the exact B1 integer graph. Prior sigmoid proxies have
tiny derivatives for confident-but-wrong fields. Use the actual clipping ramp
with a small training-only leakage instead, including the signed PW/ReLU pair.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_recover_boundary_gradients import BoundaryGradientTraining,recover_gradient,driver,evaluate,preview
from .train_b1_recover_boundary_gradients import BoundaryGradientB1
from .train_b1_winner_surrogate import ExactValueProxyGradient

class CompleteBoundaryGradientB1(BoundaryGradientB1):
    def training_activation_proxy(self,name,value,pre,r):
        if name in ('res3_conv1.pw','up2_low_dw'):return recover_gradient(value,pre)
        return super().training_activation_proxy(name,value,pre,r)

    def training_node_proxy(self,name,value,nodes):
        if name!='up1_low_dw':return value
        key='res3_conv2__pw';idx=torch.arange(30,device=value.device)
        gain=self.weight[key][idx,idx,0,0][None,:,None,None]
        pre=nodes['res3_conv2.dw'][:,:30]*gain+self.bias[key][:30][None,:,None,None]
        clipped=pre.clamp(0,127);proxy=clipped+.05*(pre-clipped)
        return ExactValueProxyGradient.apply(value,F.pad(proxy,(0,0,0,0,0,2)))

class CompleteBoundaryGradientTraining(BoundaryGradientTraining):
    def __init__(self,args,data):
        super().__init__(args,data);self.net.__class__=CompleteBoundaryGradientB1

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(complete_boundary_gradient_proxy=['res3_conv1.pw','up1_low_dw','up2_low_dw'],
          hard_forward_unchanged=True,visual_acceptance=False);torch.save(s,path)

def regression(checkpoint):
    from .probe_b1_encoder_gauge import load
    x=torch.tensor([0.,127.],requires_grad=True);old=127*torch.sigmoid((x-63.5)/6)
    old.sum().backward();prior=x.grad.clone();x=x.detach().requires_grad_()
    pre=(x-63.5)*3+63.5;new=recover_gradient(pre.clamp(0,127),pre);new.sum().backward()
    assert (x.grad>prior*100).all() and torch.allclose(x.grad,torch.full_like(x,.15))
    state,net=load(checkpoint);source=torch.rand(1,3,64,80)*1.9-1
    with torch.no_grad():before=net(source,all_nodes=True)
    raw=net.export();net.__class__=CompleteBoundaryGradientB1;net.coverage_gain=state['binary_coverage720']['gain']
    after=net(source,all_nodes=True);assert all(torch.equal(v,after[k]) for k,v in before.items()) and net.export()==raw
    after['rgb_head'].square().mean().backward();g=net.weight['res3_conv1__pw'].grad
    assert g is not None and g.norm()>0 and torch.isfinite(g).all()
    print('COMPLETE_BOUNDARY_GRADIENT_PASS',dict(previous_extreme_derivative=prior.tolist(),new_extreme_derivative=x.grad.tolist(),all_nodes_error=0,export_bytes_unchanged=True),flush=True)

if __name__=='__main__':
    import sys
    torch.set_num_threads(2)
    if '--check' in sys.argv:regression(sys.argv[sys.argv.index('--check')+1])
    else:driver.main(training_factory=CompleteBoundaryGradientTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
