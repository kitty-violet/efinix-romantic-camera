"""TRAIN-calibrated input centering in parameter updates, no inference op.

For y=W*x+b, optimize the coordinates W and c=b+W*mu. Gradients become
gW-gb*mu and gb. Compensate the deployed bias for the change of the actual
quantized weights, not just the latent FP32 weights. This is a QAT adaptation
of centering, not PRONG/whitening or a convergence guarantee.
"""
import torch
from torch.nn import functional as F
from .train_b1_quantum_regions import RowScaledAdam


class CenteredB1Adam(RowScaledAdam):
    def __init__(self,groups,net,means):
        self.net=net
        by_parameter={id(v):k for k,v in net.weight.items()}
        for group in groups:
            if 'row_gain' not in group:continue
            key=by_parameter[id(group['params'][0])]
            group['node_key']=key;group['input_mean']=means[key].detach().clone()
        super().__init__(groups)
        self.prepared=False

    def mu_for(self,key,mean):
        w=self.net.weight[key]
        return mean.reshape(-1,1,1,1) if w.shape[1]==1 and len(mean)==len(w) else mean.reshape(1,-1,1,1)

    @torch.no_grad()
    def prepare_gradients(self):
        assert not self.prepared
        for group in self.param_groups:
            if 'node_key' not in group:continue
            k=group['node_key'];w=self.net.weight[k];b=self.net.bias[k]
            if w.grad is not None and b.grad is not None:
                w.grad.sub_(b.grad[:,None,None,None]*self.mu_for(k,group['input_mean'].to(w.device)))
        self.prepared=True

    @torch.no_grad()
    def step(self,closure=None):
        assert closure is None and self.prepared
        previous={}
        for group in self.param_groups:
            if 'node_key' not in group:continue
            k=group['node_key'];w=self.net.weight[k];s=(getattr(self.net,k+'_m')/131072).float().flatten()[:,None,None,None]
            previous[k]=self.net.round(w/s).clamp(-127,127)*s
        result=super().step()
        for group in self.param_groups:
            if 'node_key' not in group:continue
            k=group['node_key'];w=self.net.weight[k];s=(getattr(self.net,k+'_m')/131072).float().flatten()[:,None,None,None]
            w.copy_(w.maximum(-127*s).minimum(127*s))
            delta=self.net.round(w/s).clamp(-127,127)*s-previous[k]
            self.net.bias[k].sub_((delta*self.mu_for(k,group['input_mean'].to(w.device))).sum((1,2,3)))
        self.prepared=False
        return result


@torch.no_grad()
def calibration_means(net,x):
    nodes=net(x,all_nodes=True);nodes['input']=net.round(x*128).clamp(-128,127)
    return {row['id'].replace('.','__'):nodes[row['inputs'][0]].mean((0,2,3)).detach()
            for row in net.plan['nodes'] if row['id'].replace('.','__') in net.weight}


def regression():
    torch.manual_seed(819)
    for depthwise in (False,True):
        groups=3 if depthwise else 1;ni=3;no=3 if depthwise else 4
        x=torch.randn(2,ni,12,16,dtype=torch.float64)+torch.tensor([20.,50.,90.],dtype=torch.float64)[None,:,None,None]
        mu=x.mean((0,2,3));w=torch.randn(no,1 if depthwise else ni,3,3,dtype=torch.float64,requires_grad=True);b=torch.randn(no,dtype=torch.float64,requires_grad=True)
        y=F.conv2d(F.pad(x,(1,1,1,1),mode='replicate'),w,b,groups=groups);seed=torch.randn_like(y);gw,gb=torch.autograd.grad((y*seed).sum(),(w,b))
        mm=mu[:,None,None,None] if depthwise else mu[None,:,None,None]
        wc=w.detach().clone().requires_grad_();bc=(b.detach()+(w.detach()*mm).sum((1,2,3))).requires_grad_()
        yc=F.conv2d(F.pad(x-mu[None,:,None,None],(1,1,1,1),mode='replicate'),wc,bc,groups=groups)
        g2,b2=torch.autograd.grad((yc*seed).sum(),(wc,bc))
        assert (y-yc).abs().max()<1e-10 and (gw-gb[:,None,None,None]*mm-g2).abs().max()<1e-9 and torch.allclose(gb,b2)
    # Quantized-code update compensation preserves the response to mu exactly
    # before activation, even when FP32 coefficients cross an INT8 threshold.
    from .qat_b1_deploy_preview import Round
    rnd=Round();mu=torch.tensor([17.,63.,99.]);scale=.125;w=torch.tensor([[.02,.12,-.13],[.18,-.32,.12]])
    b=torch.tensor([8.,-9.]);old=rnd(w/scale)*scale;new=rnd((w+.09)/scale)*scale
    corrected=b-((new-old)*mu).sum(1)
    assert torch.equal(old@mu+b,new@mu+corrected)
    print('CENTERED_B1_PASS PW_and_DW_forward_equivalence/gradient_chain_rule/quantized_DC_compensation',flush=True)


if __name__=='__main__':torch.set_num_threads(2);regression()
