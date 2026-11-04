"""Fixed TRAIN patch-covariance coordinates, with no inference transform.

W=A P, c=b+W mu. Adam stores moments for A and c. The deployed W,b remain
ordinary B1 coefficients. P is damped and RMS-normalized; this is not the
complete PRONG algorithm and carries no convergence guarantee.
Reference: https://arxiv.org/abs/1507.00210
"""
import torch
from torch.nn import functional as F
from .train_b1_quantum_regions import RowScaledAdam


def transform(v,p):
    return (v[:,None,:]@p).squeeze(1) if p.ndim==3 else v@p


@torch.no_grad()
def patch_statistics(net,x,count=2048):
    nodes=net(x,all_nodes=True);nodes['input']=net.round(x*128).clamp(-128,127);stats={}
    gen=torch.Generator().manual_seed(2026100722)
    for row in net.plan['nodes']:
        key=row['id'].replace('.','__')
        if key not in net.weight or not net.weight[key].requires_grad:continue
        v=nodes[row['inputs'][0]];stride=row['stride'];kernel=row['kernel'];h,w=nodes[row['id']].shape[-2:]
        n=min(count,x.shape[0]*h*w);ids=torch.randperm(x.shape[0]*h*w,generator=gen)[:n].to(x.device)
        bb=ids//(h*w);yy=ids%(h*w)//w*stride;xx=ids%w*stride
        v=F.pad(v,(1,1,1,1),mode='replicate') if kernel==3 else v
        patch=torch.stack([v[bb,:,yy+dy,xx+dx] for dy in range(kernel) for dx in range(kernel)],-1).double()
        if row['kind']=='dw3x3':patch=patch.permute(1,0,2)
        else:patch=patch.flatten(1)
        mu=patch.mean(-2);d=patch-mu.unsqueeze(-2);cov=d.transpose(-1,-2)@d/n
        stats[key]=dict(mean=mu.float(),covariance=cov.float())
    return stats


def whitener(cov,damping=.05):
    val,vec=torch.linalg.eigh(cov.double());val=val.clamp_min(0)
    scale=val.mean(-1,keepdim=True).clamp_min(1e-6)
    gain=(val/scale+damping).rsqrt()
    gain=gain/gain.square().mean(-1,keepdim=True).sqrt()
    p=(vec*gain.unsqueeze(-2))@vec.transpose(-1,-2)
    return p.float(),dict(relative_eigen_min=(val/scale).amin(-1),relative_eigen_max=(val/scale).amax(-1),
        retained_rank=(val>.01*scale).sum(-1),dimension=val.shape[-1])


class WhitenedB1Adam(RowScaledAdam):
    def __init__(self,groups,net,stats,enabled=True):
        self.net=net;self.prepared=False;keys={id(v):k for k,v in net.weight.items()}
        for g in groups:
            if 'row_gain' not in g:continue
            if g.get('unwhitened',False):continue
            k=keys[id(g['params'][0])];s=stats[k];p,_=whitener(s['covariance'])
            if not enabled:p=torch.eye(p.shape[-1],device=p.device).expand_as(p).clone()
            g.update(node_key=k,patch_mean=s['mean'].clone(),input_whitener=p)
        super().__init__(groups)

    @torch.no_grad()
    def prepare_gradients(self):
        assert not self.prepared
        for g in self.param_groups:
            if 'node_key' not in g:continue
            k=g['node_key'];w=self.net.weight[k];b=self.net.bias[k]
            if w.grad is None or b.grad is None:continue
            gw=w.grad.flatten(1)-b.grad[:,None]*g['patch_mean'].to(w.device)
            p=g['input_whitener'].to(w.device)
            w.grad.copy_(transform(gw,p.transpose(-1,-2)).reshape_as(w))
        self.prepared=True

    @torch.no_grad()
    def step(self,closure=None):
        assert closure is None and self.prepared;previous={}
        for g in self.param_groups:
            if 'node_key' in g:previous[g['node_key']]=self.net.weight[g['node_key']].clone()
        result=super().step()
        for g in self.param_groups:
            if 'node_key' not in g:continue
            k=g['node_key'];w=self.net.weight[k];old=previous[k]
            # super made an Adam A-step in W's storage; map only its delta.
            delta=transform((w-old).flatten(1),g['input_whitener'].to(w.device)).reshape_as(w)
            s=(getattr(self.net,k+'_m')/131072).float().flatten()[:,None,None,None]
            w.copy_((old+delta).maximum(-127*s).minimum(127*s))
            deployed=(self.net.round(w/s).clamp(-127,127)-self.net.round(old/s).clamp(-127,127))*s
            self.net.bias[k].sub_((deployed.flatten(1)*g['patch_mean'].to(w.device)).sum(1))
        self.prepared=False
        return result


def regression():
    torch.manual_seed(431)
    for depthwise in (False,True):
        no,ni,n=4,9,37
        x=torch.randn(no if depthwise else 1,n,ni,dtype=torch.float64)+3
        cov=(x-x.mean(1,keepdim=True)).transpose(-1,-2)@(x-x.mean(1,keepdim=True))/n
        p,_=whitener(cov if depthwise else cov[0]);p=p.double();mu=x.mean(1) if depthwise else x[0].mean(0)
        a=torch.randn(no,ni,dtype=torch.float64,requires_grad=True);c=torch.randn(no,dtype=torch.float64,requires_grad=True)
        w=transform(a,p);b=c-(w*mu).sum(1)
        z=torch.einsum('oi,oni->on',w,x.expand(no,-1,-1))+b[:,None]
        zz=torch.einsum('oi,oni->on',a,torch.einsum('oij,onj->oni',p if depthwise else p.expand(no,-1,-1),x.expand(no,-1,-1)-mu[:,None] if depthwise else x.expand(no,-1,-1)-mu)) +c[:,None]
        assert torch.allclose(z,zz,atol=1e-10)
        seed=torch.randn_like(z);ga,gc=torch.autograd.grad((z*seed).sum(),(a,c))
        gw=torch.einsum('on,oni->oi',seed,x.expand(no,-1,-1));gb=seed.sum(1)
        expected=transform(gw-gb[:,None]*mu,p.transpose(-1,-2))
        assert torch.allclose(ga,expected,atol=1e-9) and torch.allclose(gc,gb)
        pp=p@p.transpose(-1,-2);assert torch.allclose(pp.diagonal(dim1=-2,dim2=-1).sum(-1),torch.full_like(pp[...,0,0],ni),atol=1e-5)
    print('WHITENED_COORDINATES_PASS PW_DW_forward/chain_rule/RMS_gain',flush=True)


if __name__=='__main__':torch.set_num_threads(2);regression()
