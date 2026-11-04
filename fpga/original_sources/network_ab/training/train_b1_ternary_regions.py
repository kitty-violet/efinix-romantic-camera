"""Learn15/9/9 stable colour levels in the unchanged B1 integer graph.

Three-state field likelihoods supervise group totals, invariant to permutations
within a group. Probability DP, labels and smooth derivative proxies exist only
in training. Forward and export are ordinary INT8/Q17 B1, no external filters.
"""
from pathlib import Path
from functools import lru_cache
import itertools,math
import numpy as np
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from . import train_b1_late_region_supervision as driver
from .train_b1_quantum_regions import RowScaledAdam
from .train_b1_winner_surrogate import ExactValueProxyGradient
from .train_b1_clean_regions import CleanRegionTraining
from .train_b1_native_coverage_counts import planned_native
from .train_b1_joint_region_colour import joint_region_loss,evaluate,preview
from .build_b1_ordered_palette import GROUPS
from .probe_b1_decoder_coverage import fragment


def log_states(pre,temperature=.15):
    a=(pre-.5)/temperature;b=(pre-1.5)/temperature
    middle=F.logsigmoid(a)+F.logsigmoid(-b)+math.log1p(-math.exp(-1/temperature))
    return torch.stack((F.logsigmoid(-a),middle,F.logsigmoid(b)),2)


def count_distribution(logp):
    dist=torch.zeros_like(logp[:,:1,0])
    for field in logp.unbind(1):
        options=[F.pad(dist+field[:,k:k+1],(0,0,0,0,k,2-k),value=float('-inf')) for k in range(3)]
        dist=torch.logsumexp(torch.stack(options),0)
    return dist


def field_pre(net,nodes):
    key='up2_low_dw';scale=(getattr(net,key+'_m')/131072).float().flatten()
    w=(net.weight[key]/scale[:,None,None,None]).round().clamp(-127,127)*scale[:,None,None,None]
    b=(net.bias[key]/scale).round()*scale
    return F.conv2d(F.pad(nodes['up1_high_dw'],(1,1,1,1),mode='replicate'),w,b,groups=16)[:,:15]


class TernaryB1(m.t.r.c.ColourB1):
    def training_activation_proxy(self,name,value,pre,r):
        if name!='up1_high_dw':return value
        proxy=8*(F.softplus(pre/8)-F.softplus((pre-127)/8))
        return ExactValueProxyGradient.apply(value,proxy)

    def training_node_proxy(self,name,value,nodes):
        if name!='up2_low_dw':return value
        prob=log_states(field_pre(self,nodes)).exp()
        expected=prob[:,:,1]+2*prob[:,:,2]
        return ExactValueProxyGradient.apply(value,F.pad(expected,(0,0,0,0,0,1)))


@lru_cache(maxsize=64)
def region_codes(target_path,region_path,palette_tuple):
    rgb=m.t.r.read(target_path).astype(np.float32);lab=m.load_region(region_path).astype(np.int64)
    count=np.bincount(lab.ravel()).clip(1)
    means=np.stack([np.bincount(lab.ravel(),weights=rgb[...,c].ravel(),minlength=len(count))/count for c in range(3)],1).astype(np.float32)
    pal=np.asarray(palette_tuple,dtype=np.float32)
    distances=(means*means).sum(1)[:,None]+(pal*pal).sum(1)[None]-2*means@pal.T
    #1215 entries requireuint16. The historic16/200-class helper usesuint8.
    return distances.argmin(1)[lab].astype(np.uint16)


class TernaryRegionTraining(m.AreaTraining):
    preflight_native_resume=True
    input_stability_loss=CleanRegionTraining.input_stability_loss

    def __init__(self,args,data):
        super().__init__(args,data);assert args.qat and not args.freeze_head
        warm=torch.load(args.warm,map_location='cpu',weights_only=True);assert warm.get('ternary_region_decoder')
        self.gain=warm['ternary_gain'];self.net.__class__=TernaryB1
        self.fixed={};groups=[];fixed=False
        for key,w in self.net.weight.items():
            if key=='up1_high_dw':fixed=True
            b=self.net.bias[key];w.requires_grad_(not fixed);b.requires_grad_(not fixed)
            if fixed:self.fixed[key]=(w.detach().clone(),b.detach().clone())
            else:
                s=(getattr(self.net,key+'_m')/131072).float().flatten()
                groups.extend([dict(params=[w],lr=args.lr,row_gain=(s*256)[:,None,None,None]),dict(params=[b],lr=.002)])
        self.opt=RowScaledAdam(groups);self.base_lrs=[g['lr'] for g in groups]
        with torch.no_grad():
            levels=torch.cartesian_prod(torch.arange(15,device='cuda'),torch.arange(9,device='cuda'),torch.arange(9,device='cuda'))
            z=torch.zeros(len(levels),16,1,1,device='cuda')
            for c,(a,b) in enumerate(GROUPS):z[:,a:b,0,0]=(levels[:,c,None]-2*torch.arange(b-a,device='cuda')[None]).clamp(0,2)
            self.palette=fragment(self.net,z,'up2_low_dw')['rgb_head'][:,:,0,0]/128
            self.pk=tuple(map(tuple,(self.palette.cpu().numpy()*128+128).tolist()))
        def remember(module,inputs,result):
            if isinstance(result,dict):self.nodes=result
        self.net.register_forward_hook(remember)

    def batch_style(self,native):
        x,target,specs,flow,valid=super().batch_style(native);codes=[]
        for spec in specs:
            row=self.by_id.get(spec['id']) or self.by_stem[spec['id']]
            a=region_codes(row['target'],row['region'],self.pk);xx,yy=spec['xy'];w,h=spec['wh'];a=a[yy:yy+h,xx:xx+w]
            if spec['flip']:a=a[:,::-1]
            codes.append(torch.from_numpy(np.ascontiguousarray(a.astype(np.int64))).cuda())
        self.class_labels=torch.stack(codes)
        return x,target,specs,flow,valid

    @staticmethod
    def appearance_loss(p,t,mode):return p.sum()*0,{}

    def additional_loss(self,p,target,native):
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        region,values=joint_region_loss(p,target,labels)
        code=self.class_labels;levels=(code//81,code//9%9,code%9)
        logits=log_states(field_pre(self.net,self.nodes));losses=[]
        for (a,b),level,n in zip(GROUPS,levels,(15,9,9)):
            truth=F.avg_pool2d(F.one_hot(level,n).permute(0,3,1,2).float(),2)
            logp=count_distribution(logits[:,a:b])
            if not native:truth=truth[...,20:-20,20:-20];logp=logp[...,20:-20,20:-20]
            freq=truth.mean((0,2,3),keepdim=True);w=freq.clamp_min(1/(4*n)).rsqrt()
            w=(w/(freq*w).sum()).clamp(.25,4).detach();weighted=truth*w
            losses.append(-(weighted*logp).sum()/weighted.sum().clamp_min(1))
        cls=torch.stack(losses).mean();values['ternary_count_ce']=cls
        return .7*region+self.args.classification*cls,values

    def update(self,native=None):
        result=super().update(planned_native(self.step) if native is None else native)
        with torch.no_grad():
            for key,w in self.net.weight.items():
                if w.requires_grad:
                    lim=127*(getattr(self.net,key+'_m')/131072).float().flatten()[:,None,None,None]
                    w.copy_(w.maximum(-lim).minimum(lim))
            for key,(w,b) in self.fixed.items():assert torch.equal(w,self.net.weight[key]) and torch.equal(b,self.net.bias[key])
        return result

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(ternary_region_decoder=True,ternary_gain=self.gain,ternary_group_levels=[15,9,9],
            extra_inference_operations=0,minimum_area_inference_guarantee=False,visual_acceptance=False,
            probabilities_training_only=True,original_target_rgb_preserved=True)
        torch.save(s,path)


def regression(checkpoint):
    torch.manual_seed(319)
    x=torch.randn(1,3,1,1,dtype=torch.float64,requires_grad=True)
    lp=log_states(x);assert torch.allclose(lp.exp().sum(2),torch.ones_like(x),atol=1e-12)
    dist=count_distribution(lp).exp();brute=torch.zeros_like(dist)
    for states in itertools.product(range(3),repeat=3):
        probability=torch.ones_like(x[:,0])
        for i,k in enumerate(states):probability*=lp[:,i,k].exp()
        brute[:,sum(states)]+=probability
    assert torch.allclose(dist,brute,atol=1e-12,rtol=0)
    assert torch.allclose(count_distribution(lp[:,[2,0,1]]),dist.log(),atol=1e-10,rtol=0)
    assert torch.autograd.gradcheck(lambda a:count_distribution(log_states(a)),(x,),eps=1e-6,atol=1e-5)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
    xx=torch.randn(1,3,64,64).clamp(-1,127/128)
    with torch.no_grad():reference=net(xx,all_nodes=True)
    raw=net.export();net.__class__=TernaryB1;out=net(xx,all_nodes=True)
    assert all(torch.equal(v,out[k]) for k,v in reference.items()) and raw==net.export()
    out['rgb_head'].square().mean().backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    assert net.weight['up1_low_pw'].grad.abs().sum()>0
    print('TERNARY_REGRESSION_PASS probability/count/permutation/gradient/forward/bytes/backward',flush=True)


if __name__=='__main__':
    import sys
    if '--check' in sys.argv:regression(sys.argv[sys.argv.index('--check')+1])
    else:driver.main(training_factory=TernaryRegionTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
