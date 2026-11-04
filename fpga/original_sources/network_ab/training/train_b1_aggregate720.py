"""Train aggregate colour totals, each rounded only once at720.

Same B1 operators and parameter-image layout. Region labels, shared edge
profiles, likelihoods and synthetic boundaries are training-only supervision.
"""
import math
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from .train_b1_ternary_regions import TernaryB1
from .train_b1_ternary_packed720 import pre720,log_values,bright_envelope_loss,driver,preview
from .train_b1_packed720_step_guard import StepGuardTraining
from .train_b1_packed720_local_rims import local_rim_loss
from .train_b1_quantum_regions import RowScaledAdam
from .train_b1_winner_surrogate import ExactValueProxyGradient
from .train_b1_joint_region_colour import joint_region_loss
from .train_b1_region_tail_flatness import region_tail_flatness,evaluate as base_evaluate
from .train_b1_contour_coherence import contour_coherence
from .train_b1_boundary_profile import boundary_profile
from .minimum_area_blocks import components

MAXIMA=(14,8,8)


class AggregateB1(TernaryB1):
    def training_node_proxy(self,name,value,nodes):
        if name!='up2_high_dw':return value
        pre=pre720(self,nodes);expected=[]
        for j,maximum in enumerate(MAXIMA):
            prob=log_values(pre[:,j],maximum).exp()
            expected.append((prob*torch.arange(maximum+1,device=pre.device)[None,:,None,None]).sum(1))
        return ExactValueProxyGradient.apply(value,F.pad(torch.stack(expected,1),(0,0,0,0,0,5)))


class AggregateTraining(StepGuardTraining):
    def __init__(self,args,data):
        m.AreaTraining.__init__(self,args,data);assert args.qat and not args.freeze_head
        state=torch.load(args.warm,map_location='cpu',weights_only=True);self.setup=state['aggregate720']
        self.net.__class__=AggregateB1;self.fixed={};groups=[];tail=False
        for key,w in self.net.weight.items():
            if key=='up1_high_dw':tail=True
            fixed=tail or key in ('down1__dw','down2__dw');b=self.net.bias[key]
            w.requires_grad_(not fixed);b.requires_grad_(not fixed)
            if fixed:self.fixed[key]=(w.detach().clone(),b.detach().clone())
            else:
                s=(getattr(self.net,key+'_m')/131072).float().flatten()
                groups.extend([dict(params=[w],lr=args.lr,row_gain=(s*256)[:,None,None,None]),dict(params=[b],lr=.002)])
        self.opt=RowScaledAdam(groups);self.base_lrs=[g['lr'] for g in groups]
        levels=torch.cartesian_prod(torch.arange(15,device='cuda'),torch.arange(9,device='cuda'),torch.arange(9,device='cuda')).float()
        self.palette=(levels@levels.new_tensor(self.setup['matrix']).T+levels.new_tensor(self.setup['bias'])).clamp(-128,127)/128
        self.pk=tuple(map(tuple,(self.palette.cpu().numpy()*128+128).tolist()))
        def remember(module,inputs,result):
            if isinstance(result,dict):self.nodes=result
        self.net.register_forward_hook(remember)
        rng=np.random.default_rng(20261007193)
        colours=np.concatenate((np.repeat(np.arange(16,241,16)[:,None],3,axis=1),rng.integers(16,241,(81,3))),axis=0)
        self.edge_colours=torch.tensor(colours,device='cuda',dtype=torch.float32)
        with torch.no_grad():self.edge_anchors=(self.net(self.edge_colours[:,:,None,None].expand(-1,-1,16,16)/128-1)[0][:,:,8,8]*128+128).detach()

    def additional_loss(self,p,t,native):
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        region,v=joint_region_loss(p,t,labels);tail=region_tail_flatness(p,labels)
        rim=local_rim_loss(p,t,labels);profile,terms=boundary_profile(p,t,labels)
        if native:
            yy=self.rng.randrange(0,p.shape[-2]-384+1);xx=self.rng.randrange(0,p.shape[-1]-384+1)
            curve=contour_coherence(p[...,yy:yy+384,xx:xx+384],t[...,yy:yy+384,xx:xx+384])
        else:curve=contour_coherence(p,t)
        pre=pre720(self.net,self.nodes);code=self.class_labels
        if native:
            yy,xx=self.rng.randrange(2),self.rng.randrange(2);pre=pre[...,yy::2,xx::2];code=code[...,yy::2,xx::2]
        else:pre=pre[...,40:-40,40:-40];code=code[...,40:-40,40:-40]
        losses=[]
        for j,truth in enumerate((code//81,code//9%9,code%9)):
            n=MAXIMA[j]+1;logp=log_values(pre[:,j],MAXIMA[j]);freq=torch.bincount(truth.flatten(),minlength=n).float()/truth.numel()
            w=freq.clamp_min(1/(4*n)).rsqrt();w=(w/(freq*w).sum()).clamp(.25,4).detach()[truth]
            losses.append((-logp.gather(1,truth[:,None])[:,0]*w).sum()/w.sum().clamp_min(1))
        cls=torch.stack(losses).mean()
        v.update(terms);v.update(aggregate_count_ce=cls,region_tail_flatness=tail,mean_centred_local_bright_rim=rim,
            short_contour_turn_excess=curve,motion_aligned_target_fraction=self.motion_fraction)
        return .7*region+4*v['region_flatness']+v['adjacent_region_contrast']+4*tail+4*curve+4*rim+4*profile+self.args.classification*cls,v

    def preflight_extras(self):
        while self.step%4:self.update(True)
        result=self.update(False)
        assert result['temporal_pair'] and 'tiny_input_stability' in result['values'] and 'synthetic_bright_rim' in result['values']
        assert result['values']['aggregate_count_ce']>0
        return result

    def save(self,path):
        m.AreaTraining.save(self,path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(aggregate720=self.setup,ternary_packed720=False,ternary_region_decoder=False,
            synthetic_boundary_guard=dict(interval=4,batch=3,colours=self.edge_colours.cpu(),immutable_warm_anchors=self.edge_anchors.cpu()),
            full_boundary_profile_training=True,probabilities_training_only=True,added_inference_ops=0,
            external_filter=False,visual_acceptance=False)
        torch.save(state,path)


@torch.inference_mode()
def evaluate(net,rows):
    result=base_evaluate(net,rows)
    for record,row in zip(result['records'],rows):
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];t=m.t.r.tensor(m.t.r.read(row['target']))[None]
        labels=torch.as_tensor(m.load_region(row['region']).astype('int64'),device=x.device)[None]
        nodes=net(x,all_nodes=True);p=nodes['rgb_head']/128;_,terms=boundary_profile(p,t,labels)
        record.update({k:float(v) for k,v in terms.items()});record['bright_envelope']=float(bright_envelope_loss(p,t))
        z=nodes['up2_high_dw'][0];code=(81*z[0]+9*z[1]+z[2]).long().cpu().numpy()
        areas=np.bincount(components(code).ravel());record.update(regions720=len(areas),small_region_pixels=float(areas[areas<256].sum()/areas.sum()))
    for k in ('rgb_profile_shape','rgb_profile_contrast','profile_samples','bright_envelope','regions720','small_region_pixels'):
        result['mean'][k]=sum(r[k] for r in result['records'])/len(rows)
    mean=result['mean'];mean['previous_selection_score']=mean['score']
    mean['score']+=4*(mean['rgb_profile_shape']+mean['rgb_profile_contrast'])-2*mean['joint_rgb_edge_vector']
    return result


def regression(checkpoint):
    from .probe_b1_encoder_gauge import load
    for mx in MAXIMA:
        x=torch.linspace(-3,mx+3,201,dtype=torch.float64)
        assert torch.allclose(log_values(x,mx).exp().sum(1),torch.ones_like(x),atol=1e-12)
    _,net=load(checkpoint);x=torch.linspace(-1,127/128,64*80).reshape(1,1,64,80).expand(1,3,-1,-1)
    with torch.no_grad():before=net(x,all_nodes=True)
    raw=net.export();net.__class__=AggregateB1;after=net(x,all_nodes=True)
    assert all(torch.equal(v,after[k]) for k,v in before.items()) and raw==net.export()
    after['rgb_head'].square().mean().backward()
    assert net.weight['up1_low_pw'].grad.norm()>0 and all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    print('AGGREGATE720_REGRESSION_PASS probability/forward/bytes/backward',flush=True)


if __name__=='__main__':
    import sys
    torch.set_num_threads(2)
    if '--check' in sys.argv:regression(sys.argv[sys.argv.index('--check')+1])
    else:driver.main(training_factory=AggregateTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
