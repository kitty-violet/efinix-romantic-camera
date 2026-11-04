"""Learn weighted binary colour regions with symmetric continuous coverage.

The forward graph is the original B1, with exact byte-replayable rounding.
Weighted count probabilities and canonical score targets exist only in training.
"""
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from .train_b1_ternary_packed720 import Packed720Training,driver,preview,bright_envelope_loss
from .train_b1_quantum_regions import RowScaledAdam
from .train_b1_winner_surrogate import ExactValueProxyGradient
from .b1_binary_layout import fields_for_layout
from .train_b1_joint_region_colour import joint_region_loss
from .train_b1_region_tail_flatness import region_tail_flatness,evaluate as base_evaluate
from .train_b1_packed720_local_rims import local_rim_loss
from .train_b1_boundary_profile import boundary_profile
from .train_b1_contour_coherence import contour_coherence


def weighted_logp(logits,weights):
    dist=torch.zeros_like(logits[:,:1])
    for j,w in enumerate(weights):
        no=F.pad(dist+F.logsigmoid(-logits[:,j:j+1]),(0,0,0,0,0,w),value=float('-inf'))
        yes=F.pad(dist+F.logsigmoid(logits[:,j:j+1]),(0,0,0,0,w,0),value=float('-inf'))
        dist=torch.logaddexp(no,yes)
    return dist


class BinaryCoverageB1(m.t.r.c.ColourB1):
    def training_activation_proxy(self,name,value,pre,r):
        if name!='up2_low_dw':return value
        proxy=127*torch.sigmoid((pre[:,:15]-63.5)/(self.coverage_gain*6))
        return ExactValueProxyGradient.apply(value,F.pad(proxy,(0,0,0,0,0,1)))


class BinaryCoverageTraining(Packed720Training):
    def __init__(self,args,data):
        m.AreaTraining.__init__(self,args,data);assert args.qat and not args.freeze_head
        state=torch.load(args.warm,map_location='cpu',weights_only=True);self.setup=state['binary_coverage720']
        assert not self.setup['source_readout_not_fitted']
        self.net.binary_layout_setup=self.setup
        self.net.__class__=BinaryCoverageB1;self.net.coverage_gain=self.setup['gain'];self.fixed={};groups=[];tail=False
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

    def target_fields(self,code):
        return fields_for_layout(code,self.setup)

    def additional_loss(self,p,t,native):
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        region,v=joint_region_loss(p,t,labels);tail=region_tail_flatness(p,labels)
        rim=local_rim_loss(p,t,labels);profile,terms=boundary_profile(p,t,labels)
        if native:
            yy=self.rng.randrange(0,p.shape[-2]-384+1);xx=self.rng.randrange(0,p.shape[-1]-384+1)
            curve=contour_coherence(p[...,yy:yy+384,xx:xx+384],t[...,yy:yy+384,xx:xx+384])
        else:curve=contour_coherence(p,t)
        code=self.class_labels;fields=self.target_fields(code)
        logits=(self.nodes['up1_high_dw'][:,:15]-64)/6
        truth_bits=F.avg_pool2d(fields,2);wanted180=F.avg_pool2d(fields*96+16,4)
        readout=self.nodes['up1_low_pw'][:,:15]
        if not native:
            logits=logits[...,20:-20,20:-20];truth_bits=truth_bits[...,20:-20,20:-20]
            wanted180=wanted180[...,10:-10,10:-10];readout=readout[...,10:-10,10:-10]
        losses=[]
        offset=0
        for ww,truth in zip(self.setup['weights'],(code//81,code//9%9,code%9)):
            n=sum(ww)+1;wanted=F.avg_pool2d(F.one_hot(truth,n).permute(0,3,1,2).float(),2)
            if not native:wanted=wanted[...,20:-20,20:-20]
            logp=weighted_logp(logits[:,offset:offset+len(ww)],ww);offset+=len(ww)
            freq=wanted.mean((0,2,3),keepdim=True);w=freq.clamp_min(1/(4*n)).rsqrt()
            w=(w/(freq*w).sum()).clamp(.25,4).detach();weighted=wanted*w
            losses.append(-(weighted*logp).sum()/weighted.sum().clamp_min(1))
        count=torch.stack(losses).mean();canonical=F.binary_cross_entropy_with_logits(logits,truth_bits)
        margin=F.smooth_l1_loss(readout/128,wanted180/128,beta=.05)
        cls=.5*count+.5*canonical+.2*margin
        v.update(terms);v.update(weighted_binary_count_ce=count,canonical_binary_ce=canonical,binary_score_margin=margin,
            region_tail_flatness=tail,mean_centred_local_bright_rim=rim,short_contour_turn_excess=curve,
            motion_aligned_target_fraction=self.motion_fraction)
        return .7*region+4*v['region_flatness']+v['adjacent_region_contrast']+4*tail+4*curve+4*rim+4*profile+self.args.classification*cls,v

    def preflight_extras(self):
        while self.step%4:self.update(True)
        result=self.update(False)
        assert result['temporal_pair'] and 'tiny_input_stability' in result['values']
        assert result['values']['weighted_binary_count_ce']>0
        return result

    def save(self,path):
        m.AreaTraining.save(self,path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(binary_coverage720=self.setup,aggregate720=False,ternary_packed720=False,ternary_region_decoder=False,
            source_readout_not_fitted=False,full_boundary_profile_training=True,
            probabilities_training_only=True,added_inference_ops=0,external_filter=False,visual_acceptance=False)
        torch.save(state,path)


@torch.inference_mode()
def evaluate(net,rows):
    result=base_evaluate(net,rows)
    for record,row in zip(result['records'],rows):
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];t=m.t.r.tensor(m.t.r.read(row['target']))[None]
        labels=torch.as_tensor(m.load_region(row['region']).astype('int64'),device=x.device)[None]
        p=net(x)[0];_,terms=boundary_profile(p,t,labels)
        record.update({k:float(v) for k,v in terms.items()});record['bright_envelope']=float(bright_envelope_loss(p,t))
    for k in ('rgb_profile_shape','rgb_profile_contrast','profile_samples','bright_envelope'):
        result['mean'][k]=sum(r[k] for r in result['records'])/len(rows)
    mean=result['mean'];mean['previous_selection_score']=mean['score']
    mean['score']+=4*(mean['rgb_profile_shape']+mean['rgb_profile_contrast'])-2*mean['joint_rgb_edge_vector']
    return result


def regression(checkpoint):
    from .probe_b1_encoder_gauge import load
    import itertools
    state,net=load(checkpoint)
    for ww in state['binary_coverage720']['weights']:
        z=torch.randn(1,len(ww),1,1,dtype=torch.float64,requires_grad=True);dist=weighted_logp(z,ww).exp();brute=torch.zeros_like(dist)
        for bits in itertools.product((0,1),repeat=len(ww)):
            probability=z.new_ones((1,1,1))
            for j,b in enumerate(bits):probability=probability*torch.sigmoid(z[:,j] if b else -z[:,j])
            brute[:,sum(w*b for w,b in zip(ww,bits))]+=probability
        assert torch.allclose(dist,brute,atol=1e-12) and torch.allclose(dist.sum(1),torch.ones_like(dist[:,0]),atol=1e-12)
        assert torch.autograd.gradcheck(lambda x:weighted_logp(x,ww),(z,),atol=1e-5)
    x=torch.rand(1,3,64,80)*1.9-1
    with torch.no_grad():before=net(x,all_nodes=True)
    raw=net.export();net.__class__=BinaryCoverageB1;net.coverage_gain=state['binary_coverage720']['gain']
    after=net(x,all_nodes=True);assert all(torch.equal(v,after[k]) for k,v in before.items()) and net.export()==raw
    after['rgb_head'].square().mean().backward()
    assert net.weight['up1_low_pw'].grad.norm()>0 and all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    print('SYMMETRIC_BINARY_TRAIN_REGRESSION_PASS enumeration/probability/gradcheck/forward/bytes/backward',flush=True)


if __name__=='__main__':
    import sys
    torch.set_num_threads(2)
    if '--check' in sys.argv:regression(sys.argv[sys.argv.index('--check')+1])
    else:driver.main(training_factory=BinaryCoverageTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
