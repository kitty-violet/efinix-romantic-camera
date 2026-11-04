"""Learn the actual ordered area fields used by a nonnegative pair decoder.

The decoder target is tested by direct injection. Existing res3 and up1 DW
weights are trainable rather than permanently voting away thin boundaries.
All probabilities/labels/flow are training-only; export is unchanged B1.
"""
import copy
from pathlib import Path
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from .train_b1_ternary_packed720 import Packed720Training,driver,preview
from .train_b1_quantum_regions import RowScaledAdam
from .train_b1_winner_surrogate import ExactValueProxyGradient
from .train_b1_symmetric_binary_coverage import evaluate as rgb_evaluate
from .train_b1_joint_region_colour import joint_region_loss
from .train_b1_region_tail_flatness import region_tail_flatness
from .train_b1_packed720_local_rims import local_rim_loss
from .train_b1_boundary_profile import boundary_profile
from .train_b1_contour_coherence import contour_coherence
from .train_b1_ordered_boundary_fields import rank_loss
from .build_b1_thirty_binary_regions import code_bits
from .probe_b1_monotone_pair_decoder import install

class DirectPairB1(m.t.r.c.ColourB1):
    def training_activation_proxy(self,name,value,pre,r):
        if name not in ('res3_conv1.pw','res3_conv2.dw','res3_conv2.pw','up1_low_dw'):return value
        low=0 if r['activation']=='relu' else -128
        clipped=pre.clamp(low,127)
        return ExactValueProxyGradient.apply(value,clipped+.05*(pre-clipped))

def ordinal_loss(field,truth):
    p=field[:,:30]/127
    terms=F.smooth_l1_loss(p,truth,beta=.08,reduction='none')
    frequency=truth.mean((0,2,3),keepdim=True).detach()
    # Equal influence for rare positive and negative predicates, bounded so
    # an absent class does not acquire unbounded weight.
    weight=truth*.5/frequency.clamp_min(.125)+(1-truth)*.5/(1-frequency).clamp_min(.125)
    full=(terms*weight).sum()/weight.sum().clamp_min(1)
    mixed=(truth>0)&(truth<1)
    boundary=(terms*mixed).sum()/mixed.sum().clamp_min(1)
    return full,boundary

class DirectOrdinalPairTraining(Packed720Training):
    def __init__(self,args,data):
        m.AreaTraining.__init__(self,args,data);assert args.qat and not args.freeze_head
        state=torch.load(args.warm,map_location='cpu',weights_only=True)
        assert state.get('monotone_pair_decoder')
        self.setup=copy.deepcopy(state.get('ordinal_pair_setup') or {k:state['binary_coverage720'][k] for k in ('matrix','bias')})
        self.net.__class__=DirectPairB1;self.fixed={};groups=[]
        for key,w in self.net.weight.items():
            fixed=key.startswith(('up1_low_pw','up1_high','up2_','rgb_head')) or key in ('down1__dw','down2__dw')
            b=self.net.bias[key];w.requires_grad_(not fixed);b.requires_grad_(not fixed)
            if fixed:self.fixed[key]=(w.detach().clone(),b.detach().clone())
            else:
                scale=(getattr(self.net,key+'_m')/131072).float().flatten()
                groups.extend([dict(params=[w],lr=args.lr,row_gain=(scale*256)[:,None,None,None]),dict(params=[b],lr=.008)])
        self.opt=RowScaledAdam(groups);self.base_lrs=[g['lr'] for g in groups]
        levels=torch.cartesian_prod(torch.arange(15,device='cuda'),torch.arange(9,device='cuda'),torch.arange(9,device='cuda')).float()
        self.palette=(levels@levels.new_tensor(self.setup['matrix']).T+levels.new_tensor(self.setup['bias'])).clamp(-128,127)/128
        self.pk=tuple(map(tuple,(self.palette.cpu().numpy()*128+128).tolist()))
        def remember(module,inputs,result):
            if isinstance(result,dict):self.nodes=result
        self.net.register_forward_hook(remember)

    def additional_loss(self,p,t,native):
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        region,v=joint_region_loss(p,t,labels);tail=region_tail_flatness(p,labels)
        rim=local_rim_loss(p,t,labels);profile,terms=boundary_profile(p,t,labels)
        if native:
            y=self.rng.randrange(p.shape[-2]-384+1);x=self.rng.randrange(p.shape[-1]-384+1)
            curve=contour_coherence(p[...,y:y+384,x:x+384],t[...,y:y+384,x:x+384])
        else:curve=contour_coherence(p,t)
        field=self.nodes['up1_low_dw'];truth=F.avg_pool2d(code_bits(self.class_labels),4)
        if not native:field=field[...,10:-10,10:-10];truth=truth[...,10:-10,10:-10]
        full,boundary=ordinal_loss(field,truth);rank=rank_loss(field)
        v.update(terms);v.update(region_tail_flatness=tail,mean_centred_local_bright_rim=rim,
          short_contour_turn_excess=curve,ordered_area=full,ordered_boundary_area=boundary,
          ordinal_rank_violation=rank,motion_aligned_target_fraction=self.motion_fraction)
        rgb=.7*region+4*v['region_flatness']+v['adjacent_region_contrast']+4*tail+4*curve+4*rim+4*profile
        return rgb+self.args.classification*(8*full+2*boundary)+10*rank,v

    def preflight_extras(self):
        while self.step%4:self.update(True)
        result=self.update(False);assert result['temporal_pair'] and 'tiny_input_stability' in result['values']
        assert 'ordered_boundary_area' in result['values'];return result

    def save(self,path):
        m.AreaTraining.save(self,path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(monotone_pair_decoder=True,ordinal_pair_setup=self.setup,ordinal_target='actual4x4 native area at up1_low_dw',
          trainable_spatial_prediction=['res3_conv2.dw','res3_conv2.pw','up1_low_dw'],
          extra_inference_operations=0,external_filter=False,visual_acceptance=False)
        torch.save(s,path)

def regression(checkpoint):
    from .probe_b1_encoder_gauge import load
    t=torch.tensor([0.,.25,.5,.75,1.])[None,None,None].expand(1,30,1,5)
    assert sum(ordinal_loss(t*127,t))==0
    bad=(127-t*127).clone().requires_grad_();loss=sum(ordinal_loss(bad,t));loss.backward()
    assert torch.isfinite(bad.grad).all() and bad.grad[...,0].mean()>0 and bad.grad[...,-1].mean()<0
    _,net=load(checkpoint);x=torch.rand(1,3,64,80)*1.9-1
    with torch.no_grad():before=net(x,all_nodes=True)
    raw=net.export();net.__class__=DirectPairB1;after=net(x,all_nodes=True)
    assert all(torch.equal(v,after[k]) for k,v in before.items()) and net.export()==raw
    after['rgb_head'].square().mean().backward()
    for key in ('res3_conv1__pw','res3_conv2__dw','res3_conv2__pw','up1_low_dw'):
        g=net.weight[key].grad;assert g is not None and torch.isfinite(g).all() and g.norm()>0
    print('DIRECT_ORDINAL_PAIR_REGRESSION_PASS exact_target0/directional_gradient/all_nodes0/bytes_same/four_layer_gradient',flush=True)

if __name__=='__main__':
    import sys
    torch.set_num_threads(2)
    if '--check' in sys.argv:regression(sys.argv[sys.argv.index('--check')+1])
    else:driver.main(training_factory=DirectOrdinalPairTraining,extra_sources=[Path(__file__)],evaluation=rgb_evaluate,preview=preview)
