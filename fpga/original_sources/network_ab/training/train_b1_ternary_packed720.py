"""Learn final colour counts on the real720 grid, in the unchanged B1.

The previous360 count proxy is deliberately replaced, not reused: the last
eight existing DW rows now quantize paired fields into0..4/0..2 at720p.
All probability, region, bright-envelope and flow operations are training-only.
"""
import math
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from . import train_b1_late_region_supervision as driver
from .train_b1_quantum_regions import RowScaledAdam
from .train_b1_ternary_regions import TernaryB1,region_codes
from .train_b1_winner_surrogate import ExactValueProxyGradient
from .train_b1_joint_region_colour import joint_region_loss,preview
from .train_b1_region_tail_flatness import region_tail_flatness,evaluate as base_evaluate
from .train_b1_joint_rgb_edges import rgb_edge_vector_loss
from .train_b1_contour_coherence import contour_coherence
from .train_b1_clean_regions import CleanRegionTraining
from .train_b1_real_region_texture import texture_intervention,align_pair
from .train_b1_native_coverage_counts import planned_native
from .minimum_area_blocks import components,boundary_mode

MAXIMA=(4,4,4,2,4,4,4,4)
GROUPS=((0,4),(4,6),(6,8))


def log_values(pre,maximum,temperature=.15):
    out=[F.logsigmoid((.5-pre)/temperature)]
    for value in range(1,maximum):
        out.append(F.logsigmoid((pre-(value-.5))/temperature)+F.logsigmoid(((value+.5)-pre)/temperature)+math.log1p(-math.exp(-1/temperature)))
    out.append(F.logsigmoid((pre-(maximum-.5))/temperature))
    return torch.stack(out,1)


def count_logp(pre,start,end):
    dist=torch.zeros_like(pre[:,:1])
    for j in range(start,end):
        logp=log_values(pre[:,j],MAXIMA[j]);options=[]
        for k in range(MAXIMA[j]+1):options.append(F.pad(dist+logp[:,k:k+1],(0,0,0,0,k,MAXIMA[j]-k),value=float('-inf')))
        dist=torch.logsumexp(torch.stack(options),0)
    return dist


def pre720(net,nodes):
    key='up2_high_dw';row=next(r for r in net.plan['nodes'] if r['id']==key);s=(getattr(net,key+'_m')/131072).float().flatten()
    w=net.round(net.weight[key]/s[:,None,None,None]).clamp(-127,127)*s[:,None,None,None];b=net.round(net.bias[key]/s)*s
    return F.conv2d(F.pad(nodes[row['inputs'][0]],(1,1,1,1),mode='replicate'),w,b,groups=8)


class Packed720B1(TernaryB1):
    def training_node_proxy(self,name,value,nodes):
        if name!='up2_high_dw':return value
        pre=pre720(self,nodes);expected=[]
        for j,maximum in enumerate(MAXIMA):
            prob=log_values(pre[:,j],maximum).exp();expected.append((prob*torch.arange(maximum+1,device=pre.device)[None,:,None,None]).sum(1))
        return ExactValueProxyGradient.apply(value,torch.stack(expected,1))


def bright_envelope_loss(p,t):
    y=lambda a:a[:,0:1]*.25+a[:,1:2]*.5+a[:,2:3]*.25
    upper=F.max_pool2d(y(t),9,1,4);excess=F.relu(y(p)-upper-4/128)[...,4:-4,4:-4]
    # Worst5percent prevents thin bright rims disappearing in a whole-frame
    # mean. Local reference admits a small contour displacement and real white.
    top=excess.flatten().topk(max(1,int(excess.numel()*.05))).values
    return (top.square().mean()+1e-12).sqrt()-1e-6


class Packed720Training(m.AreaTraining):
    preflight_native_resume=True
    @staticmethod
    def appearance_loss(p,t,mode):return p.sum()*0,{}

    def __init__(self,args,data):
        super().__init__(args,data);assert args.qat and not args.freeze_head
        state=torch.load(args.warm,map_location='cpu',weights_only=True);assert state.get('ternary_packed720')
        self.setup=state['packed720'];self.net.__class__=Packed720B1;self.fixed={};groups=[];tail=False
        for k,w in self.net.weight.items():
            if k=='up1_high_dw':tail=True
            fixed=tail or k in ('down1__dw','down2__dw');b=self.net.bias[k]
            w.requires_grad_(not fixed);b.requires_grad_(not fixed)
            if fixed:self.fixed[k]=(w.detach().clone(),b.detach().clone())
            else:
                s=(getattr(self.net,k+'_m')/131072).float().flatten()
                groups.extend([dict(params=[w],lr=args.lr,row_gain=(s*256)[:,None,None,None]),dict(params=[b],lr=.002)])
        self.opt=RowScaledAdam(groups);self.base_lrs=[g['lr'] for g in groups]
        levels=torch.cartesian_prod(torch.arange(15,device='cuda'),torch.arange(9,device='cuda'),torch.arange(9,device='cuda')).float()
        mat=torch.tensor(self.setup['matrix'],device='cuda');bias=torch.tensor(self.setup['bias'],device='cuda')
        self.palette=(levels@mat.T+bias).clamp(-128,127)/128;self.pk=tuple(map(tuple,(self.palette.cpu().numpy()*128+128).tolist()))
        def remember(module,inputs,result):
            if isinstance(result,dict):self.nodes=result
        self.net.register_forward_hook(remember)

    def batch_style(self,native):
        x,t,specs,flow,valid=super().batch_style(native);codes=[]
        for spec in specs:
            row=self.by_id.get(spec['id']) or self.by_stem[spec['id']];a=region_codes(row['target'],row['region'],self.pk)
            xx,yy=spec['xy'];w,h=spec['wh'];a=a[yy:yy+h,xx:xx+w]
            if spec['flip']:a=a[:,::-1]
            codes.append(torch.from_numpy(np.ascontiguousarray(a.astype(np.int64))).cuda())
        self.class_labels=torch.stack(codes);self.motion_fraction=x.new_zeros(())
        if flow is not None:t,self.labels,self.class_labels,self.motion_fraction=align_pair(x,t,self.labels,self.class_labels,flow,valid)
        elif self.step%4 in (1,2):x=texture_intervention(x,self.labels,rng=self.rng)
        return x,t,specs,flow,valid

    def additional_loss(self,p,t,native):
        labels=self.labels if native else self.labels[...,40:-40,40:-40];region,values=joint_region_loss(p,t,labels)
        tail=region_tail_flatness(p,labels);edge=rgb_edge_vector_loss(p,t);white=bright_envelope_loss(p,t)
        if native:
            yy=self.rng.randrange(0,p.shape[-2]-384+1);xx=self.rng.randrange(0,p.shape[-1]-384+1)
            curve=contour_coherence(p[...,yy:yy+384,xx:xx+384],t[...,yy:yy+384,xx:xx+384])
        else:curve=contour_coherence(p,t)
        pre=pre720(self.net,self.nodes);code=self.class_labels
        if native:
            # Loss subsampling covers all four720 phases over training;
            # forward and export still compute every720 pixel.
            yy,xx=self.rng.randrange(2),self.rng.randrange(2);pre=pre[...,yy::2,xx::2];code=code[...,yy::2,xx::2]
        else:pre=pre[...,40:-40,40:-40];code=code[...,40:-40,40:-40]
        losses=[]
        for (lo,hi),truth,n in zip(GROUPS,(code//81,code//9%9,code%9),(15,9,9)):
            logp=count_logp(pre,lo,hi);freq=torch.bincount(truth.flatten(),minlength=n).float()/truth.numel()
            weight=freq.clamp_min(1/(4*n)).rsqrt();weight=(weight/(freq*weight).sum()).clamp(.25,4).detach()[truth]
            losses.append((-logp.gather(1,truth[:,None])[:,0]*weight).sum()/weight.sum().clamp_min(1))
        cls=torch.stack(losses).mean();values.update(packed720_count_ce=cls,region_tail_flatness=tail,joint_rgb_edge_vector=edge,bright_envelope=white,short_contour_turn_excess=curve,motion_aligned_target_fraction=self.motion_fraction)
        return .7*region+4*values['region_flatness']+values['adjacent_region_contrast']+2*edge+4*tail+4*curve+4*white+self.args.classification*cls,values

    def input_stability_loss(self,x,pred,native):
        loss,v=CleanRegionTraining.input_stability_loss(self,x,pred,native)
        if native and self.step%8==3:
            other=self.net(texture_intervention(x[:1],self.labels[:1],rng=self.rng))[0];stable=F.smooth_l1_loss(other,pred[:1],beta=4/128);loss=loss+2*stable;v['real_texture_output_invariance']=stable
        return loss,v

    def update(self,native=None):
        result=super().update(planned_native(self.step) if native is None else native)
        with torch.no_grad():
            for key,w in self.net.weight.items():
                if w.requires_grad:
                    lim=127*(getattr(self.net,key+'_m')/131072).float().flatten()[:,None,None,None];w.copy_(w.maximum(-lim).minimum(lim))
            for k,(w,b) in self.fixed.items():assert torch.equal(w,self.net.weight[k]) and torch.equal(b,self.net.bias[k])
        return result

    def preflight_extras(self):
        while self.step%4:self.update(True)
        result=self.update(False);assert result['temporal_pair'] and 'tiny_input_stability' in result['values'] and result['values']['packed720_count_ce']>0;return result

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(ternary_packed720=True,packed720=self.setup,ternary_region_decoder=False,encoder_antialias_keys=['down1__dw','down2__dw'],
            final_colour_decision_resolution='1280x720',probabilities_training_only=True,added_inference_ops=0,external_filter=False,visual_acceptance=False)
        torch.save(state,path)


@torch.inference_mode()
def evaluate(net,rows):
    result=base_evaluate(net,rows)
    for record,row in zip(result['records'],rows):
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];t=m.t.r.tensor(m.t.r.read(row['target']))[None];nodes=net(x,all_nodes=True);p=nodes['rgb_head']/128
        record['bright_envelope']=float(bright_envelope_loss(p,t));z=nodes['up2_high_dw'][0];codes=(81*z[:4].sum(0)+9*z[4:6].sum(0)+z[6:8].sum(0)).long().cpu().numpy()
        labels=components(codes);areas=np.bincount(labels.ravel());record.update(regions720=len(areas),small_region_pixels=float(areas[areas<256].sum()/areas.sum()))
    for k in ('bright_envelope','regions720','small_region_pixels'):result['mean'][k]=sum(r[k] for r in result['records'])/len(rows)
    return result


def regression(checkpoint):
    for maximum in (2,4):
        p=torch.linspace(-4,8,101,dtype=torch.float64);assert torch.allclose(log_values(p,maximum).exp().sum(1),torch.ones_like(p),atol=1e-12)
    z=torch.randn(1,8,1,1,dtype=torch.float64,requires_grad=True);dist=count_logp(z,2,4).exp();brute=torch.zeros_like(dist)
    for i in range(5):
        for j in range(3):brute[:,i+j]+=log_values(z[:,2],4).exp()[:,i]*log_values(z[:,3],2).exp()[:,j]
    assert torch.allclose(dist,brute,atol=1e-12);assert torch.autograd.gradcheck(lambda v:count_logp(v,2,4),(z,),atol=1e-5)
    t=torch.zeros(1,3,64,96);t[...,48:]=.5;p=t.clone();p[...,30:32]=.8
    assert bright_envelope_loss(t,t).abs()<1e-7 and bright_envelope_loss(p,t)>.1
    shifted=t.roll(2,-1);assert bright_envelope_loss(shifted[...,8:-8],t[...,8:-8]).abs()<1e-7
    from .probe_b1_encoder_gauge import load
    _,net=load(checkpoint);x=torch.randn(1,3,64,80).clamp(-1,127/128);raw=net.export()
    with torch.no_grad():before=net(x,all_nodes=True)
    net.__class__=Packed720B1;after=net(x,all_nodes=True);assert all(torch.equal(v,after[k]) for k,v in before.items()) and net.export()==raw
    after['rgb_head'].square().mean().backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters());assert net.weight['up1_low_pw'].grad.norm()>0
    print('PACKED720_REGRESSION_PASS probability/enumeration/gradcheck/envelope/forward/bytes/backward',flush=True)


if __name__=='__main__':
    import sys
    torch.set_num_threads(2)
    if '--check' in sys.argv:regression(sys.argv[sys.argv.index('--check')+1])
    else:driver.main(training_factory=Packed720Training,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
