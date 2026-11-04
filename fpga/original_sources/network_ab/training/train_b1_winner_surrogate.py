"""Exact integer winner forward, differentiable backward for region choice.

Hard comparisons can cancel or clip gradients; the remaining rounding-STE
gradients need not describe a change of winner. This tests a probability
surrogate, not a proven root cause. The baseline gradient audit was nonzero.
Proxy operations exist only in backward; every forward tensor stays integer.
"""
import argparse
from pathlib import Path
import torch
import numpy as np
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from .train_b1_winner_regions import WinnerTraining
from . import train_b1_late_region_supervision as driver
from . import train_b1_region_formation as formation
from .runtime import configure_workspace,write_json


class ExactValueProxyGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx,actual,proxy):return actual
    @staticmethod
    def backward(ctx,gradient):return None,gradient


class WinnerGradientB1(m.t.r.c.ColourB1):
    def training_node_proxy(self,name,value,nodes):
        if name!='up1_low_pw' or not getattr(self,'proxy_enabled',True):return value
        scores=nodes['res2_conv1.dw'][:,:8]
        logits=(scores+torch.arange(8,device=scores.device)[None,:,None,None]/8)*2
        probability=logits.softmax(1)
        # Brightness passed through the same fixed spatial kernel as scores.
        luminance=nodes['res2_conv1.dw'][:,16:17]
        proxy=torch.cat((127*probability,probability*luminance),1)
        return ExactValueProxyGradient.apply(value,proxy)


class WinnerSurrogateTraining(WinnerTraining):
    def __init__(self,args,data):
        super().__init__(args,data);assert self.compact
        self.net.__class__=WinnerGradientB1

    @torch.no_grad()
    def classification_weights(self,labels,shape):
        small=F.interpolate(labels[:,None].float(),size=shape,mode='nearest')[:,0].long();rows=[]
        for lab in small:
            _,ids,count=lab.unique(return_inverse=True,return_counts=True)
            weight=count[ids].float().rsqrt();weight=(weight/weight.mean()).clamp(.25,4)
            rows.append(weight)
        return torch.stack(rows)[:,None]

    def additional_loss(self,p,target,native):
        loss,values=super().additional_loss(p,target,native)
        if native or self.step%8:return loss,values
        # Explicitly correct tiny predicted colour islands, in training only.
        # Projection is categorical area merging, never RGB Gaussian blur.
        from .minimum_area_blocks import components,area_merge,boundary_mode
        scores=self.nodes['res2_conv1.dw'][:1,:8,10:-10,10:-10]
        y=self.nodes['res2_conv1.dw'][:1,16:17,10:-10,10:-10]
        logits=(scores+torch.arange(8,device='cuda')[None,:,None,None]/8)*2
        with torch.no_grad():
            classes=logits.argmax(1)[0];wanted=(y[0,0,:,:,None]-self.palette_y).abs()
            wanted+=(self.palette_classes[None,None]!=classes[:,:,None])*1000
            ids=wanted.argmin(-1).cpu().numpy();pal=(self.palette.cpu().numpy()*128+128).clip(0,255).astype(np.uint8)
            rgb=pal[ids];lab=components(ids)
            merged,_=area_merge(lab,rgb,16);simple=boundary_mode(merged,1)
            cleaned,_=area_merge(components(simple),rgb,16)
            count=np.bincount(cleaned.ravel());means=np.stack([np.bincount(cleaned.ravel(),weights=rgb[...,c].ravel())/count for c in range(3)],1)
            closest=((means[:,None]-pal[None].astype(float))**2).sum(2).argmin(1)
            changed=torch.from_numpy(closest[cleaned]!=ids).to('cuda')[None,None]
            selected=torch.from_numpy(closest[cleaned]).to('cuda')
            trueclass=self.palette_classes[selected][None];truey=self.palette_y[selected][None,None]
        ce=F.cross_entropy(logits,trueclass,reduction='none')[:,None]
        cy=F.smooth_l1_loss(y/64,truey/64,reduction='none')
        project=((ce+cy)*changed).sum()/changed.sum().clamp_min(1)
        values.update(predicted_small_area_projection=project,projected_pixel_fraction=changed.float().mean())
        return loss+.3*project,values

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state['winner_backward_proxy']=True;torch.save(state,path)


def audit(checkpoint,out):
    configure_workspace();torch.set_num_threads(2);state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert state.get('compact_winner')
    net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=True;net.package['residual_encoding']=state['residual_encoding']
    net.__class__=WinnerGradientB1;row=m.load_data()['validation'][2]
    x=m.t.r.tensor(m.t.r.read(row['path']))[None,...,200:456,320:576]
    target=m.t.r.tensor(m.t.r.read(row['target']))[None,...,200:456,320:576]
    outputs=[];grads=[];vectors=[]
    for enabled in (False,True):
        net.proxy_enabled=enabled;n=net(x,all_nodes=True);p=n['rgb_head']/128
        loss=(p-target).square().mean();g=torch.autograd.grad(loss,n['res2_conv1.dw'])[0][:,:8]
        outputs.append({k:v.detach() for k,v in n.items()});grads.append(float(g.abs().sum()));vectors.append(g.detach().flatten())
    errors={k:float((v-outputs[0][k]).abs().max()) for k,v in outputs[1].items()}
    assert max(errors.values())==0 and grads[1]>0
    write_json(Path(out),dict(all_forward_node_errors=errors,score_gradient_l1_without=grads[0],score_gradient_l1_with=grads[1],
        gradient_cosine=float(torch.nn.functional.cosine_similarity(vectors[0][None],vectors[1][None])),
        baseline_gradient_was_nonzero=bool(grads[0]>0),gradient_proxy_only=True,inference_changed=False,visual_acceptance=False))
    print('WINNER_GRADIENT_AUDIT',grads,max(errors.values()),flush=True)


if __name__=='__main__':
    import sys
    if '--audit' in sys.argv:
        ap=argparse.ArgumentParser();ap.add_argument('--audit',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();audit(a.audit,a.out)
    else:
        here=Path(__file__).parent
        driver.main(training_factory=WinnerSurrogateTraining,extra_sources=[Path(__file__),here/'train_b1_winner_regions.py',
            here/'compact_b1_winner.py',here/'audit_b1_winner_palette.py',here/'probe_b1_decoder_coverage.py',
            here/'train_b1_quantum_regions.py',here/'train_b1_clean_regions.py',Path(formation.__file__)],evaluation=formation.evaluate)
