"""Penalize visible outliers inside flat target regions, without RGB blur.

An L1 regional mean can dilute small bright islands in a large uniform area.
Use a fourth-moment norm in the same eroded target interiors while retaining
count, colour, adjacent contrast and shared RGB-edge objectives. Only the loss
changes; real forward/export remain the fixed B1 graph.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_ternary_antialias import TernaryAntialiasTraining
from .train_b1_joint_region_colour import evaluate as base_evaluate,preview
from .train_b1_joint_rgb_edges import rgb_edge_vector_loss
from . import train_b1_late_region_supervision as driver
from . import train_b1_minimum_area as m


def region_tail_flatness(p,labels):
    boundary=torch.zeros_like(labels,dtype=torch.bool)
    boundary[:,:,1:]|=labels[:,:,1:]!=labels[:,:,:-1]
    boundary[:,1:]|=labels[:,1:]!=labels[:,:-1]
    core=(F.max_pool2d(boundary[:,None].float(),5,1,2)==0)[:,0]
    result=[]
    for pp,ll,ok in zip(p,labels,core):
        _,index=torch.unique(ll.flatten(),sorted=True,return_inverse=True)
        n=int(index.max())+1;ids=index[None].expand(3,-1);valid=ok.flatten().float()
        count=valid.new_zeros(n).scatter_add_(0,index,valid);den=count.clamp_min(1)
        mu=pp.new_zeros(3,n).scatter_add(1,ids,pp.flatten(1)*valid)/den
        fourth=(pp.flatten(1)-mu[:,index]).pow(4).mean(0)*valid
        moment=pp.new_zeros(n).scatter_add(0,index,fourth)/den
        # Subtract numerical floor so exactly constant regions give zero.
        norm=(moment+1e-12).pow(.25)-.001
        weight=count.sqrt();result.append((norm*weight).sum()/weight.sum().clamp_min(1))
    return torch.stack(result).mean()


class RegionTailTraining(TernaryAntialiasTraining):
    def additional_loss(self,p,target,native):
        loss,values=super().additional_loss(p,target,native)
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        tail=region_tail_flatness(p,labels);values['region_tail_flatness']=tail
        return loss+4*tail,values

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(region_tail_flatness=dict(moment=4,weight=4,core_exclusion=2),visual_acceptance=False)
        torch.save(s,path)


@torch.inference_mode()
def evaluate(net,rows):
    result=base_evaluate(net,rows)
    for record,row in zip(result['records'],rows):
        x=m.t.r.tensor(m.t.r.read(row['path']))[None]
        labels=torch.as_tensor(m.load_region(row['region']).astype('int64'),device=x.device)[None]
        p=net(x)[0];target=m.t.r.tensor(m.t.r.read(row['target']))[None]
        record['region_tail_flatness']=float(region_tail_flatness(p,labels))
        record['joint_rgb_edge_vector']=float(rgb_edge_vector_loss(p,target))
        record['selection_score']=(.7*record['score']+4*record['region_flatness']+
            record['adjacent_region_contrast']+2*record['joint_rgb_edge_vector']+4*record['region_tail_flatness'])
    result['mean']['region_tail_flatness']=sum(r['region_tail_flatness'] for r in result['records'])/len(rows)
    result['mean']['joint_rgb_edge_vector']=sum(r['joint_rgb_edge_vector'] for r in result['records'])/len(rows)
    result['mean']['base_joint_score']=result['mean']['score']
    result['mean']['score']=sum(r['selection_score'] for r in result['records'])/len(rows)
    return result


def regression():
    labels=torch.zeros(1,64,96,dtype=torch.long);labels[:,:,48:]=1
    target=torch.zeros(1,3,64,96);target[:,:,:,:48]=-.5;target[:,:,:,48:]=.5
    noisy=target.clone();noisy[:,:,20:24,20:24]+=.5
    exact=region_tail_flatness(target,labels);bad=region_tail_flatness(noisy,labels)
    assert exact.abs()<1e-7 and bad>.03
    shifted=target+torch.tensor([.1,-.1,.2])[None,:,None,None]
    assert region_tail_flatness(shifted,labels).abs()<1e-7
    p=noisy.clone().requires_grad_();loss=region_tail_flatness(p,labels);loss.backward()
    assert torch.isfinite(p.grad).all() and p.grad[:,:,20:24,20:24].mean()>0
    assert p.grad[:,:,30:34,20:24].mean()<0
    print('REGION_TAIL_REGRESSION_PASS',dict(exact=float(exact),island=float(bad)),flush=True)


if __name__=='__main__':
    import sys
    regression()
    if '--check' not in sys.argv:
        driver.main(training_factory=RegionTailTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
