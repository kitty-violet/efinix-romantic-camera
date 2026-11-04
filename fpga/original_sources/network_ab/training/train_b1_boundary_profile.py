"""Supervise the whole RGB transition across a two-region boundary.

A local peak-gradient match can overlook additional bands beside that peak.
Brightness envelopes miss in-range pale rims. Fit one shared target coverage
profile to both predicted endpoint colours, with a separate contrast check.
All sampling/templates below are TRAINING ONLY.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_packed720_step_guard import StepGuardTraining,evaluate as base_evaluate,preview,driver
from . import train_b1_minimum_area as m


def boundary_profile(p,t,labels,max_samples=1024):
    shapes=[];contrasts=[];counts=[]
    for transpose in (False,True):
        pp=p.transpose(-2,-1) if transpose else p
        tt=t.transpose(-2,-1) if transpose else t
        ll=labels.transpose(-2,-1) if transpose else labels
        h,w=ll.shape[-2:]
        left,right=ll[...,4:-5],ll[...,5:-4]
        mask=left!=right
        for k in range(5):
            mask &= ll[...,k:w-9+k]==left
            mask &= ll[...,5+k:w-4+k]==right
        strength=(tt[...,3:-6]-tt[...,6:-3]).abs().amax(1)
        mask &= strength>12/128
        mask[:,:8]=False;mask[:,-8:]=False
        idx=mask.nonzero()
        if len(idx)>max_samples:
            idx=idx[torch.linspace(0,len(idx)-1,max_samples,device=idx.device).long()]
        if not len(idx):continue
        bb,yy,xx=idx.unbind(1);xx=xx+4
        offsets=torch.arange(-4,6,device=p.device)
        def gather(a):
            return a.permute(0,2,3,1)[bb[:,None],yy[:,None],xx[:,None]+offsets]
        ps,ts=gather(pp),gather(tt)
        pa,pb=ps[:,:2].mean(1),ps[:,-2:].mean(1)
        ta,tb=ts[:,:2].mean(1),ts[:,-2:].mean(1);vt=tb-ta;vp=pb-pa
        alpha=(((ts-ta[:,None])*vt[:,None]).sum(-1)/vt.square().sum(-1).clamp_min(1e-5)[:,None]).clamp(0,1).detach()
        options=[]
        for shift in (-2,-1,0,1,2):
            moved=alpha[:,(torch.arange(10,device=p.device)+shift).clamp(0,9)]
            desired=pa[:,None]+moved[:,:,None]*vp[:,None]
            options.append((ps-desired).square().mean((1,2)))
        # One shift for the WHOLE RGB profile, not per channel or per pixel.
        best=torch.stack(options).amin(0)
        top=best.topk(max(1,len(best)//4)).values
        shapes.append((top.mean()+1e-12).sqrt()-1e-6)
        contrasts.append(F.smooth_l1_loss(vp,vt,beta=4/128))
        counts.append(len(idx))
    if not shapes:return p.sum()*0,dict(rgb_profile_shape=p.sum()*0,rgb_profile_contrast=p.sum()*0,profile_samples=p.new_zeros(()))
    shape=torch.stack(shapes).mean();contrast=torch.stack(contrasts).mean()
    return shape+contrast,dict(rgb_profile_shape=shape,rgb_profile_contrast=contrast,profile_samples=p.new_tensor(sum(counts)))


class BoundaryProfileTraining(StepGuardTraining):
    def additional_loss(self,p,t,native):
        loss,values=super().additional_loss(p,t,native)
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        profile,terms=boundary_profile(p,t,labels)
        # Replace the single-gradient matching contribution, retain its metric.
        loss=loss-2*values['joint_rgb_edge_vector']+4*profile
        values.update(terms);return loss,values

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(full_boundary_profile_training=True,profile_span720=10,profile_shift720=2,
                 gradient_peak_term_replaced=True,visual_acceptance=False)
        torch.save(s,path)


@torch.inference_mode()
def evaluate(net,rows):
    result=base_evaluate(net,rows)
    for record,row in zip(result['records'],rows):
        x=m.t.r.tensor(m.t.r.read(row['path']))[None]
        t=m.t.r.tensor(m.t.r.read(row['target']))[None]
        labels=torch.as_tensor(m.load_region(row['region']).astype('int64'),device=x.device)[None]
        _,terms=boundary_profile(net(x)[0],t,labels)
        record.update({k:float(v) for k,v in terms.items()})
    for k in ('rgb_profile_shape','rgb_profile_contrast','profile_samples'):
        result['mean'][k]=sum(r[k] for r in result['records'])/len(rows)
    mean=result['mean'];mean['previous_selection_score']=mean['score']
    mean['score']+=4*(mean['rgb_profile_shape']+mean['rgb_profile_contrast'])-2*mean['joint_rgb_edge_vector']
    return result


def regression():
    labels=torch.zeros(1,64,96,dtype=torch.long);labels[...,48:]=1
    t=torch.zeros(1,3,64,96);t[...,:48]=-.5;t[...,48:]=.5
    same,v=boundary_profile(t,t,labels);assert same.abs()<1e-7
    shifted,_=boundary_profile(t.roll(2,-1),t,labels);assert shifted.abs()<1e-7
    offset,v=boundary_profile(t+.1,t,labels);assert offset.abs()<1e-6
    blur=F.avg_pool2d(F.pad(t,(6,6,6,6),mode='replicate'),13,1)
    broad,_=boundary_profile(blur,t,labels);assert broad>.1
    ring=t.clone();ring[...,46:48]=.1
    band,_=boundary_profile(ring,t,labels);assert band>.03
    wrong=t.clone();wrong[:,0]=wrong[:,0].roll(2,-1);wrong[:,2]=wrong[:,2].roll(-2,-1)
    colour,_=boundary_profile(wrong,t,labels);assert colour>.05
    flat,v=boundary_profile(t*0,t,labels);assert v['rgb_profile_contrast']>.5
    variable=ring.clone().requires_grad_();loss,_=boundary_profile(variable,t,labels);loss.backward()
    assert torch.isfinite(variable.grad).all() and variable.grad.abs().sum()>0
    assert variable.grad.sum((2,3)).abs().max()<1e-5
    none,v=boundary_profile(t,t,labels*0);assert none==0 and v['profile_samples']==0
    print('BOUNDARY_PROFILE_REGRESSION_PASS',dict(exact=float(same),shift=float(shifted),
        global_offset=float(offset),blur=float(broad),in_range_band=float(band),
        RGB_misalignment=float(colour),constant_rejected=float(flat)),flush=True)


if __name__=='__main__':
    import sys
    torch.set_num_threads(2);regression()
    if '--check' not in sys.argv:
        driver.main(training_factory=BoundaryProfileTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
