"""Compare complete target RGB profiles, retaining off-chord target colour.

The earlier endpoint projection could penalize p==target on real profiles:
spatial antialiasing near a third colour need not lie on one RGB chord.
Preserve that target residual while allowing shared colour offset and2px
displacement. This is a training-only replacement, with no inference change.
"""
import torch
from torch.nn import functional as F


def reference_boundary_profile(p,t,labels,max_samples=1024):
    shapes=[];contrasts=[];counts=[]
    for transpose in (False,True):
        pp=p.transpose(-2,-1) if transpose else p
        tt=t.transpose(-2,-1) if transpose else t
        ll=labels.transpose(-2,-1) if transpose else labels
        h,w=ll.shape[-2:];left,right=ll[...,4:-5],ll[...,5:-4];mask=left!=right
        for k in range(5):
            mask &= ll[...,k:w-9+k]==left
            mask &= ll[...,5+k:w-4+k]==right
        strength=(tt[...,3:-6]-tt[...,6:-3]).abs().amax(1);mask &= strength>12/128
        mask[:,:8]=False;mask[:,-8:]=False;idx=mask.nonzero()
        if len(idx)>max_samples:idx=idx[torch.linspace(0,len(idx)-1,max_samples,device=idx.device).long()]
        if not len(idx):continue
        bb,yy,xx=idx.unbind(1);xx=xx+4;offsets=torch.arange(-4,6,device=p.device)
        def gather(a):return a.permute(0,2,3,1)[bb[:,None],yy[:,None],xx[:,None]+offsets]
        ps,ts=gather(pp),gather(tt);pa,pb=ps[:,:2].mean(1),ps[:,-2:].mean(1)
        ta,tb=ts[:,:2].mean(1),ts[:,-2:].mean(1);vt,vp=tb-ta,pb-pa
        with torch.no_grad():
            alpha=(((ts-ta[:,None])*vt[:,None]).sum(-1)/vt.square().sum(-1).clamp_min(1e-5)[:,None]).clamp(0,1)
            residual=ts-ta[:,None]-alpha[:,:,None]*vt[:,None]
        options=[]
        for shift in (-2,-1,0,1,2):
            index=(torch.arange(10,device=p.device)+shift).clamp(0,9)
            desired=pa[:,None]+alpha[:,index,None]*vp[:,None]+residual[:,index]
            options.append((ps-desired).square().mean((1,2)))
        best=torch.stack(options).amin(0);top=best.topk(max(1,len(best)//4)).values
        shapes.append((top.mean()+1e-12).sqrt()-1e-6)
        contrasts.append(F.smooth_l1_loss(vp,vt,beta=4/128));counts.append(len(idx))
    if not shapes:return p.sum()*0,dict(rgb_profile_shape=p.sum()*0,rgb_profile_contrast=p.sum()*0,profile_samples=p.new_zeros(()))
    shape=torch.stack(shapes).mean();contrast=torch.stack(contrasts).mean()
    return shape+contrast,dict(rgb_profile_shape=shape,rgb_profile_contrast=contrast,profile_samples=p.new_tensor(sum(counts)))


def regression():
    from .train_b1_boundary_profile import boundary_profile
    labels=torch.zeros(1,64,96,dtype=torch.long);labels[...,48:]=1
    target=torch.zeros(1,3,64,96);target[...,:48]=-.5;target[...,48:]=.5
    # Legitimate off-chord colour in a target boundary. The old function
    # invents a nonzero error although prediction and target are identical.
    curved=target.clone();curved[:,1,:,47:49]+=.2;curved[:,2,:,47:49]-=.1
    old,_=boundary_profile(curved,curved,labels);new,_=reference_boundary_profile(curved,curved,labels)
    assert old>.01 and new.abs()<1e-7,(old,new)
    assert reference_boundary_profile(curved+.1,curved,labels)[0].abs()<1e-6
    assert reference_boundary_profile(curved.roll(2,-1),curved,labels)[0].abs()<1e-6
    blur=F.avg_pool2d(F.pad(target,(6,6,6,6),mode='replicate'),13,1)
    ring=target.clone();ring[...,46:48]=.1
    wrong=target.clone();wrong[:,0]=wrong[:,0].roll(2,-1);wrong[:,2]=wrong[:,2].roll(-2,-1)
    scores={k:float(reference_boundary_profile(z,target,labels)[0]) for k,z in [('blur',blur),('ring',ring),('channel_shift',wrong),('constant',target*0)]}
    assert min(scores.values())>.03,scores
    variable=ring.clone().requires_grad_();loss,_=reference_boundary_profile(variable,target,labels);loss.backward()
    assert torch.isfinite(variable.grad).all() and variable.grad.abs().sum()>0 and variable.grad.sum((2,3)).abs().max()<1e-5
    assert reference_boundary_profile(target,target,labels*0)[0]==0
    print('REFERENCE_PROFILE_PASS exact_curved0/offset0/shift0/blur/rim/channel_shift/constant/gradient/empty0',float(old),scores,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2);regression()
