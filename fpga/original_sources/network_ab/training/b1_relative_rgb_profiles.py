"""Contrast-relative joint RGB shape supervision, including thin structures.

One common spatial offset is selected for all RGB values AND derivatives.
The normalization comes only from target contrast (minimum8 RGB bytes), so
reducing prediction contrast cannot reduce the denominator or hide an edge.
This is training-only, with no target access in deployed inference.
"""
import torch


def relative_profiles(p,t,labels,max_samples=2048):
    losses=[];counts=[]
    for axis in (-1,-2):
        pp=p if axis==-1 else p.transpose(-1,-2)
        tt=t if axis==-1 else t.transpose(-1,-2)
        ll=labels if axis==-1 else labels.transpose(-1,-2)
        mask=ll[...,1:]!=ll[...,:-1]
        mask[...,:8]=False;mask[...,-8:]=False;mask[:,:8]=False;mask[:,-8:]=False
        idx=mask.nonzero()
        if len(idx)>max_samples:idx=idx[torch.linspace(0,len(idx)-1,max_samples,device=p.device).long()]
        if not len(idx):continue
        bb,yy,xx=idx.unbind(1);offset=torch.arange(-4,6,device=p.device)
        def gather(v,shift):return v.permute(0,2,3,1)[bb[:,None],yy[:,None],xx[:,None]+offset+shift]
        ref=gather(tt,0);contrast=(ref.amax(1)-ref.amin(1)).amax(1).clamp_min(8/128)
        centred=ref-ref.mean(1,keepdim=True);difference=torch.diff(ref,dim=1)
        choices=[]
        for shift in (-2,-1,0,1,2):
            q=gather(pp,shift);qcentre=q-q.mean(1,keepdim=True)
            shape=(qcentre-centred).abs().mean((1,2))
            slope=(torch.diff(q,dim=1)-difference).abs().mean((1,2))
            choices.append((shape+2*slope)/contrast)
        losses.append(torch.stack(choices).amin(0).mean());counts.append(len(idx))
    value=torch.stack(losses).mean() if losses else p.sum()*0
    return value,dict(relative_rgb_shape=value,relative_shape_profiles=p.new_tensor(sum(counts)))


def regression():
    from torch.nn import functional as F
    ratios=[]
    for contrast in (12,24,72):
        lab=torch.zeros(1,64,96,dtype=torch.long);lab[...,48:]=1
        hard=torch.full((1,3,64,96),-.4);hard[...,48:]+=contrast/128
        t=F.avg_pool2d(F.pad(hard,(1,1,1,1),mode='replicate'),3,1)
        assert relative_profiles(t,t,lab)[0]==0
        assert relative_profiles(t+.2,t,lab)[0]<2e-6
        assert relative_profiles(t.roll(2,-1),t,lab)[0]<2e-6
        blur=F.avg_pool2d(F.pad(t,(4,4,4,4),mode='replicate'),9,1).requires_grad_()
        loss,_=relative_profiles(blur,t,lab);loss.backward()
        assert torch.isfinite(blur.grad).all() and (blur.grad*(t-blur.detach())).sum()<0
        ratios.append(float(loss.detach()))
    assert max(ratios)-min(ratios)<1e-5
    for width in (1,2,3,4):
        lab=torch.zeros(1,64,96,dtype=torch.long);lab[...,48:48+width]=1
        t=lab[:,None].float().expand(-1,3,-1,-1)*.4
        assert relative_profiles(t*0,t,lab)[0]>.04
    assert relative_profiles(t,t,lab*0)[0]==0
    print('RELATIVE_RGB_PROFILE_PASS weak_strong_equal/target_AA0/shift2/DC0/thin1to4/blur_gradient/empty',ratios,flush=True)


if __name__=='__main__':torch.set_num_threads(2);regression()
