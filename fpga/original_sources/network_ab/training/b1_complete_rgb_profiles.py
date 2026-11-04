"""Keep full RGB profiles, including thin strips and multi-region junctions.

The old profile mask required five pixels of each of two region IDs. It
therefore provided no profile supervision for a three-pixel-wide branch.
Use the full target profile with one shared spatial offset and colour offset;
do not project its amplitude onto predicted endpoints.
"""
import torch


def complete_rgb_profiles(p,t,labels,max_samples=2048):
    results=[];counts=[]
    for axis in (-1,-2):
        pp=p if axis==-1 else p.transpose(-1,-2)
        tt=t if axis==-1 else t.transpose(-1,-2)
        ll=labels if axis==-1 else labels.transpose(-1,-2)
        mask=(ll[...,1:]!=ll[...,:-1]) & (torch.diff(tt,dim=-1).abs().amax(1)>4/128)
        mask[...,:8]=False;mask[...,-8:]=False;mask[:,:8]=False;mask[:,-8:]=False
        idx=mask.nonzero()
        if len(idx)>max_samples:idx=idx[torch.linspace(0,len(idx)-1,max_samples,device=p.device).long()]
        if not len(idx):continue
        bb,yy,xx=idx.unbind(1);offsets=torch.arange(-4,6,device=p.device)
        def gather(v,shift):return v.permute(0,2,3,1)[bb[:,None],yy[:,None],xx[:,None]+offsets+shift]
        ref=gather(tt,0);ref=ref-ref.mean(1,keepdim=True)
        options=[]
        for shift in (-2,-1,0,1,2):
            candidate=gather(pp,shift);candidate=candidate-candidate.mean(1,keepdim=True)
            options.append((candidate-ref).abs().mean((1,2)))
        results.append(torch.stack(options).amin(0).mean());counts.append(len(idx))
    loss=torch.stack(results).mean() if results else p.sum()*0
    return loss,dict(complete_rgb_profile=loss,complete_profile_samples=p.new_tensor(sum(counts)))


def regression():
    from .train_b1_reference_boundary_profile import reference_boundary_profile
    scores={}
    for width in (1,2,3,4,8,20):
        labels=torch.zeros(1,64,96,dtype=torch.long);labels[...,48:48+width]=1
        t=torch.full((1,3,64,96),.5);t[...,48:48+width]=-.5;p=torch.full_like(t,.5)
        old=reference_boundary_profile(p,t,labels)[0];new=complete_rgb_profiles(p,t,labels)[0]
        if width<5:assert old==0
        assert new>.01 and complete_rgb_profiles(t,t,labels)[0]==0
        assert complete_rgb_profiles(t+.1,t,labels)[0]<1e-6
        assert complete_rgb_profiles(t.roll(2,-1),t,labels)[0]<1e-6
        p.requires_grad_();complete_rgb_profiles(p,t,labels)[0].backward()
        assert torch.isfinite(p.grad).all() and p.grad[...,48:48+width].mean()>0
        scores[width]=dict(old=float(old),new=float(new))
    assert complete_rgb_profiles(t,t,labels*0)[0]==0
    print('COMPLETE_PROFILE_PASS exact0/offset0/shift2/width1to20/missing_direction/finite/empty0',scores,flush=True)


if __name__=='__main__':torch.set_num_threads(2);regression()
