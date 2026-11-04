"""Object-balanced supervision of target-selected 1--4px RGB strips.

Unlike a patch mean, the foreground of a thin strip contributes a full term
independently of how few pixels it occupies. Background, shape, and signed
RGB foreground contrast share ONE xy displacement over the full patch.
Training only; no inference filter or target-based source modification.
"""
import torch
from torch.nn import functional as F


def narrow_recall(p,t,labels,maximum_per_width=32):
    losses=[];ratios=[];counts=[]
    for axis in (-1,-2):
        pp=p if axis==-1 else p.transpose(-1,-2)
        tt=t if axis==-1 else t.transpose(-1,-2)
        ll=labels if axis==-1 else labels.transpose(-1,-2)
        py,ty=pp.permute(0,2,3,1),tt.permute(0,2,3,1)
        for width in (1,2,3,4):
            # Candidate x is the first foreground pixel, not its boundary.
            valid=torch.ones_like(ll,dtype=torch.bool)
            valid[:,:,:12]=False;valid[:,:,-(12+width):]=False
            valid[:,:5]=False;valid[:,-5:]=False
            valid &= ll!=torch.roll(ll,1,-1)
            for k in range(1,width):valid &= ll==torch.roll(ll,-k,-1)
            valid &= ll!=torch.roll(ll,-width,-1)
            left=torch.roll(ll,1,-1);right=torch.roll(ll,-width,-1)
            for k in (2,3,4):valid &= left==torch.roll(ll,k,-1)
            for k in (1,2,3):valid &= right==torch.roll(ll,-(width+k),-1)
            idx=valid.nonzero()
            if not len(idx):continue
            bb,yy,xx=idx.unbind(1)
            a=ty[bb,yy,xx-3];b=ty[bb,yy,xx+width+2]
            c=torch.stack([ty[bb,yy,xx+k] for k in range(width)],1).mean(1)
            keep=((a-b).abs().amax(1)<=10/128)&((c-(a+b)/2).norm(dim=1)>=12/128)
            idx=idx[keep]
            if len(idx)>maximum_per_width:
                idx=idx[torch.linspace(0,len(idx)-1,maximum_per_width,device=p.device).long()]
            if not len(idx):continue
            bb,yy,xx=idx.unbind(1)
            dy=torch.arange(-1,2,device=p.device)[None,:,None]
            dx=torch.arange(-8,width+8,device=p.device)[None,None,:]
            def gather(v,sy=0,sx=0):return v[bb[:,None,None],yy[:,None,None]+dy+sy,xx[:,None,None]+dx+sx]
            ref=gather(ty)
            def pieces(q):
                bg=torch.cat((q[:,:,:4],q[:,:,-4:]),2).mean((1,2))
                fg=q[:,:,8:8+width].mean((1,2))-bg
                return bg,fg,q-bg[:,None,None]
            rb,rv,rc=pieces(ref)
            norm=rv.norm(dim=1).clamp_min(12/128)
            # Keep all three rows to penalise short zigzags; the same target
            # gives the local thin foreground AND its neighbouring background.
            costs=[];contrast=[]
            offsets=sorted(((sy,sx) for sy in range(-2,3) for sx in range(-2,3)),key=lambda v:(v[0]**2+v[1]**2,v))
            for sy,sx in offsets:
                q=gather(py,sy,sx);qb,qv,qc=pieces(q)
                core=(qv-rv).norm(dim=1)/norm
                shape=(qc-rc).abs().mean((1,2,3))*3**.5/norm
                background=(qb-rb).norm(dim=1)/norm
                costs.append(2*core+shape+.25*background)
                contrast.append((qv*rv).sum(1)/rv.square().sum(1).clamp_min((12/128)**2))
            cost=torch.stack(costs);best=cost.detach().argmin(0)
            chosen=cost.gather(0,best[None])[0]
            ratio=torch.stack(contrast).gather(0,best[None])[0]
            losses.append(chosen.mean());ratios.append(ratio.detach());counts.append(len(idx))
    zero=p.sum()*0
    loss=torch.stack(losses).mean() if losses else zero
    values=dict(narrow_rgb_recall=loss,narrow_samples=p.new_tensor(sum(counts)),
        narrow_below_half=(torch.cat(ratios)<.5).float().mean() if ratios else zero.detach())
    return loss,values


def regression():
    scores={}
    for width in (1,2,3,4):
        lab=torch.zeros(1,64,96,dtype=torch.long);lab[...,48:48+width]=1
        bg=torch.tensor([-.3,-.1,.1])[None,:,None,None]
        delta=torch.tensor([.1,.2,.3])[None,:,None,None]
        target=bg.expand(1,3,64,96)+delta*lab[:,None]
        exact=narrow_recall(target,target,lab)[0];assert exact<1e-6
        blank=bg.expand_as(target).clone().requires_grad_()
        lost,values=narrow_recall(blank,target,lab)
        assert values['narrow_samples']>0 and lost>2
        lost.backward();assert torch.isfinite(blank.grad).all()
        assert (blank.grad*(target-blank.detach())).sum()<0
        shifted=target.roll((2,-2),(-2,-1))
        assert narrow_recall(shifted,target,lab)[0]<1e-6
        blurred=F.avg_pool2d(F.pad(target,(4,4,4,4),mode='replicate'),9,1)
        assert narrow_recall(blurred,target,lab)[0]>1
        bad=target.clone();bad[:,0]=bad[:,0].roll(2,-1)
        assert narrow_recall(bad,target,lab)[0]>.1
        assert narrow_recall(target,target,lab*0)[0]==0
        scores[width]=float(lost.detach())
    assert max(scores.values())/min(scores.values())<1.25
    print('NARROW_RGB_RECALL_PASS exact/shared_xy/missing_RGB/blur/chroma_shift/empty/width_balance',scores,flush=True)


if __name__=='__main__':torch.set_num_threads(2);regression()
