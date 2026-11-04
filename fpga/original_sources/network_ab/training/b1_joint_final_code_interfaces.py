"""One shared transition coordinate for a two-code RGB boundary.

This is a training constraint in the actual eight decoder fields. It is not
an output filter. The colour head can have opposite-signed contributions, so
independently correct endpoint bits can create a bright intermediate code.
"""
import cv2,numpy as np,torch
from torch.nn import functional as F


def interface_pairs(labels,palette):
    """labels N8HW binary; exactly two codes in5x5, colour contrast>=16."""
    bit=(labels.detach().cpu().numpy()>0).astype(np.uint16)
    codes=(bit*(2**np.arange(8,dtype=np.uint16))[None,:,None,None]).sum(1)
    lo=[];hi=[];valid=[];pal=palette.detach().cpu().numpy()
    for code in codes:
        a=cv2.erode(code.astype(np.uint16),np.ones((5,5),np.uint8)).astype(np.int64)
        b=cv2.dilate(code.astype(np.uint16),np.ones((5,5),np.uint8)).astype(np.int64)
        h,w=code.shape;pad=np.pad(code,2,mode='edge');other=np.zeros((h,w),bool)
        for dy in range(5):
            for dx in range(5):
                z=pad[dy:dy+h,dx:dx+w];other|=(z!=a)&(z!=b)
        lo.append(a);hi.append(b);valid.append((a!=b)&~other&(np.linalg.norm(pal[b]-pal[a],axis=-1)>=16))
    decode=lambda v:torch.from_numpy(((np.stack(v)[:,None]//(2**np.arange(8)[None,:,None,None]))%2).astype(np.float32)).to(labels.device)
    return decode(lo),decode(hi),torch.from_numpy(np.stack(valid)[:,None]).to(labels.device)


def loss(pre,aa,bb,valid,relevance,temperature=128.):
    changed=(aa!=bb).float();direction=bb-aa
    oriented=(pre-63.5)/temperature*direction
    weight=changed*relevance
    centre=(oriented*weight).sum(1,keepdim=True)/weight.sum(1,keepdim=True).clamp_min(1e-8)
    error=F.smooth_l1_loss(oriented,centre.expand_as(oriented),reduction='none')
    take=weight*valid
    return (error*take).sum()/take.sum().clamp_min(1)


def regression():
    aa=torch.zeros(1,8,3,3);bb=aa.clone();bb[:,:4]=1
    mask=torch.ones(1,1,3,3,dtype=torch.bool);rel=torch.ones(1,8,1,1)
    p=torch.full_like(aa,63.5);p[:,:4]=90
    assert loss(p,aa,bb,mask,rel)==0
    p[:,0]=190;p.requires_grad_();v=loss(p,aa,bb,mask,rel);g=torch.autograd.grad(v,p)[0]
    assert v>0 and g[:,0].mean()>0 and g[:,1:4].mean()<0 and g[:,4:].abs().sum()==0
    assert loss(p,aa,bb,torch.zeros_like(mask),rel)==0
    # Swapping endpoints must not change the objective or its gradients.
    q=p.detach().clone().requires_grad_();u=loss(q,bb,aa,mask,rel);gg=torch.autograd.grad(u,q)[0]
    assert torch.allclose(u,v) and torch.allclose(g,gg)
    return dict(synchronized_zero=True,desynchronized_gradient_correct=True,unused_bits_untouched=True,endpoint_swap_invariant=True,empty_mask_zero=True)
