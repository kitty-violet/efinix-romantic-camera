"""One RGB/xy alignment per 2-D contour neighbourhood, not per scanline.

Independent 1-D profile offsets can regard a +/-2px zigzag as the same straight
edge on every row. Here a whole11x11 patch shares one two-dimensional offset;
values and both derivative directions share that offset and all RGB channels.
This is only a training loss. It introduces no runtime image operation.
"""
import torch
from torch.nn import functional as F


def coherent_profiles(p,t,labels,max_samples=256):
    seam=torch.zeros_like(labels,dtype=torch.bool)
    seam[:,:,1:]|=labels[:,:,1:]!=labels[:,:,:-1]
    seam[:,1:]|=labels[:,1:]!=labels[:,:-1]
    seam[...,:8]=False;seam[...,-8:]=False;seam[:,:8]=False;seam[:,-8:]=False
    idx=seam.nonzero()
    if len(idx)>max_samples:idx=idx[torch.linspace(0,len(idx)-1,max_samples,device=p.device).long()]
    if not len(idx):
        z=p.sum()*0
        return z,dict(coherent_rgb_patch=z,coherent_patch_samples=z.detach())
    bb,yy,xx=idx.unbind(1);d=torch.arange(-5,6,device=p.device)
    dy,dx=torch.meshgrid(d,d,indexing='ij')
    py=p.permute(0,2,3,1);ty=t.permute(0,2,3,1)
    def gather(v,sy=0,sx=0):return v[bb[:,None,None],yy[:,None,None]+dy+sy,xx[:,None,None]+dx+sx]
    ref=gather(ty);centred=ref-ref.mean((1,2),keepdim=True)
    contrast=(ref.amax((1,2))-ref.amin((1,2))).amax(1).clamp_min(8/128)
    gx,gy=torch.diff(ref,dim=2),torch.diff(ref,dim=1)
    choices=[]
    offsets=sorted(((sy,sx) for sy in range(-2,3) for sx in range(-2,3)),key=lambda v:(v[0]**2+v[1]**2,v))
    for sy,sx in offsets:
        q=gather(py,sy,sx);qc=q-q.mean((1,2),keepdim=True)
        shape=(qc-centred).abs().mean((1,2,3))
        derivative=((torch.diff(q,dim=2)-gx).abs().mean((1,2,3))+
                    (torch.diff(q,dim=1)-gy).abs().mean((1,2,3)))/2
        choices.append((shape+2*derivative)/contrast)
    options=torch.stack(choices)
    # amin splits gradients among equal minima. On a vanished thin branch,
    # every offset is tied, so the correction was spread over a5x5 footprint.
    # One detached correspondence keeps the RGB/xy geometry intact; prefer
    # zero displacement for ties rather than the top-left arbitrary offset.
    chosen=options.detach().argmin(0)
    value=options.gather(0,chosen[None])[0].mean()
    return value,dict(coherent_rgb_patch=value,coherent_patch_samples=p.new_tensor(len(idx)))


def regression():
    from .b1_relative_rgb_profiles import relative_profiles
    lab=torch.zeros(1,72,96,dtype=torch.long);lab[...,48:]=1
    target=lab[:,None].float().expand(-1,3,-1,-1)*.5
    # Every individual horizontal profile can perfectly realign this sawtooth;
    # the 2-D neighbourhood must still reject its changing line direction.
    jagged=target.clone()
    for y in range(72):jagged[:,:,y]=target[:,:,y].roll(2 if y%6<3 else -2,-1)
    old=relative_profiles(jagged,target,lab)[0]
    new=coherent_profiles(jagged,target,lab)[0]
    assert old<1e-6 and new>.02,(old,new)
    assert coherent_profiles(target,target,lab)[0]==0
    assert coherent_profiles(target+.1,target,lab)[0]<1e-6
    assert coherent_profiles(target.roll((2,-2),(-2,-1)),target,lab)[0]<1e-6
    blur=F.avg_pool2d(F.pad(target,(4,4,4,4),mode='replicate'),9,1).requires_grad_()
    loss=coherent_profiles(blur,target,lab)[0];loss.backward()
    assert torch.isfinite(blur.grad).all() and (blur.grad*(target-blur.detach())).sum()<0
    for width in (1,2,3,4):
        li=lab*0;li[...,48:48+width]=1;t=li[:,None].float().expand(-1,3,-1,-1)*.5
        assert coherent_profiles(t*0,t,li)[0]>.02
    corner=lab*0;corner[:,32:,48:]=1;t=corner[:,None].float().expand(-1,3,-1,-1)*.5
    assert coherent_profiles(t.roll((1,2),(-2,-1)),t,corner)[0]<1e-6
    assert coherent_profiles(t,t,corner*0)[0]==0
    print('COHERENT_RGB_PATCH_PASS old_zigzag',float(old),'new_zigzag',float(new),
          'exact/DC/common_xy_shift/blur_gradient/thin1to4/corner/empty',flush=True)


if __name__=='__main__':torch.set_num_threads(2);regression()
