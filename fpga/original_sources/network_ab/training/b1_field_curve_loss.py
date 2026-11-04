"""Train-only short-turn regularization on ternary iso-contours.

Tangential curvature, unlike a Laplacian penalty, is zero on a straight ramp
regardless of its steepness. The bounded allowance preserves broad curves.
This is a custom regularizer inspired by level-set curvature, not an elastica
solver or an added FPGA operator.
"""
import torch
from torch.nn import functional as F


def field_curve_loss(field,allowance=.18):
    c=field.shape[1];k=field.new_tensor([[1,2,1],[2,4,2],[1,2,1]])[None,None]/16
    f=F.conv2d(F.pad(field,(1,1,1,1),mode='replicate'),k.expand(c,1,3,3),groups=c)
    centre=f[...,1:-1,1:-1];left=f[...,1:-1,:-2];right=f[...,1:-1,2:];up=f[...,:-2,1:-1];down=f[...,2:,1:-1]
    gx=(right-left)/2;gy=(down-up)/2;gxx=right+left-2*centre;gyy=up+down-2*centre
    gxy=(f[...,2:,2:]-f[...,2:,:-2]-f[...,:-2,2:]+f[...,:-2,:-2])/4
    g2=gx.square()+gy.square();curvature=(gxx*gy.square()-2*gx*gy*gxy+gyy*gx.square())/(g2+1e-3).pow(1.5)
    with torch.no_grad():
        proximity=sum(torch.sigmoid((centre-q)/.15)*(1-torch.sigmoid((centre-q)/.15))/.15 for q in (.5,1.5))
        weight=proximity*g2.sqrt()
        mask=torch.zeros_like(weight);mask[...,4:-4,4:-4]=1;weight=weight*mask
    excess=F.relu(curvature.abs()-allowance).clamp_max(2).square()
    return (excess*weight).sum()/weight.sum().clamp_min(1e-6)


def regression():
    yy,xx=torch.meshgrid(torch.arange(80.),torch.arange(100.),indexing='ij')
    straight=(.5+(xx-50)*.4)[None,None].clamp(0,2)
    shallow=(.5+(xx-50)*.08)[None,None].clamp(0,2)
    zig=(.5+(xx-50-3*torch.sin(yy*1.2))*.4)[None,None].clamp(0,2)
    a,b,c=map(field_curve_loss,(straight,shallow,zig))
    assert a<1e-6 and b<1e-6 and c>.01,(a,b,c)
    leaf=zig.clone().requires_grad_();field_curve_loss(leaf).backward();assert torch.isfinite(leaf.grad).all() and leaf.grad.norm()>0
    print('FIELD_CURVE_REGRESSION_PASS',dict(sharp_straight=float(a),shallow_straight=float(b),short_waves=float(c)),flush=True)


if __name__=='__main__':regression()
