"""Teach small hard geometry under texture, using unchanged B1 coefficients.

Synthetic examples supplement real source/style pairs, not replace them.
The old step guard only used two half planes and chiefly a halo penalty.
Here thin strips, corners, apertures and curved patches have explicit RGB
and full-profile targets. Constant response anchors are fixed at warm start.
"""
import math
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from .train_b1_phase_geometry_native import PhaseGeometryNativeTraining,evaluate
from .b1_complete_rgb_profiles import complete_rgb_profiles
from .train_b1_contour_coherence import contour_coherence
from .train_b1_late_region_supervision import main as driver
from . import train_b1_minimum_area as m


def shape_batch(colours,anchors,rng,batch=4):
    h,w=192,256;device=colours.device
    yy,xx=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
    xs=[];ts=[];ls=[];specs=[]
    kernel=colours.new_tensor([.19,.62,.19]);kernel=(kernel[:,None]*kernel[None])[None,None]
    for j in range(batch):
        for _ in range(100):
            a,b=rng.randrange(len(colours)),rng.randrange(len(colours))
            if float((anchors[a]-anchors[b]).abs().max())>=24 and float((colours[a]-colours[b]).abs().max())>=32:break
        angle=rng.choice((0,math.pi/2,math.pi/6,math.pi/4,math.pi/3))
        u=(xx-w/2-rng.uniform(-24,24))*math.cos(angle)+(yy-h/2-rng.uniform(-16,16))*math.sin(angle)
        v=-(xx-w/2)*math.sin(angle)+(yy-h/2)*math.cos(angle)
        kind=rng.randrange(5);width=rng.choice((2,3,4,6,8,12))
        if kind==0:inside=u>=0
        elif kind==1:inside=(u.abs()<width/2)
        elif kind==2:inside=(u>=0)&(v>=0)
        elif kind==3:inside=(u.abs()<28)&(v.abs()<36)&~((u.abs()<28-width)&(v.abs()<36-width))
        else:inside=(u.square()/rng.uniform(25,55)**2+v.square()/rng.uniform(18,40)**2)<1
        label=inside.long();edge=F.conv2d(F.pad(inside.float()[None,None],(1,1,1,1),mode='replicate'),kernel)[0]
        source=colours[a,:,None,None]+(colours[b]-colours[a])[:,None,None]*edge
        target=anchors[a,:,None,None]+(anchors[b]-anchors[a])[:,None,None]*edge
        # Zero-mean fine texture and smooth illumination do not become new
        # target regions. Never draw synthetic labels into natural inputs.
        texture=torch.randn_like(source)*rng.choice((0.,2.,5.,8.))
        texture=texture+torch.sin(xx.float()/rng.uniform(1.5,4.))[None]*rng.choice((0.,2.,4.))
        source=(source+texture).round().clamp(0,255)
        xs.append(source/128-1);ts.append(target/128-1);ls.append(label)
        specs.append(dict(kind=kind,width=width,colour_indices=[a,b],angle=angle))
    return torch.stack(xs),torch.stack(ts),torch.stack(ls),specs


def shape_loss(p,t,labels):
    pixel=F.smooth_l1_loss(p,t,beta=2/128)
    profile,_=complete_rgb_profiles(p,t,labels)
    curve=contour_coherence(p,t)
    return 4*pixel+4*profile+2*curve,dict(synthetic_rgb=pixel,synthetic_profile=profile,synthetic_contour=curve)


class ShapeCurriculumTraining(PhaseGeometryNativeTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        rand=np.random.default_rng(2026100743)
        colours=np.concatenate((np.repeat(np.arange(24,233,16)[:,None],3,1),rand.integers(24,233,(82,3))))
        self.shape_colours=torch.tensor(colours,device='cuda',dtype=torch.float32)
        with torch.no_grad():
            x=self.shape_colours[:,:,None,None].expand(-1,-1,16,16)/128-1
            self.shape_anchors=(self.net(x)[0][:,:,8,8]*128+128).detach()

    def input_stability_loss(self,x,pred,native):
        loss,values=super().input_stability_loss(x,pred,native)
        if self.step%4==0:
            source,target,labels,specs=shape_batch(self.shape_colours,self.shape_anchors,self.rng)
            p=self.net(source)[0];cut=lambda v:v[...,40:-40,40:-40]
            extra,detail=shape_loss(cut(p),cut(target),cut(labels))
            loss=loss+.5*extra;values.update(detail)
        return loss,values

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(synthetic_shape_curriculum=dict(interval=4,weight=.5,batch=4,
            kinds=['half_plane','thin_strip','right_corner','frame_aperture','ellipse'],
            widths=[2,3,4,6,8,12],source_texture_byte_sigma=[0,2,5,8],
            colours=self.shape_colours.cpu(),immutable_warm_anchors=self.shape_anchors.cpu()),
            synthetic_data_inference_operations=0,visual_acceptance=False)
        torch.save(state,path)

    def restore(self,path):
        super().restore(path)
        state=torch.load(path,map_location='cpu',weights_only=True)['synthetic_shape_curriculum']
        self.shape_colours.copy_(state['colours']);self.shape_anchors.copy_(state['immutable_warm_anchors'])

    def preflight_extras(self):
        out=super().preflight_extras()
        assert 'synthetic_profile' in out['update']['values']
        out['synthetic_shapes_included']=True
        return out


def regression():
    import random
    torch.manual_seed(43)
    colours=torch.tensor([[32.,52.,72.],[188.,164.,148.]])
    x,t,labels,specs=shape_batch(colours,colours,random.Random(6),8)
    assert torch.equal(x*128+128,(x*128+128).round())
    assert x.shape==t.shape==(8,3,192,256) and len({s['kind'] for s in specs})>=3
    cut=lambda v:v[...,40:-40,40:-40]
    exact,_=shape_loss(cut(t),cut(t),cut(labels));assert exact.abs()<1e-7
    blur=F.avg_pool2d(F.pad(t,(5,5,5,5),mode='replicate'),11,1).requires_grad_()
    bad,_=shape_loss(cut(blur),cut(t),cut(labels));bad.backward()
    assert bad>.1 and torch.isfinite(blur.grad).all() and blur.grad.norm()>0
    print('SHAPE_CURRICULUM_PASS byte_input/types/exact0/blur_rejected/finite_gradient',float(bad.detach()),flush=True)


if __name__=='__main__':
    import sys
    torch.set_num_threads(2);regression()
    if '--check' not in sys.argv:driver(training_factory=ShapeCurriculumTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=m.t.preview)
