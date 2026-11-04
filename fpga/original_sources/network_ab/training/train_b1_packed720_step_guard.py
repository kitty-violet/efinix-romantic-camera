"""Train away reproduced halos using constant-anchored synthetic boundaries.

The auxiliary batch has no texture, target-region extraction or image codec.
Its endpoint colours are fixed to the warm network's actual integer response.
Natural-image region/colour/contour and temporal objectives stay enabled.
"""
from pathlib import Path
import math
import numpy as np
import torch
from torch.nn import functional as F
from .train_b1_packed720_local_rims import LocalRimTraining
from .train_b1_ternary_packed720 import bright_envelope_loss,evaluate,preview,driver
from .train_b1_joint_rgb_edges import rgb_edge_vector_loss


def boundary_batch(colours, anchors, indices, angles, phases, widths, size=(128,192)):
    h,w=size;device=colours.device
    y,x=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
    distance=torch.stack([(x-w/2)*math.cos(a)+(y-h/2)*math.sin(a)-p for a,p in zip(angles,phases)])
    alpha=torch.stack([(d/s+.5).clamp(0,1) if s else (d>=0).float() for d,s in zip(distance,widths)])
    edge=(distance+.5).clamp(0,1)
    i=torch.tensor(indices,device=device);left,right=colours[i[:,0]],colours[i[:,1]]
    lo,hi=anchors[i[:,0]],anchors[i[:,1]]
    source=left[:,:,None,None]+(right-left)[:,:,None,None]*alpha[:,None]
    target=lo[:,:,None,None]+(hi-lo)[:,:,None,None]*edge[:,None]
    return source.round()/128-1,target/128-1,(distance.abs()>20)[:,None]


def step_guard_loss(p,t,core):
    white=bright_envelope_loss(p,t)
    # Immutable endpoint anchors reject reducing all brightness to evade the
    # one-sided halo loss. They are used only on synthetic constant interiors.
    anchor=((p-t).abs()*core).sum()/(3*core.sum()).clamp_min(1)
    boundary=rgb_edge_vector_loss(p,t)
    return 2*white+8*anchor+boundary,dict(synthetic_bright_rim=white,
        synthetic_endpoint_error=anchor,synthetic_edge_vector=boundary)


class StepGuardTraining(LocalRimTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        rng=np.random.default_rng(20261007193)
        colours=np.concatenate((np.repeat(np.arange(16,241,16)[:,None],3,axis=1),
                                rng.integers(16,241,(81,3))),axis=0)
        self.edge_colours=torch.tensor(colours,device='cuda',dtype=torch.float32)
        with torch.no_grad():
            x=self.edge_colours[:,:,None,None].expand(-1,-1,16,16)/128-1
            self.edge_anchors=self.net(x)[0][:,:,8,8]*128+128
        self.edge_anchors=self.edge_anchors.detach()

    def input_stability_loss(self,x,pred,native):
        loss,values=super().input_stability_loss(x,pred,native)
        if self.step%4==0:
            indices=[]
            for _ in range(3):
                for trial in range(100):
                    a=self.rng.randrange(len(self.edge_colours));b=self.rng.randrange(len(self.edge_colours))
                    if float((self.edge_anchors[a]-self.edge_anchors[b]).abs().max())>=24:break
                indices.append((a,b))
            angles=[self.rng.random()*2*math.pi for _ in indices]
            phases=[self.rng.uniform(-4,4) for _ in indices]
            widths=[self.rng.choice((0,0,2,4,12)) for _ in indices]
            source,target,core=boundary_batch(self.edge_colours,self.edge_anchors,indices,angles,phases,widths)
            output=self.net(source)[0]
            extra,terms=step_guard_loss(output[...,24:-24,24:-24],target[...,24:-24,24:-24],core[...,24:-24,24:-24])
            loss=loss+extra;values.update(terms)
        return loss,values

    def preflight_extras(self):
        result=super().preflight_extras()
        assert 'synthetic_bright_rim' in result['values']
        return result

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(synthetic_boundary_guard=dict(interval=4,batch=3,
            colours=self.edge_colours.cpu(),immutable_warm_anchors=self.edge_anchors.cpu()),
            synthetic_boundary_guard_inference_ops=0)
        torch.save(state,path)


def regression():
    c=torch.tensor([[48,48,48],[192,192,192]]).float()
    s,t,core=boundary_batch(c,c,[(0,1),(1,0)],[0,math.pi/3],[.25,-.25],[0,4])
    assert torch.equal(s*128+128,(s*128+128).round())
    zero,terms=step_guard_loss(t,t,core);assert zero.abs()<1e-7
    d=t-.1;bad,_=step_guard_loss(d,t,core);assert bad>.7
    p=t.clone();p[:,:,40:44,120:124]+=.25;p.requires_grad_()
    loss,terms=step_guard_loss(p,t,core);assert terms['synthetic_bright_rim']>0
    loss.backward();assert torch.isfinite(p.grad).all() and p.grad[:,:,40:44,120:124].sum()>0
    print('STEP_GUARD_REGRESSION_PASS',dict(exact=float(zero),global_dark_rejected=float(bad),
          local_bright=float(terms['synthetic_bright_rim'].detach()),gradient_finite=True),flush=True)


if __name__=='__main__':
    import sys
    torch.set_num_threads(2);regression()
    if '--check' not in sys.argv:
        driver.main(training_factory=StepGuardTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
