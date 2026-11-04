"""Control the hard-region gradient and constrain the RGB palette neutral axis.

The forward/export graph is identical. The training-only backward factor is
based on the measured 414:1 RGB/classification gradient imbalance. Parameter
projection prevents independent brightness bands acquiring arbitrary hues.
"""
from pathlib import Path
import torch
from . import train_b1_morphological_area as a


class RoutedB1(a.m.t.r.c.ColourB1):
    def forward(self,x,gate=None,mix=0.,all_nodes=False):
        nodes=super().forward(x,gate=gate,mix=mix,all_nodes=True)
        if nodes['down2.pw'].requires_grad:
            nodes['down2.pw'].register_hook(lambda g:g*.001)
        return nodes if all_nodes else (nodes['rgb_head']/128,nodes['res3_add'][:,24:25])


class RoutedMorphTraining(a.MorphTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        self.net.__class__=RoutedB1
        self.net.weight['rgb_head'].requires_grad_(False);self.net.bias['rgb_head'].requires_grad_(False)
        self.frozen['rgb_head']=(self.net.weight['rgb_head'].detach().clone(),self.net.bias['rgb_head'].detach().clone())
        self.project_palette()

    @torch.no_grad()
    def project_palette(self):
        w=self.net.weight['up2_low_pw'];b=self.net.bias['up2_low_pw']
        y=w[:3,:8,0,0].mean(0).clamp(6/127,22/127)
        co=((w[0,8:12,0,0]-w[2,8:12,0,0])/2).clamp(3/127,16/127)
        cg=((-w[0,12:16,0,0]+w[1,12:16,0,0]-w[2,12:16,0,0])/3).clamp(3/127,16/127)
        co0=co[:2].sum()*127;cg0=cg[:2].sum()*127
        origin=torch.stack((b[0]+co0-cg0+1,b[1]+cg0,b[2]-co0-cg0-1.5)).mean().clamp(0,14)
        w[:3].zero_();w[:3,:8,0,0]=y
        w[0,8:12,0,0]=co;w[2,8:12,0,0]=-co
        w[0,12:16,0,0]=-cg;w[1,12:16,0,0]=cg;w[2,12:16,0,0]=-cg
        b[0]=origin-co0+cg0-1;b[1]=origin-cg0;b[2]=origin+co0+cg0+1.5

    def update(self,native=None):
        result=super().update(native)
        self.project_palette();result['rgb_to_encoder_backward_factor']=.001
        return result

    def invariant(self):
        super().invariant()
        w=self.net.weight['up2_low_pw']
        assert torch.equal(w[0,:8],w[1,:8]) and torch.equal(w[1,:8],w[2,:8])
        assert torch.equal(w[0,8:12],-w[2,8:12]) and not torch.count_nonzero(w[1,8:12])
        assert torch.equal(-w[0,12:16],w[1,12:16]) and torch.equal(w[0,12:16],w[2,12:16])


if __name__=='__main__':
    a.main(training_factory=RoutedMorphTraining,extra_sources=[Path(__file__)])
