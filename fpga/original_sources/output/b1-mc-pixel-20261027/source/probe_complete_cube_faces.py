"""Sweep one shared top/right boundary along the same depth vector.

The previous D-H top definition misclassified the upper part of the side face.
This test uses the existing DW/ReLU to extract top and right boundaries first,
then sweeps them together. No extra inference nodes or masks are introduced.
"""
from core import *
import probe_one_complete_cube as base
from r4_box_projector import BODY

@torch.no_grad()
def seed():
    n=base_seed()
    for i,key in enumerate(BODY):
        dw=n.weight[key+'__dw'];bias=n.bias[key+'__dw'];dw[:15].zero_();bias[:15].zero_()
        dw[:5,0,1,1]=1
        for ch in range(5,15):
            dw[ch,0,1,1]=1
            if i==0:
                if ch<10:dw[ch,0,0,1]=-1 # top boundary F(x,y)-F(x,y-1)
                else:dw[ch,0,1,2]=-1 # right boundary F(x,y)-F(x+1,y)
            else:dw[ch,0,2,0]=1 # same up/right depth sweep for both boundaries
    w=n.weight['up1_low_dw'];b=n.bias['up1_low_dw'];w[:15].zero_();b[:15].zero_();w[:5,0,1,1]=1
    w[5:15,0,1,1]=1;w[5:15,0,2,0]=1
    w=n.weight['up1_low_pw'];b=n.bias['up1_low_pw'];w[:3].zero_();b[:3].zero_()
    w[0,:5,0,0]=.2
    w[1,5:10,0,0]=.2;w[1,:5,0,0]=-.2
    w[2,10:15,0,0]=.2;w[2,:10,0,0]=-.2
    w=n.weight['up2_low_pw'];b=n.bias['up2_low_pw'];w.zero_();b.zero_()
    for ch in range(5):w[ch,ch,0,0]=1
    n.weight['rgb_head'][:,:3,0,0]=torch.tensor([-180.,-138.,-206.],device='cuda')/127
    n.capacity_face_channels=(1,2)
    for key,w in n.weight.items():
        need=(w.abs().flatten(1).amax(1)*131072/127).clamp_min(64)
        getattr(n,key+'_m').copy_((2**need.log2().ceil()).clamp(64,131071).double()[None,:,None,None])
    return n

base_seed=base.seed
if __name__=='__main__':
    base.seed=seed;base.main('r27-full-cube-planes',front_size=48,depth=24)
    report=json.loads((OUT/'r27-full-cube-planes/report.json').read_text('utf-8'))
    assert report['face_plane_uniformity_pass'] and report['positive_control_pass']
