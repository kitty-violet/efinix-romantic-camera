"""Allocate nine existing low channels to a learned structural support field.

Five chroma levels free eight existing channels. This changes coefficients only.
The original single support field collapsed in R18; this richer branch is tested
with continuous warmup and exact ordinary B1 QAT, never an external classifier.
"""
from core import *
from r15_structure_guard_probe import seed as guard_seed
from r4_box_projector import BODY
from try_user_pixel_curves import requantise
import r18_learn_structure_support as fit
import argparse

@torch.no_grad()
def seed():
    n=guard_seed();n.support_rows=slice(23,32);n.support_down_rows=slice(3,8);n.training_saturation_leak=.02
    cv=np.array([-96,-24,0,24,96],np.float32);ct=(cv[1:]+cv[:-1])/2
    n.weight['down2__pw'][15:].zero_();n.bias['down2__pw'][15:].zero_()
    for group,vec in enumerate([[.5,0,-.5],[-.25,.5,-.25]]):
        for j,t in enumerate(ct):
            row=15+4*group+j;n.weight['down2__pw'][row,:3,0,0]=torch.tensor(vec,device='cuda')*126
            n.bias['down2__pw'][row]=float(63.5-126*t/2)
    n.weight['up1_low_pw'][4:6].zero_();n.bias['up1_low_pw'][4:6].zero_()
    for group in range(2):
        row=4+group;n.weight['up1_low_pw'][row,15+4*group:19+4*group,0,0]=torch.tensor(np.diff(cv)/(2*127),device='cuda')
        n.bias['up1_low_pw'][row]=float((cv[0]+127)/2)
    dx=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],device='cuda',dtype=torch.float32)/8
    lap=torch.full((3,3),-1/8,device='cuda');lap[1,1]=1
    for j,kernel in enumerate([dx,-dx,dx.T,-dx.T,lap]):
        row=3+j;n.weight['stem'][row].zero_()
        for ch,v in enumerate([.25,.5,.25]):n.weight['stem'][row,ch]=kernel*v
        n.bias['stem'][row]=0;n.weight['down1__dw'][row].fill_(1/9);n.bias['down1__dw'][row]=0
    n.weight['down1__pw'][3:8].zero_();n.bias['down1__pw'][3:8].zero_()
    for row in range(3,8):
        n.weight['down1__pw'][row,row,0,0]=1;n.weight['down2__dw'][row].fill_(1/9);n.bias['down2__dw'][row]=0
    for row in range(23,32):
        n.weight['down2__pw'][row,3+(row-23)%5,0,0]=1;n.bias['down2__pw'][row]=-(row-23)//5*2
    for base in BODY:
        n.weight[base+'__dw'][23:].fill_(1/9);n.bias[base+'__dw'][23:].zero_()
        n.weight[base+'__pw'][23:].zero_();n.bias[base+'__pw'][23:].zero_()
        for row in range(23,32):n.weight[base+'__pw'][row,row,0,0]=1
        n.weight[base+'__pw'][23:,23:].add_(torch.randn_like(n.weight[base+'__pw'][23:,23:])*.015)
    n.weight['up1_low_dw'][23:].fill_(1/9);n.bias['up1_low_dw'][23:].zero_()
    n.weight['up1_low_pw'][3].zero_();n.weight['up1_low_pw'][3,23:32,0,0]=1;n.bias['up1_low_pw'][3]=-3
    requantise(n)
    return n

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=700);args=p.parse_args()
    fit.start=seed;fit.main(argparse.Namespace(steps=args.steps,warmup=200,dest='r20-multifeature-structure'))
