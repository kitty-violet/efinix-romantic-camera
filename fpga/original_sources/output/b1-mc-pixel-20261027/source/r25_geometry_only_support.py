"""Force the learned geometry branch to use derivatives, not absolute RGB/Y.

INT8 stem filters have exact zero DC per input channel. Existing support PW
rows can only read support features. This removes a colour/brightness shortcut
without adding any operators or channels to the deployed B1.
"""
from core import *
from r20_multifeature_structure import seed as base_seed
from r4_box_projector import BODY
import r18_learn_structure_support as fit
import argparse

original_samples=fit.samples

def data(count,seed,size=128):
    if seed!=271111:return original_samples(count,seed,size)
    a,y,k=original_samples(1536,271140,size)
    ids=json.loads((OUT/'r24-hard-lighting-negatives/mining.json').read_text('utf-8'))['selected_indices']
    return [a[i] for i in ids],y[ids],k[ids]

@torch.no_grad()
def seed():
    n=base_seed();n.support_input_masks={}
    for key,rows,cols in [('down1__pw',slice(3,8),slice(3,8)),('down2__pw',slice(23,32),slice(3,8)),
        *[(b+'__pw',slice(23,32),slice(23,32)) for b in BODY],('up1_low_pw',3,slice(23,32))]:
        mask=torch.zeros_like(n.weight[key]);mask[rows,cols,0,0]=1;n.support_input_masks[key]=mask
        keep=torch.ones_like(mask);keep[rows]=mask[rows];n.weight[key].mul_(keep)
    def project():
        with torch.no_grad():
            # Keep noncentral trainable values continuous. Choose only the
            # centre coefficient to cancel the rounded eight neighbours.
            for row in range(3,8):
                scale=float(n.stem_m[0,row,0,0]/131072)
                for ch in range(3):
                    w=n.weight['stem'][row,ch];q=HalfAway.apply(w/scale).clamp(-127,127)
                    other=float(q.sum()-q[1,1])
                    if abs(other)>127:
                        w.mul_(125/abs(other));q=HalfAway.apply(w/scale).clamp(-127,127);other=float(q.sum()-q[1,1])
                    assert abs(other)<=127
                    w[1,1]=-other*scale
                n.bias['stem'][row].clamp_(max=0)
            # A spatially constant input cannot manufacture an offset signal
            # later through a positive support-only bias.
            for key,rows in [('down1__dw',slice(3,8)),('down1__pw',slice(3,8)),('down2__dw',slice(3,8)),('down2__pw',slice(23,32)),
                *[(b+s,slice(23,32)) for b in BODY for s in ['__dw','__pw']],('up1_low_dw',slice(23,32)),('up1_low_pw',3)]:
                n.bias[key][rows].clamp_(max=0)
    n.project_support_geometry=project;project()
    return n

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=600);args=p.parse_args()
    fit.start=seed;fit.samples=data
    fit.main(argparse.Namespace(steps=args.steps,warmup=150,dest='r25-geometry-only-support',lr_scale=.5))
