"""Unchanged-B1 self-dual region cleanup; explicit coefficients, not a postfilter.

Unlike erosion, majority treats foreground/background symmetrically. This probe
does not claim a strict connected-component area bound. All exported bytes are
reloaded and compared at every node before previewing.
"""
import argparse
import shutil
import torch
from .seed_b1_morphological_regions import build as initial
from . import train_b1_style_recovery as t
from .minimum_area_blocks import OUT
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible


@torch.no_grad()
def build(center=1.,passes=6,geometry=4.,outline=.5):
    net=initial(True,outline)
    for stage in range(1,4):
        for half in (1,2):
            key=f'res{stage}_conv{half}__dw'
            w,b=net.weight[key],net.bias[key]
            w[:16].zero_();b[:16].zero_()
            if (stage-1)*2+half<=passes:
                w[:16,0]=2
                w[:16,0,1,1]=2*center
                b[:16]=-(8+center)*127
            else:
                w[:16,0,1,1]=1
    for c in range(16):
        k=net.weight['up2_low_dw'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
        net.weight['up2_low_dw'][c,0]=geometry*k
        net.bias['up2_low_dw'][c]=-(geometry-1)*63.5
    for key,w in net.weight.items():
        peak=w.abs().flatten(1).amax(1).clamp_min(1e-5)
        scale=torch.pow(2,torch.ceil(torch.log2(peak/127)))
        getattr(net,key+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])
    return net


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True)
    ap.add_argument('--center',type=float,default=1.)
    ap.add_argument('--passes',type=int,default=6)
    ap.add_argument('--geometry',type=float,default=4.)
    args=ap.parse_args();configure_workspace();set_reproducible(2026100671)
    dest=OUT/args.name;dest.mkdir(exist_ok=False)
    net=build(args.center,args.passes,args.geometry);raw=net.export()
    (dest/'initial.bin').write_bytes(raw)
    actual=t.r.IntegerB1(t.r.runtime_from_image(raw));d=t.data();errors=[]
    for row in d['validation']:
        x=t.r.tensor(t.r.read(row['path']))[None]
        a=net(x,all_nodes=True);b,_=actual(x*128,True)
        errors.append(max(float((v-b[k]).abs().max()) for k,v in a.items()))
    assert max(errors)==0
    torch.save(dict(model=net.state_dict(),qat=True,step=0,contract_sha256=t.r.CONTRACT_SHA,
                    parameter_initialization_only=True),dest/'initial.pt')
    t.preview(net,d['validation'],dest/'preview')
    shutil.copy2(__file__,dest/'seed_source.py')
    write_json(dest/'review.json',dict(args=vars(args),all_node_error=max(errors),training_steps=0,
        minimum_area_guarantee=False,added_operators=0,code_sha256=file_hash(__file__)))
    print('MAJORITY_PROBE',dest,max(errors),flush=True)


if __name__=='__main__':main()
