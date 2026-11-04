"""Exact binary opening/closing implemented by the unchanged B1 coefficients.

A 3x3 erosion followed by an anchored 2x2 erosion is a 4x4 erosion on the
180p grid. Reflected dilation reconstructs the surviving regions. Thus an
interior foreground component of each individual level set contains >=16
180p cells (256 native720 pixels). This does NOT guarantee the area of every
RGB intersection, nor border-clipped components. Those remain measured goals.
"""
import shutil
import argparse
import torch
import numpy as np
import cv2

from .seed_b1_binary_regions import build as base
from . import train_b1_style_recovery as t
from .minimum_area_blocks import OUT
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible


@torch.no_grad()
def build(geometry=False,outline=0.):
    net=base()
    thresholds=[(0,v) for v in [14,28,42,56,70,84,98,112]]+[(1,v) for v in [49,59,69,79]]+[(2,v) for v in [49,59,69,79]]
    w=net.weight['down2__pw'];b=net.bias['down2__pw'];w.zero_()
    for j,(c,threshold) in enumerate(thresholds):
        for k in (j,j+16):
            w[k,c,0,0]=127;w[k,c+3,0,0]=127;b[k]=-254*threshold-127
    net.down2__pw_m.fill_(131071)
    full=w.new_ones(3,3);upper=w.new_zeros(3,3);upper[:2,:2]=1
    lower=w.new_zeros(3,3);lower[1:,1:]=1
    operations=[(full,'erode',upper,'erode'),(lower,'dilate',full,'dilate'),(full,'dilate',full,'erode')]
    for stage,(a,op1,bkernel,op2) in enumerate(operations,1):
        for half,kernel,op in [(1,a,op1),(2,bkernel,op2)]:
            key=f'res{stage}_conv{half}__dw';dw=net.weight[key];bias=net.bias[key]
            dw.zero_();bias.zero_()
            dw[:16,0]=2*kernel
            bias[:16]=-(2*int(kernel.sum())-1)*127 if op=='erode' else -127
            for c in range(16,32):dw[c,0,1,1]=1
            getattr(net,key+'_m').fill_(65536)
        key=f'res{stage}_conv1__pw';net.weight[key].zero_();net.bias[key].zero_()
        for c in range(32):net.weight[key][c,c,0,0]=1
        getattr(net,key+'_m').fill_(65536)
        # conv2 PW is already processed-original; skip supplies original.
    if geometry:
        kernel=w.new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
        for c in range(16):
            net.weight['up2_low_dw'][c,0]=8*kernel
            net.bias['up2_low_dw'][c]=-7*63.5
    if outline:
        # Four existing decoder channels carry coarse luminance to four signed
        # Sobel DW kernels; RGB head subtracts their positive magnitudes. These
        # contours come from simplified regions, never the photographic source.
        sobel=w.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])/4
        for c,kernel in zip(range(3,7),(sobel,-sobel,sobel.T,-sobel.T)):
            net.weight['up2_low_pw'][c,:8,0,0]=14/127
            net.bias['up2_low_pw'][c]=7
            net.weight['up2_high_dw'][c,0]=kernel
            net.weight['rgb_head'][:,c,0,0]=-outline
    for key,w in net.weight.items():
        peak=w.abs().flatten(1).amax(1).clamp_min(1e-5)
        scale=torch.pow(2,torch.ceil(torch.log2(peak/127)))
        getattr(net,key+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])
    return net


def check_cells(a,minimum=16):
    """Interior connected components only, with explicit border exclusions."""
    sizes=[];border=[]
    for mask in a:
        n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=4)
        edge=set(np.unique(np.concatenate([labels[0],labels[-1],labels[:,0],labels[:,-1]])))
        for j in range(1,n):
            (border if j in edge else sizes).append(int(stats[j,4]))
    return dict(interior_components=len(sizes),interior_min_cells=min(sizes) if sizes else None,
                interior_violations=sum(s<minimum for s in sizes),border_components=len(border))


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='morphological-region-initialization')
    ap.add_argument('--geometry',action='store_true');ap.add_argument('--outline',type=float,default=0.)
    args=ap.parse_args()
    configure_workspace();set_reproducible(2026100651);cv2.setNumThreads(2)
    dest=OUT/args.name;dest.mkdir(exist_ok=False)
    net=build(args.geometry,args.outline);raw=net.export();(dest/'initial.bin').write_bytes(raw)
    actual=t.r.IntegerB1(t.r.runtime_from_image(raw));d=t.data();errors=[];checks=[]
    for row in d['validation']:
        x=t.r.tensor(t.r.read(row['path']))[None];a=net(x,all_nodes=True);b,_=actual(x*128,True)
        errors.append(max(float((v-b[k]).abs().max()) for k,v in a.items()))
        check={}
        for key in ('down2.pw','res1_add','res2_add','res3_add'):
            states=sorted(float(v) for v in a[key].unique());assert states==[0.,127.],(key,states)
            check[key]=check_cells(a[key][0,:16].cpu().numpy()>0)
        assert check['res2_add']['interior_violations']==0
        assert check['res3_add']['interior_violations']==0
        checks.append(dict(id=row['id'],stages=check))
    assert max(errors)==0
    torch.save(dict(model=net.state_dict(),qat=True,step=0,contract_sha256=t.r.CONTRACT_SHA,parameter_initialization_only=True),dest/'initial.pt')
    t.preview(net,d['validation'],dest/'preview')
    shutil.copy2(__file__,dest/'seed_source.py')
    write_json(dest/'review.json',dict(all_node_error=max(errors),checks=checks,training_steps=0,
        added_operators=0,minimum_area_per_plane720=256,minimum_rgb_component_guaranteed=False,
        border_area_guaranteed=False,geometry=args.geometry,outline=args.outline,code_sha256=file_hash(__file__)))
    print('EXACT_MORPHOLOGY_INIT',max(errors),checks,flush=True)


if __name__=='__main__':main()
