"""Opening/closing of ordered colour regions, using only original B1 DWs.

An erosion/dilation pair removes narrow regions instead of averaging RGB.
Ordinal nesting survives monotone morphology. Intersections across colour
coordinates still require measured area checks; no universal area claim.
"""
import argparse
import copy
import shutil
from pathlib import Path
import torch
from . import train_b1_minimum_area as m
from .build_b1_median_cartoon import build
from .publish_b1_style_recovery import export
from .runtime import configure_workspace,write_json


@torch.no_grad()
def install(net,sequence):
    build(net,0,True)
    layers=['res1_conv1__dw','res1_conv2__dw','res2_conv1__dw','res2_conv2__dw','res3_conv1__dw','res3_conv2__dw']
    for key,operation in zip(layers,sequence):
        net.weight[key].zero_();net.bias[key].zero_()
        net.weight[key][:15,0]=1
        net.bias[key][:15]=dict(E=-8*127,D=0,M=-4*127)[operation]
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    net.weight['up1_high_dw'].zero_();net.weight['up1_high_dw'][:15,0]=kernel;net.bias['up1_high_dw'].zero_()
    net.weight['up2_low_dw'].zero_();net.weight['up2_low_dw'][:15,0,1,1]=127*(131071/131072)
    net.bias['up2_low_dw'].zero_();net.bias['up2_low_dw'][:15]=-63*127*(131071/131072)
    for key in layers+['up1_high_dw','up2_low_dw']:
        w=net.weight[key];s=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,key+'_m').copy_((s*131072).clamp(1,131071).double()[None,:,None,None])
    net.up2_low_dw_m[:,:15]=131071


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False);d=m.load_data();records=[]
    for sequence in ('EDD EMM'.replace(' ',''),'DEEDMM','EED DMM'.replace(' ','')):
        net=m.t.r.initialize();net.quantized=True;install(net,sequence);sub=dest/sequence;sub.mkdir()
        for row in d['validation']:
            z=net(m.t.r.tensor(m.t.r.read(row['path']))[None],all_nodes=True)['up2_low_dw'][:,:15]
            assert torch.all((z==0)|(z==127))
            for start,end in ((0,7),(7,11),(11,15)):assert torch.all(z[:,start:end-1]>=z[:,start+1:end])
        export(net,d['validation'],sub/'package');m.t.preview(net,d['validation'],sub/'preview')
        torch.save(dict(model=copy.deepcopy(net.state_dict()),qat=True,step=0,residual_encoding=net.package['residual_encoding'],
            ordered_regions=True,manual_morphological_regions=True,morphology_sequence=sequence,
            coverage_layout='15ordered360_rgb200',trained=False),sub/'initial.pt')
        records.append(dict(sequence=sequence,binary_and_monotone=True));print('MORPHOLOGY_PACKAGE',sequence,flush=True)
    shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(records=records,manually_designed=True,trained=False,added_operators=0,
        postfilter=False,global_minimum_area_guaranteed=False,visual_acceptance=False))


if __name__=='__main__':main()
