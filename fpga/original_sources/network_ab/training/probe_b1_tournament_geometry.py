"""Separate region cleanup from interpolation softness inside actual B1.

Binary tournament choices can be locally regularized while still selecting
exactly one leaf. This does not promise a global component area lower bound.
All variants use existing DW coefficients; no image postprocessing is used.
"""
import argparse
import copy
import shutil
from pathlib import Path
import torch
from . import train_b1_minimum_area as m
from .probe_b1_decoder_coverage import fragment
from .publish_b1_style_recovery import export
from .runtime import configure_workspace,write_json


@torch.no_grad()
def variant(net,name):
    keys=[]
    if name.startswith('choice'):
        key='res3_conv2__dw'
        choices=[0]+list(range(3,17))
        for j in choices:
            net.weight[key][j,0]=1;net.bias[key][j]=-4*127
        keys.append(key)
    if name.endswith('single'):
        key='up2_low_dw';net.weight[key].zero_();net.weight[key][:16,0,1,1]=1;net.bias[key].zero_();keys.append(key)
    elif name.endswith('half'):
        key='up2_low_dw';net.weight[key]*=.5;net.weight[key][:16,0,1,1]+=.5;keys.append(key)
    for key in keys:
        w=net.weight[key];scale=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,key+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])


@torch.no_grad()
def check_partition(net):
    gen=torch.Generator(device='cuda').manual_seed(61206)
    z=torch.zeros(16,32,32,32,device='cuda');z[:,:16]=torch.randint(0,128,(16,16,32,32),generator=gen,device='cuda')
    q=fragment(net,z,'res1_conv2.dw')['up1_low_pw']
    assert torch.all((q==0)|(q==127));assert torch.all(q.sum(1)==127)
    return dict(random_spatial_pixels=16*32*32,exclusive=True,coverage_error=0)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--variants',nargs='+',default=['half','single','choice-half','choice-single'],choices=['half','single','choice-half','choice-single']);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False);d=m.load_data()
    state=torch.load(a.warm,map_location='cpu',weights_only=True);assert state['exact_tournament']
    net=m.t.r.initialize();net.quantized=True;net.package['residual_encoding']=state['residual_encoding'];records=[]
    for name in a.variants:
        net.load_state_dict(state['model']);variant(net,name);check=check_partition(net);sub=dest/name;sub.mkdir()
        export(net,d['validation'],sub/'package');m.t.preview(net,d['validation'],sub/'preview')
        s=dict(state);s.update(model=copy.deepcopy(net.state_dict()),geometry_variant=name,parent=a.warm)
        torch.save(s,sub/'initial.pt');records.append(dict(variant=name,check=check))
        print('TOURNAMENT_GEOMETRY',name,check,flush=True)
    shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(records=records,warm=a.warm,manual_existing_decoder_adjustment=True,
        unchanged_trained_classifier=True,added_operators=0,postfilter=False,global_minimum_area_guaranteed=False,visual_acceptance=False))


if __name__=='__main__':main()
