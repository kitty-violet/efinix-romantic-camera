"""Enumerate actual constant-region decoder states, not nominal RGB levels."""
import argparse
import shutil
from pathlib import Path
import torch
from . import train_b1_minimum_area as m
from .probe_b1_decoder_coverage import fragment
from .runtime import configure_workspace,write_json


@torch.no_grad()
def palette(net):
    z=torch.zeros(8*128,16,1,1,device='cuda')
    for j in range(8):
        z[j*128:(j+1)*128,j]=127
        z[j*128:(j+1)*128,j+8,0,0]=torch.arange(128,device='cuda')
    out=fragment(net,z,start_node='up1_low_pw')
    features=out['up2_high_dw'][:,:,0,0];rgb=out['rgb_head'][:,:,0,0]/128
    palettes=[];classes=[];ys=[];feature_codes=[]
    for j in range(8):
        f=features[j*128:(j+1)*128,j]
        for code in f.unique(sorted=True):
            positions=(f==code).nonzero()[:,0];middle=positions[len(positions)//2]
            palettes.append(rgb[j*128+middle]);classes.append(j);ys.append(float(middle));feature_codes.append(int(code))
    return torch.stack(palettes),torch.tensor(classes,device='cuda'),torch.tensor(ys,device='cuda'),feature_codes


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);s=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.initialize();net.load_state_dict(s['model']);net.quantized=True;net.package['residual_encoding']=s['residual_encoding']
    sample=torch.arange(3*32*32,device='cuda').reshape(1,3,32,32).remainder(256).float()/128-1
    full=net(sample,all_nodes=True);part=fragment(net,full['up1_low_pw'],start_node='up1_low_pw')
    errors={k:float((v-full[k]).abs().max()) for k,v in part.items() if k not in ('up1_low_pw','res2_add')};assert max(errors.values())==0
    pal,classes,y,codes=palette(net)
    result=dict(colours=len(pal),feature_codes=sorted(set(codes)),palette=pal.cpu().tolist(),classes=classes.cpu().tolist(),
        y_values=y.cpu().tolist(),fragment_errors=errors,training_only=True,visual_acceptance=False)
    write_json(Path(a.out),result);print('WINNER_ACTUAL_PALETTE',len(pal),sorted(set(codes)),max(errors.values()),flush=True)


if __name__=='__main__':main()
