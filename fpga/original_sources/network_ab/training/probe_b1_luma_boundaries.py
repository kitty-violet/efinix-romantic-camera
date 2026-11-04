"""Delay luminance boundary decisions to the existing720p DW stage.

Six ordered luminance indicators and two continuous chroma fields fit the
existing eight channels. Region classification and all sampling sizes stay
unchanged. These are diagnostic coefficient changes, not further training.
"""
import argparse
import copy
import shutil
from pathlib import Path
import torch
from . import train_b1_minimum_area as m
from .probe_b1_sixteen_decoder import palette_actual
from .publish_b1_style_recovery import export
from .runtime import configure_workspace,write_json

LEVELS=[29,55,80,107,140,177,220]


@torch.no_grad()
def build(net,width):
    palette=(palette_actual(net)*128+128)
    r,g,b=palette.T;y=(r+2*g+b)/4;co=(r-b)/2;cg=(-r+2*g-b)/4
    keys=('up2_low_pw','up2_high_dw','rgb_head')
    for k in keys:net.weight[k].zero_();net.bias[k].zero_()
    kernel=palette.new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    for j,(a,b) in enumerate(zip(LEVELS[:-1],LEVELS[1:])):
        threshold=(a+b)/4
        net.weight['up2_low_pw'][j,:16,0,0]=y/(2*127)
        net.bias['up2_low_pw'][j]=64-threshold
        gain=127/width;net.weight['up2_high_dw'][j,0]=gain*kernel
        net.bias['up2_high_dw'][j]=-(64-width/2)*gain
        net.weight['rgb_head'][:,j,0,0]=(b-a)/127
    for c,coord in ((6,co),(7,cg)):
        net.weight['up2_low_pw'][c,:16,0,0]=coord/(2*127)
        net.bias['up2_low_pw'][c]=64
        net.weight['up2_high_dw'][c,0]=kernel
    net.weight['rgb_head'][:,6,0,0]=palette.new_tensor([2,0,-2])
    net.weight['rgb_head'][:,7,0,0]=palette.new_tensor([-2,2,-2])
    net.bias['rgb_head'][:]=palette.new_tensor([LEVELS[0]-128,LEVELS[0]-256,LEVELS[0]+128])
    for k in keys:
        w=net.weight[k];s=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,k+'_m').copy_((s*131072).clamp(1,131071).double()[None,:,None,None])
    return palette


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False);d=m.load_data()
    state=torch.load(a.warm,map_location='cpu',weights_only=True);assert state['exact_tournament']
    net=m.t.r.initialize();net.quantized=True;net.package['residual_encoding']=state['residual_encoding'];records=[]
    for width in (1,2,4):
        net.load_state_dict(state['model']);before=build(net,width);after=(palette_actual(net)*128+128)
        sub=dest/f'width{width}';sub.mkdir();export(net,d['validation'],sub/'package');m.t.preview(net,d['validation'],sub/'preview')
        s=dict(state);s.update(model=copy.deepcopy(net.state_dict()),luma720_boundary_width=width,parent=a.warm)
        torch.save(s,sub/'initial.pt');records.append(dict(width=width,flat_colours_before=before.cpu().tolist(),flat_colours_after=after.cpu().tolist()))
        print('LUMA720_PACKAGE',width,flush=True)
    shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(records=records,warm=a.warm,luminance_levels=LEVELS,decision_resolution='1280x720',
        classifier_resolution='320x180',chroma_continuous=True,manual_existing_decoder_adjustment=True,
        unchanged_trained_classifier=True,added_operators=0,postfilter=False,global_minimum_area_guaranteed=False,visual_acceptance=False))


if __name__=='__main__':main()
