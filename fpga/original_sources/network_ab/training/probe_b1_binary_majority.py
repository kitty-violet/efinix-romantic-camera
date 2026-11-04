"""Encode local decision cleanup before palette colouring, within B1 DWs.

Each of four region bits is cleaned, then combined into exactly one code.
Unlike sharpening16 independent masks, this cannot leave colour-coverage holes.
This is a local vote; it does not guarantee global connected-component area.
"""
import argparse
import shutil
from pathlib import Path
import torch
from . import train_b1_minimum_area as m
from .probe_b1_decoder_coverage import fragment
from .publish_b1_style_recovery import export
from .runtime import configure_workspace,write_json


@torch.no_grad()
def install(net):
    scale=131071/131072
    k='up1_high_dw';net.weight[k].zero_();net.bias[k].zero_();net.up1_high_dw_m.fill_(131071)
    kernel=net.weight[k].new_tensor([[32,64,32],[64,127,64],[32,64,32]])
    net.weight[k][:4,0]=kernel*scale;net.bias[k][:4]=-32385*scale
    k='up2_low_dw';net.weight[k].zero_();net.bias[k].zero_();net.up2_low_dw_m.fill_(131071)
    net.weight[k][:4,0]=127*scale;net.bias[k][:4]=127*(1-572)*scale


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);assert state.get('binary_regions')
    net=m.t.r.initialize();net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True;install(net)
    # The final bit decision is exhaustive across all possible3x3 integer sums.
    sums=torch.arange(1144,device='cuda');parts=(sums[:,None]-127*torch.arange(9,device='cuda')[None]).clamp(0,127)
    z=torch.zeros(1144,16,3,3,device='cuda');z[:,:4]=parts[:,None,:].reshape(1144,1,3,3)
    result=fragment(net,z,'up1_high_dw')['up2_low_dw'][:,:4,1,1]
    expected=(sums[:,None]>=572).float().expand(-1,4)*127;assert torch.equal(result,expected)
    data=m.load_data();export(net,data['validation'],dest/'package');m.t.preview(net,data['validation'],dest/'preview')
    state.update(model=net.state_dict(),binary_local_majority=True,parent=a.warm)
    torch.save(state,dest/'initial.pt');shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(parent=a.warm,exhaustive_sum_cases=1144,integer_binary_error=0,
        added_operators=0,postfilter=False,global_minimum_area_guaranteed=False,visual_acceptance=False))
    print('BINARY_MAJORITY_COMPLETE',flush=True)


if __name__=='__main__':main()
