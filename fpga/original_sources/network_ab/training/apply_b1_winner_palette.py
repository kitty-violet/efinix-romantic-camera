"""Place a train-target colour dictionary in B1's existing RGB coefficients."""
import argparse
import json
import shutil
from pathlib import Path
import torch
from . import train_b1_minimum_area as m
from .runtime import configure_workspace,write_json,file_hash
from .publish_b1_style_recovery import export


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--palette',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    s=torch.load(a.warm,map_location='cpu',weights_only=True);assert s.get('compact_winner')
    net=m.t.r.initialize();net.load_state_dict(s['model']);net.quantized=True;net.package['residual_encoding']=s['residual_encoding']
    spec=json.loads(Path(a.palette).read_text());w=torch.tensor(spec['weights'],device='cuda',dtype=torch.float32)
    net.weight['rgb_head'][:,:,0,0]=w[:8].T/16;net.bias['rgb_head'][:]=w[8]-128
    scale=net.weight['rgb_head'].abs().flatten(1).amax(1)/120
    net.rgb_head_m.copy_((scale*131072).ceil().clamp(1,131071).double()[None,:,None,None])
    d=m.load_data();export(net,d['validation'],dest/'package');m.t.preview(net,d['validation'],dest/'preview')
    torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=net.package['residual_encoding'],
        coverage_layout='8exclusive_winner_rays',soft_gate=True,winner_shape=True,bounded_scores=True,compact_winner=True,
        parent=a.warm,palette_source=a.palette),dest/'initial.pt')
    shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(parent=a.warm,parent_sha256=file_hash(a.warm),palette_source=a.palette,
        palette_sha256=file_hash(a.palette),added_operators=0,postfilter=False,visual_acceptance=False))
    print('WINNER_PALETTE_APPLIED',flush=True)


if __name__=='__main__':main()
