"""Combine separately diagnosed fixes as ordinary B1 coefficients."""
import argparse
import shutil
from pathlib import Path
import torch
from . import train_b1_minimum_area as m
from .probe_b1_region_coverage import geometry,soften_threshold,binomial_sampling
from .publish_b1_style_recovery import export
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--palette',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();set_reproducible(20261006179);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);palette=torch.load(a.palette,map_location='cpu',weights_only=True)
    assert palette.get('target_palette_fit')
    net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=True;net.package['residual_encoding']=state['residual_encoding']
    binomial_sampling(net);soften_threshold(net,16)
    # Palette was fitted to4 coverage codes per mask pair. Geometry below
    # transports8 codes and compensates the affine head by exactly1/2.
    for key in ('weight.rgb_head','bias.rgb_head','rgb_head_m'):
        if key.startswith('weight.'):net.weight['rgb_head'].copy_(palette['model'][key])
        elif key.startswith('bias.'):net.bias['rgb_head'].copy_(palette['model'][key])
        else:net.rgb_head_m.copy_(palette['model'][key])
    assert round(float(net.weight['up2_high_dw'][0,0].sum()*127))==4
    geometry(net,True,3,8)
    data=m.load_data();export(net,data['validation'],dest/'package');m.t.preview(net,data['validation'],dest/'preview')
    torch.save(dict(model=net.state_dict(),qat=True,step=0,field_node='res2_add',fixed_sampling=True,
                    residual_encoding=state['residual_encoding'],parent=a.warm,palette_parent=a.palette),dest/'initial.pt')
    for p in (Path(__file__),Path(__file__).with_name('probe_b1_region_coverage.py')):shutil.copy2(p,dest/p.name)
    write_json(dest/'review.json',dict(warm=a.warm,warm_sha256=file_hash(a.warm),palette=a.palette,palette_sha256=file_hash(a.palette),
        operators_added=0,minimum_rgb_area_guarantee=False,visual_acceptance=False,retraining_required=True))


if __name__=='__main__':main()
