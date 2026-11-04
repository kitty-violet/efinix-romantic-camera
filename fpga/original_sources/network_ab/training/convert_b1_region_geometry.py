"""Convert only specified fixed coefficients; preserve a learned encoder."""
import argparse
import shutil
from pathlib import Path
import torch
from . import train_b1_minimum_area as m
from .probe_b1_stopped_diffusion import full_gradient_range,opened_colour_regions
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible
from .publish_b1_style_recovery import export


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--opening',type=int,choices=(0,2,3),default=0)
    ap.add_argument('--coverage-gain',type=float,default=3.);a=ap.parse_args()
    assert 1<=a.coverage_gain<=8
    configure_workspace();set_reproducible(20261006141);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);net=m.t.r.initialize()
    net.load_state_dict(state['model']);net.quantized=True
    if 'residual_encoding' in state:net.package['residual_encoding']=state['residual_encoding']
    preserved={key:(net.weight[key].clone(),net.bias[key].clone(),getattr(net,key+'_m').clone())
               for key in net.weight if key=='stem' or key.startswith('down') or key=='rgb_head'}
    full_gradient_range(net)
    if a.opening:opened_colour_regions(net,a.opening)
    # Opening construction uses an untrained seed calibration. Restore every
    # learned input coefficient/scale and the fitted RGB palette exactly.
    for key,(w,b,q) in preserved.items():
        net.weight[key].copy_(w);net.bias[key].copy_(b);getattr(net,key+'_m').copy_(q)
    key='up2_low_dw'
    net.weight[key].zero_();net.weight[key][:,0,1,1]=a.coverage_gain
    net.bias[key].fill_(-(a.coverage_gain-1)*63.5)
    w=net.weight[key];s=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1)/127)))
    getattr(net,key+'_m').copy_((s*131072).clamp(1,131071).double()[None,:,None,None])
    for key,(w,b,q) in preserved.items():
        assert torch.equal(net.weight[key],w) and torch.equal(net.bias[key],b) and torch.equal(getattr(net,key+'_m'),q)
    data=m.load_data();export(net,data['validation'],dest/'package');m.t.preview(net,data['validation'],dest/'preview')
    torch.save(dict(model=net.state_dict(),qat=True,step=0,contract_sha256=m.t.r.CONTRACT_SHA,
        parent=a.warm,field_node='res2_add' if a.opening else 'res3_add',residual_encoding=net.package['residual_encoding']),dest/'initial.pt')
    paths=[Path(__file__),Path(__file__).with_name('probe_b1_stopped_diffusion.py')]
    for p in paths:shutil.copy2(p,dest/p.name)
    write_json(dest/'review.json',dict(args=vars(a),parent_sha256=file_hash(a.warm),
        code={str(p):file_hash(p) for p in paths},learned_encoder_and_palette_preserved=True,visual_acceptance=False,
        minimum_rgb_area_guaranteed=False,operators_added=0))
    print('REGION_GEOMETRY_CONVERSION',dest,flush=True)


if __name__=='__main__':main()
