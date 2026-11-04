"""Numerical capacity-probe checks; no claim of visual acceptance."""
import argparse,copy
from pathlib import Path
import torch
from .probe_b1_encoder_gauge import load
from .b1_spare_spatial_bank import install,compensate_head
from .train_b1_trainable_luma_readout import backend_palette
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(37);state,old=load(a.checkpoint);net=copy.deepcopy(old)
    original=backend_palette(old);install(net);bias=net.bias['rgb_head'].clone()
    assert torch.equal(backend_palette(net),original)
    rows=m.load_data(a.data)['validation'];checks=[]
    for row in rows:
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
        aa=old(x,all_nodes=True);bb=net(x,all_nodes=True)
        assert torch.equal(aa['rgb_head'],bb['rgb_head'])
        raw=net.export();rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');cc,_=rt(net.round(x*128),True)
        assert len(raw)==20480 and all(torch.equal(bb[k],v) for k,v in cc.items())
        checks.append(dict(id=row['id'],initial_rgb_error=0,integer_node_error=0))
        print('SPARE_BANK_INITIAL',row['id'],'RGB0/allnodes0',flush=True)
    # Test a nonzero, deliberately mixed-sign bounded deployed readout.
    net.weight['rgb_head'][:,3:,0,0].copy_(torch.tensor([[.125,-.0625,.0625,.0625,-.0625],[-.0625,.125,-.125,.03125,.03125],[.0625,.0625,-.125,-.0625,.0625]]))
    compensate_head(net,bias)
    assert torch.equal(backend_palette(net),original)
    aa=old(x,all_nodes=True);bb=net(x,all_nodes=True)
    assert torch.equal(aa['up2_high_dw'][:,:3],bb['up2_high_dw'][:,:3])
    delta=(bb['rgb_head']-aa['rgb_head']).abs().max();assert delta<=32
    rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu');cc,_=rt(net.round(x*128),True)
    assert all(torch.equal(bb[k],v) for k,v in cc.items())
    with torch.enable_grad():
        # New columns have a gradient on spatial data although their initial
        # contribution is zero. Constant data intentionally has none.
        fresh=copy.deepcopy(old);install(fresh);z=fresh(x[...,:96,:128],all_nodes=True)['rgb_head']
        loss=(z-old(x[...,:96,:128])[0]*128-1).square().mean();loss.backward()
        grad=fresh.weight['rgb_head'].grad[:,3:];assert torch.isfinite(grad).all() and grad.norm()>0
    write_json(a.out,dict(checks=checks,bytes=20480,backend_constant1215_error=0,nonzero_head_actual_nodes_error=0,
        maximum_example_correction=float(delta),analytic_correction_bound=32,new_head_gradient_norm=float(grad.norm()),
        checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),visual_acceptance=False))
    print('SPARE_BANK_AUDIT_PASS constants1215/initialRGB0/actualbytes0/bounded32/head_gradient',flush=True)


if __name__=='__main__':main()
