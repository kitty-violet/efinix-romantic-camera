"""Fix loss of mask coverage at three-colour junctions, inside B1 weights.

Independent clipped sharpening does not preserve a partition of unity.
Use a single linear mask interpolator and verify integer mask sums instead.
No output filtering or change to the hardware contract is involved.
"""
import argparse
import copy
import shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from .probe_b1_decoder_coverage import fragment
from .publish_b1_style_recovery import export
from .runtime import configure_workspace,write_json


@torch.no_grad()
def repair(net):
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    for key in ('up1_high_dw','up2_low_dw'):
        net.weight[key].zero_();net.bias[key].zero_()
    net.weight['up1_high_dw'][:16,0,1,1]=1
    net.weight['up2_low_dw'][:16,0]=kernel
    for key in ('up1_high_dw','up2_low_dw'):
        w=net.weight[key];s=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,key+'_m').copy_((s*131072).clamp(1,131071).double()[None,:,None,None])


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);net=m.t.r.initialize();net.load_state_dict(state['model'])
    net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
    old=copy.deepcopy(net);repair(net)
    # Multiple junctions and thin regions reveal holes that two-region tests miss.
    yy,xx=torch.meshgrid(torch.arange(32,device='cuda'),torch.arange(32,device='cuda'),indexing='ij')
    patterns=torch.stack(((xx>=16).long(),(xx//5+yy//7)%16,(xx+2*yy)%16))
    masks=F.one_hot(patterns,16).permute(0,3,1,2).float()*127
    report={}
    for name,model in [('before',old),('after',net)]:
        result=fragment(model,masks,'up1_low_pw');coverage=result['up2_low_dw'].sum(1)
        report[name]=dict(min_sum=float(coverage.min()),max_sum=float(coverage.max()),
            missing_over10_fraction=float((coverage<117).float().mean()),max_partition_error=float((coverage-127).abs().max()))
    assert report['before']['missing_over10_fraction']>0
    assert report['after']['max_partition_error']<=4,report
    data=m.load_data();export(net,data['validation'],dest/'package');m.t.preview(net,data['validation'],dest/'preview')
    state.update(model=net.state_dict(),partition_preserving_decoder=True,parent=a.warm)
    torch.save(state,dest/'initial.pt');shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'partition-regression.json',dict(patterns=3,report=report,actual_byte_verified=True,postfilter=False,visual_acceptance=False))
    print('PARTITION_REPAIR_PASS',report,flush=True)


if __name__=='__main__':main()
