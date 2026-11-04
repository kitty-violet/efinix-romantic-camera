"""Test an in-graph, range-controlled integer colour representation.

Rescaling changes rounding and may remove pre-existing saturation; both effects
are measured. The exported graph/operators and coefficient count stay fixed.
"""
import copy
import argparse
from pathlib import Path
import torch
from . import train_b1_style_recovery as t
from .runtime import configure_workspace,file_hash,write_json
from .engine import set_reproducible


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',type=Path,default=t.OUT/'structured-calibration/target_112.pt')
    ap.add_argument('--out',type=Path,default=t.OUT/'activation-steps')
    ap.add_argument('--factors',type=float,nargs='+',default=[1,2,4,8,12]);args=ap.parse_args()
    configure_workspace();set_reproducible(2026100686)
    dest=args.out;dest.mkdir(exist_ok=False)
    parent=args.warm
    state=torch.load(parent,map_location='cpu',weights_only=True)
    rows=t.data()['validation'];base=t.r.initialize();base.load_state_dict(state['model'])
    if 'residual_encoding' in state:base.package['residual_encoding']=state['residual_encoding']
    else:
        from .b1_trained_image import decode
        raw=(parent.parent/'coefficients-padded.bin').read_bytes()
        _,base.package['residual_encoding']=decode(raw);assert base.export()==raw
    base.quantized=False;check=t.r.tensor(t.r.read(rows[0]['path']))[None]
    expected=base(check)[0];records=[]
    for factor in args.factors:
        assert factor>=1
        n=copy.deepcopy(base)
        n.weight['up2_high_dw'].div_(factor);n.bias['up2_high_dw'].div_(factor)
        n.weight['rgb_head'].mul_(factor)
        # Downscaling before ReLU commutes with it, but the original +/-127
        # saturation can make this non-equivalent at saturated pixels. Measure.
        float_error=float((n(check)[0]-expected).abs().max())
        # Retain the warm model's integer weights where the Q17 range allows.
        # Factor1 must reproduce the warm bytes exactly, without recalibration.
        if factor!=1:
            n.up2_high_dw_m.copy_((n.up2_high_dw_m/factor).round().clamp(1,131071))
            n.rgb_head_m.copy_((n.rgb_head_m*factor).round().clamp(1,131071))
        n.quantized=True;raw=n.export();p=dest/f'factor-{factor}.bin';p.write_bytes(raw)
        if factor==1:assert raw==base.export()
        actual=t.r.IntegerB1(t.r.runtime_from_image(raw))
        ours=n(check,all_nodes=True);other,_=actual(check*128,True)
        error=max(float((v-other[k]).abs().max()) for k,v in ours.items());assert error==0
        out=dest/f'factor-{factor}';out.mkdir()
        torch.save(dict(model=n.state_dict(),qat=True,step=0,contract_sha256=t.r.CONTRACT_SHA,
                        residual_encoding=n.package['residual_encoding']),out/'initial.pt')
        t.preview(n,rows,out/'preview');ev=t.evaluate(n,rows)
        records.append(dict(factor=factor,parent=str(parent),parent_sha256=file_hash(parent),
            float_rescale_max_error=float_error,actual_byte_all_node_error=error,
            validation=ev,package_sha256=file_hash(p),hardware_changed=False))
        write_json(dest/'results.json',records)
        print('ACTIVATION_STEPS',factor,float_error,ev['mean']['score'],flush=True)


if __name__=='__main__':main()
