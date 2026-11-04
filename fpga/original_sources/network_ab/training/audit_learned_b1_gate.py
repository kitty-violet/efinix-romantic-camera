"""Counterfactual checks on byte-reloaded B1 contour packages, not image edits."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .runtime import configure_workspace, contract_guard, write_json, file_hash
from .engine import set_reproducible
from .b1_trained_image import Parameters, encode, decode, runtime_from_image
from .probe_b1_cartoon_int8 import IntegerB1


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);args=ap.parse_args()
    configure_workspace();set_reproducible(42);contract_guard()
    raw=args.package.read_bytes();params,res=decode(raw)
    variants={'learned':raw}
    for name,value in [('zero_barrier',0),('full_barrier',96)]:
        p=copy.deepcopy(params);old=p['res1_conv2.pw'];w=[list(r) for r in old.weights];b=list(old.bias)
        for c in range(24,32):w[c]=[0]*32;b[c]=round(value*131072/old.multiplier[c])
        p['res1_conv2.pw']=Parameters(tuple(tuple(r) for r in w),tuple(b),old.multiplier,False)
        binary=encode(p,res);variants[name]=binary+bytes(20480-len(binary))
    models={name:IntegerB1(runtime_from_image(value)) for name,value in variants.items()}
    root=Path(r'E:\fpga\model-training-workspace\runs')
    cases=json.loads((root/'b1-deploy-preview-20261005-v1/data.json').read_text(encoding='utf-8'))['validation']
    rows=[]
    for r in cases:
        rgb=np.array(Image.open(r['source']).convert('RGB'));x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),device='cuda',dtype=torch.float32)-128
        outputs={name:net(x,True)[0] for name,net in models.items()}
        for name in ('zero_barrier','full_barrier'):
            assert torch.equal(outputs['learned']['down2.pw'],outputs[name]['down2.pw'])
        learned=outputs['learned'];gate=learned['res2_conv1.dw'][:,24:25]
        assert torch.equal(learned['res2_add'][:,24:25],gate)
        record=dict(id=r['id'],gate_used_equals_inspected=True)
        for name in ('zero_barrier','full_barrier'):
            delta=(learned['rgb_head']-outputs[name]['rgb_head']).abs()
            record[name]=dict(rgb_mae=float(delta.mean()),changed_pixel_fraction=float((delta.max(1).values>0).float().mean()),max_error=float(delta.max()))
        rows.append(record)
    assert max(r['zero_barrier']['changed_pixel_fraction'] for r in rows)>.01
    result=dict(package_sha256=file_hash(args.package),counterfactual='Only gate-producing coefficient rows changed; no RGB image editing',
        actual_gate_drives_surface=True,not_a_style_quality_pass=True,rows=rows)
    write_json(args.package.parent/'causal_gate_audit.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
