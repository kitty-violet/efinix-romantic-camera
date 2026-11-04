"""Independent readback, common-pattern and BN-fold checks for the tiny probe."""
import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from .probe_fullres_c8 import FullresC8
from .runtime import configure_workspace, file_hash, write_json, contract_guard
from .engine import set_reproducible
from .train_b1_distillation import as_tensor


def main():
    root=configure_workspace();contract_guard();set_reproducible(202610034)
    out=root/'runs/sampling-hardware-audit-20261003/fullres-c8-probe'
    cfg=json.loads((out/'config.json').read_text(encoding='utf-8'))
    result=json.loads((out/'result.json').read_text())
    rows=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,1201))
    assert all(np.isfinite(r['loss']) and r['grad_norm']>0 for r in rows)
    assert file_hash(result['weights'])==result['weight_sha256']
    for p,h in cfg['code'].items():assert file_hash(p)==h
    model=FullresC8().cuda().eval()
    model.load_state_dict(torch.load(result['weights'],weights_only=True)['model'])
    fused=copy.deepcopy(model)
    for layer in fused.body:
        layer[1]=torch.nn.utils.fuse_conv_bn_eval(layer[1],layer[2]);layer[2]=nn.Identity()
    fold=[]
    with torch.inference_mode():
        for r in cfg['regression']:
            assert file_hash(r['source'])==r['source_sha256']
            assert file_hash(r['target'])==r['target_sha256']
            x=as_tensor(np.array(Image.open(r['source'])))[None].cuda()
            error=float((model(x)-fused(x)).abs().max());assert error<5e-5;fold.append(error)
        pattern=as_tensor(np.array(Image.open(root/'runs/b1-fit-controls-20261003/period4_input.png')))[None].cuda()
        init=FullresC8().cuda().initialize_identity()
        init_error=float((init(pattern)-pattern).abs().max())
        y=model(pattern)[:,:,80:-80,80:-80]*128+128
        spread=float((y.amax(dim=(-2,-1))-y.amin(dim=(-2,-1))).max())
    allocation=json.loads((out/'allocation.json').read_text())
    schedule=json.loads((out/'research_schedule.json').read_text())
    assert file_hash(out/'research_schedule.json')==allocation['schedule_sha256']
    # Check complete, non-overlapping final 720p coverage independently of allocator.
    cover=np.zeros((720,1280),np.uint8)
    for tile in schedule['tiles']:
        for q in tile['commands']:
            if q['kind']=='store':
                row=q['row'];a=row['origin'];b=a+row['width']
                assert 0<=row['y']<720 and 0<=a<b<=1280
                cover[row['y'],a:b]+=1
    assert np.all(cover==1)
    info=dict(status='DIAGNOSTIC_VERIFIED_NOT_DEPLOYMENT',steps=1200,
              exact_previous_period4_identity_max_error=init_error,
              exact_previous_period4_trained_output_range_rgb255=spread,
              bn_fold_full720_max_error=max(fold),bn_folding_is_not_int8_validation=True,
              complete_final_store_pixels=int(cover.sum()),stored_once=True,
              weight_sha256=file_hash(result['weights']),source_sha256=file_hash(Path(__file__)),
              shared_contract_sha256=file_hash(Path(__file__).parents[1]/'model_b_contract.json'))
    write_json(out/'verification.json',info);print(json.dumps(info),flush=True)


if __name__=='__main__':main()
