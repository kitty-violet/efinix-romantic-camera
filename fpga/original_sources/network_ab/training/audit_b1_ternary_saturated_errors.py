"""Measure lost backward sensitivity at incorrectly predicted region states.

Uses canonical field states only as a diagnostic. Count supervision is
permutation-invariant, so field mismatch alone is not causal proof.
"""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from .train_b1_ternary_regions import TernaryB1,region_codes
from .probe_b1_ternary_field_targets import canonical_states,palette
from .runtime import write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();torch.set_num_threads(2);state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=copy.deepcopy(state['residual_encoding']);pk=palette(net)
    data=m.load_data(a.data);rows=[];seen=set()
    for row in data['training']:
        if row['video'] not in seen:rows.append(row);seen.add(row['video'])
    records=[]
    for row in rows:
        rgb=m.t.r.read(row['path'])[168:552,448:832];x=torch.from_numpy(rgb.transpose(2,0,1).copy())[None].float()/128-1
        ids=torch.from_numpy(region_codes(row['target'],row['region'],pk)[168:552,448:832].astype(np.int64))[None]
        target=F.avg_pool2d(canonical_states(ids),2).round()
        lab=torch.from_numpy(m.load_region(row['region'])[168:552,448:832].astype(np.int64))[None]
        boundary=torch.zeros_like(lab);boundary[:,:,1:]|=(lab[:,:,1:]!=lab[:,:,:-1]).long();boundary[:,1:]|=(lab[:,1:]!=lab[:,:-1]).long()
        core=(F.max_pool2d(boundary[:,None].float(),17,1,8)==0).float();core=F.avg_pool2d(core,2)==1
        nodes=net(x,all_nodes=True);key='up1_high_dw';s=(getattr(net,key+'_m')/131072).float().flatten()
        w=(net.weight[key]/s[:,None,None,None]).round().clamp(-127,127)*s[:,None,None,None]
        b=(net.bias[key]/s).round()*s
        pre=F.conv2d(F.pad(nodes['up1_nn'],(1,1,1,1),mode='replicate'),w,b,groups=16)[:,:15]
        derivative=torch.sigmoid(pre/8)-torch.sigmoid((pre-127)/8)
        wrong=(nodes['up2_low_dw'][:,:15]!=target)&core
        # A changed group count is unambiguously a wrong colour coordinate,
        # independent of permutations of fields within each group.
        active=torch.zeros_like(wrong)
        for lo,hi in ((0,7),(7,11),(11,15)):
            mismatch=(nodes['up2_low_dw'][:,lo:hi].sum(1,keepdim=True)!=target[:,lo:hi].sum(1,keepdim=True))&core
            active[:,lo:hi]=mismatch
        def frac(mask,test):return float((mask&test).sum()/mask.sum().clamp_min(1))
        records.append(dict(id=row['id'],wrong_core_fields=int(wrong.sum()),core_fields=int(core.sum())*15,
            wrong_field_small_derivative=frac(wrong,derivative<1e-4),
            wrong_count_group_small_derivative=frac(active,derivative<1e-4),
            core_small_derivative=frac(core.expand_as(wrong),derivative<1e-4),
            core_pw_clipped=frac(core.expand_as(wrong),(nodes['up1_nn'][:,:15]==0)|(nodes['up1_nn'][:,:15]==127)),
            wrong_pre_abs_mean=float(pre[wrong].abs().mean()) if wrong.any() else 0.))
    write_json(a.out,dict(records=records,checkpoint_sha256=file_hash(a.checkpoint),
        source_prediction=True,core_exclusion720=8,scope='TRAIN backward-sensitivity diagnostic, not a proof of sole cause'))
    print('TERNARY_SATURATION',records,flush=True)


if __name__=='__main__':main()
