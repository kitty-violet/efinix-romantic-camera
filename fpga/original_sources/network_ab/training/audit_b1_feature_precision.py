"""Separate activation rounding from coefficient learning, on CPU.

Float diagnostic uses exactly dequantized exported coefficients; it is not
claimed deployable or a trained FP32 checkpoint. B1 graph is unchanged.
"""
import argparse
import copy
from pathlib import Path
import torch
import numpy as np
from . import train_b1_minimum_area as m
from .runtime import write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();torch.set_num_threads(2);dest=Path(a.out);dest.mkdir(exist_ok=False)
    s=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(s['model']);net.package['residual_encoding']=s['residual_encoding'];net.quantized=True
    continuous=copy.deepcopy(net);continuous.quantized=False
    for key,w in continuous.weight.items():
        q=(getattr(continuous,key+'_m')/131072).float().flatten()
        w.copy_(net.round(w/q[:,None,None,None]).clamp(-127,127)*q[:,None,None,None])
        continuous.bias[key].copy_(net.round(continuous.bias[key]/q)*q)
    assert net.export()==continuous.export(),'The coefficient bytes must be identical'
    records=[]
    for row in m.load_data()['validation']:
        rgb=m.t.r.read(row['path']);x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)/128-1
        integer=net(x,all_nodes=True);smooth=continuous(x,all_nodes=True)
        by_node=[]
        for key,v in integer.items():
            if key=='input':continue
            if key not in smooth:continue
            std=v.std((-2,-1));lo=v.amin((-2,-1));hi=v.amax((-2,-1))
            by_node.append(dict(node=key,activation_range_median=float((hi-lo).median()),
                spatial_std_median=float(std.median()),channels_std_below1=int((std<1).sum()),
                total_channels=v.shape[1],mean_rounding_gap=float((v-smooth[key]).abs().mean())))
        b=integer['up1_high_dw'][:,:15]>=63.5;c=smooth['up1_high_dw'][:,:15]>=63.5
        record=dict(id=row['id'],bit_disagreement=float((b!=c).float().mean()),
            rgb_rounding_mae=float((integer['rgb_head']-smooth['rgb_head']).abs().mean()),nodes=by_node)
        records.append(record)
        for key,nodes in [('actual_integer',integer),('no_activation_rounding_diagnostic',smooth)]:
            m.t.r.c.rgb_image(nodes['rgb_head'][0]/128).save(dest/(row['id']+'_'+key+'.png'))
        print('FEATURE_PRECISION',row['id'],record['bit_disagreement'],record['rgb_rounding_mae'],flush=True)
    write_json(dest/'review.json',dict(checkpoint=a.checkpoint,sha256=file_hash(a.checkpoint),records=records,
        device='cpu',same_dequantized_coefficients=True,float_is_diagnostic_only=True,visual_accepted=False))


if __name__=='__main__':main()
