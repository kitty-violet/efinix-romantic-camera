"""Current14Y diagnosis: remove encoder activation rounding, preserve bytes.

Counterfactual float activations are not executable on the fixed B1 hardware.
No old packed-palette statistic is applied to continuous chroma/current14Y.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from PIL import Image
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .probe_b1_monotone_pair_decoder import make_board
from .audit_b1_rim_tolerance import rim_terms
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);_,net=load(a.checkpoint);floating=copy.deepcopy(net);floating.quantized=False
    for k,w in floating.weight.items():
        s=(getattr(floating,k+'_m')/131072).float().flatten()
        w.copy_(net.round(w/s[:,None,None,None]).clamp(-127,127)*s[:,None,None,None]);floating.bias[k].copy_(net.round(floating.bias[k]/s)*s)
    assert floating.export()==net.export()
    out=Path(a.out);out.mkdir(exist_ok=False);boards=[{},{},{}];records=[]
    for row in m.load_data(a.data)['validation']:
        def tensor(path):return torch.from_numpy(m.t.r.read(path).transpose(2,0,1).copy())[None].float()/128-1
        x=tensor(row['path']);t=tensor(row['target']);labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
        actual=net(x,all_nodes=True);continuous=floating(x,all_nodes=True)
        hybrid=fragment(net,net.round(continuous['up1_low_pw']),'up1_low_pw')
        boards[0][row['id']]=Image.fromarray(m.t.r.read(row['target']))
        for j,(name,nodes) in enumerate((('actual',actual),('float_encoder_only',hybrid)),1):
            p=nodes['rgb_head']/128;rim,_=rim_terms(p,t,labels,2)
            records.append(dict(id=row['id'],mode=name,mae=float((p-t).abs().mean()*128),tight_rim=float(rim)))
            im=m.t.r.c.rgb_image(p[0]);im.save(out/f'{name}_{row["id"]}.png');boards[j][row['id']]=im
        print('CURRENT_ACTIVATION_ROUNDING',records[-2:],flush=True)
    make_board(boards,['目标','实际整数网络','仅取消编码器激活舍入：诊断'],out/'comparison.png')
    write_json(out/'review.json',dict(records=records,same_exported_coefficients=True,integer_decoder_unchanged=True,hybrid_not_deployable=True,visual_acceptance=False,source_sha256=file_hash(__file__),checkpoint_sha256=file_hash(a.checkpoint)))


if __name__=='__main__':main()
