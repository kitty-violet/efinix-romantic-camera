"""Combine a shared contour-gated surface filter with the current14Y tail.

The old engineered filter used coarse RGB-coordinate quantization. This probe
uses its source-only common Y contour for all colours, then the newer positive
14Y renderer and continuous Co/Cg. All work is inside existing B1 operations.
This explicit coefficient test is not presented as learned distillation.
"""
import argparse,copy
from pathlib import Path
import torch
from .engineer_b1_blocks import build_contour_guided
from .probe_b1_encoder_gauge import load
from .probe_b1_shared_positive_regions import install as tail_install
from .probe_b1_monotone_pair_decoder import make_board
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def install(net,threshold,field_gain):
    tail_install(net,0);params,residual=build_contour_guided(threshold=threshold,gain=2)
    for r in net.plan['nodes']:
        key=r['id'].replace('.','__')
        if r['id']=='up1_low_pw':break
        if key not in net.weight:continue
        q=params[r['id']];wi=torch.tensor(q.weights,dtype=torch.float32)
        if r['kind']=='dw3x3':wi=wi.reshape(r['channels'],1,3,3)
        else:wi=wi.reshape(r['channels'],r['kernel'],r['kernel'],r['input_channels']).permute(0,3,1,2)
        scale=torch.tensor(q.multiplier,dtype=torch.float32)/131072
        net.weight[key].copy_(wi*scale[:,None,None,None]);net.bias[key].copy_(torch.tensor(q.bias)*scale)
        getattr(net,key+'_m').copy_(torch.tensor(q.multiplier,dtype=torch.float64)[None,:,None,None])
    net.package['residual_encoding']=copy.deepcopy(residual)
    key='up1_low_pw';net.weight[key].zero_();net.bias[key].zero_();net.up1_low_pw_m.fill_(16384)
    for i in range(14):net.weight[key][i,0,0,0]=field_gain;net.bias[key][i]=63.5-field_gain*(12+8*i)
    net.weight[key][14,1,0,0]=2;net.bias[key][14]=-64
    net.weight[key][15,2,0,0]=3;net.bias[key][15]=-128


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);out=Path(a.out);out.mkdir(exist_ok=False);boards=[];records=[]
    for threshold,gain in ((12,2),(24,2),(24,8)):
        name=f't{threshold}_g{gain}';net=copy.deepcopy(base);install(net,threshold,gain);raw=net.export();actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');shots={}
        for row in m.load_data(a.data)['validation']:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(x,all_nodes=True);other,_=actual(net.round(x*128),True);assert all(torch.equal(v,other[k]) for k,v in nodes.items())
            assert len(raw)==20480 and (torch.diff(nodes['up1_high_dw'][:,:14],dim=1)<=0).all()
            im=m.t.r.c.rgb_image(nodes['rgb_head'][0]/128);im.save(out/f'{name}_{row["id"]}.png');shots[row['id']]=im
            records.append(dict(mode=name,id=row['id'],actual_all_nodes_error=0))
        saved=copy.deepcopy(state);saved.update(model=net.state_dict(),residual_encoding=copy.deepcopy(net.package['residual_encoding']),
            designed_contour_gated_direct14=True,contour_threshold=threshold,distance_gain= gain,learned=False,visual_acceptance=False)
        for k in ('optimizer','rng','torch_rng','cuda_rng'):saved.pop(k,None)
        torch.save(saved,out/f'{name}.pt');boards.append(shots);print('GUIDED_DIRECT_LUMA',name,'6_nodes0/order/20KiB',flush=True)
    make_board(boards,['共同轮廓12 / 亮度坡度2','共同轮廓24 / 亮度坡度2','共同轮廓24 / 亮度坡度8'],out/'comparison.png')
    write_json(out/'review.json',dict(records=records,source_sha256=file_hash(__file__),designed_not_learned=True,new_operators=0,visual_acceptance=False))


if __name__=='__main__':main()
