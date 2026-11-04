"""Isolate encoder activation rounding while keeping the integer colour decoder.

The hybrid float encoder is diagnostic only. It uses exactly the exported
dequantized coefficients; there is no training and no target input.
"""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .probe_b1_decoder_coverage import fragment
from .train_b1_projected_count_regions import region_geometry
from .runtime import write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();torch.set_num_threads(2);dest=Path(a.out);dest.mkdir(exist_ok=False)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=copy.deepcopy(state['residual_encoding'])
    floating=copy.deepcopy(net);floating.quantized=False
    for key,w in floating.weight.items():
        scale=(getattr(floating,key+'_m')/131072).float().flatten()
        w.copy_(net.round(w/scale[:,None,None,None]).clamp(-127,127)*scale[:,None,None,None])
        floating.bias[key].copy_(net.round(floating.bias[key]/scale)*scale)
    assert floating.export()==net.export()
    data=m.load_data(a.data);rows=[];images={}
    for row in data['validation']:
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
        original=net(x,all_nodes=True);continuous=floating(x,all_nodes=True)
        # Final encoder interface remains integer; only preceding encoder
        # activation and residual rounds are removed in this counterfactual.
        part=fragment(net,original['up1_low_pw'],'up1_low_pw')
        assert torch.equal(part['rgb_head'],original['rgb_head'])
        hybrid=fragment(net,net.round(continuous['up1_low_pw']),'up1_low_pw')
        record=dict(id=row['id'],rgb_gap=float((hybrid['rgb_head']-original['rgb_head']).abs().mean()),nodes=[])
        for name,value in original.items():
            if name=='input':continue
            record['nodes'].append(dict(name=name,std_median=float(value.std((-2,-1)).median()),
                activation_rounding_gap=float((value-continuous[name]).abs().mean())))
            if name=='up1_low_pw':break
        for label,nodes in [('actual',original),('float_encoder_diagnostic',hybrid)]:
            z=nodes['up2_low_dw'][0,:15]
            codes=(81*z[:7].sum(0)+9*z[7:11].sum(0)+z[11:15].sum(0)).long().numpy()
            record[label]=region_geometry(codes)
            im=m.t.r.c.rgb_image(nodes['rgb_head'][0]/128);images[row['id'],label]=im
            im.save(dest/f'{row["id"]}_{label}.png')
        rows.append(record);print('ENCODER_ROUNDING',row['id'],record['rgb_gap'],record['actual'],record['float_encoder_diagnostic'],flush=True)
    board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    byid={r['id']:r for r in data['validation']}
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        choices=[('target',Image.open(byid[key]['target']))]+[(k,images[key,k]) for k in ('actual','float_encoder_diagnostic')]
        for j,(title,im) in enumerate(choices):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png');write_json(dest/'review.json',dict(checkpoint_sha256=file_hash(a.checkpoint),records=rows,
        same_exported_coefficients=True,integer_decoder_unchanged=True,original_fragment_equal=True,hybrid_deployable=False))


if __name__=='__main__':main()
