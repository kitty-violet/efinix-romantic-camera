"""Separate colour-region prediction from final integer interpolation.

Compare existing integer count rounding, nearest transport, and 4 coverage
levels at native720. All variants are ordinary B1 weights; no external filter.
"""
import argparse
import copy
import hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .probe_b1_decoder_coverage import fragment
from .audit_b1_native_colour_regions import nearest_codes,spatial_permutation_audit
from .minimum_area_blocks import components
from .runtime import write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    assert state.get('group_native_count_decoder')=='box'
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
    initial=copy.deepcopy(net.state_dict());rows=m.load_data()['validation'];palette=(np.array(state['palette'])*128+128).round().clip(0,255)
    patterns=((torch.arange(32768)[:,None]>>torch.arange(15)[None])&1).float()
    z=torch.zeros(32768,16,1,1);z[:,:15,0,0]=patterns*127
    constant=fragment(net,z,'up2_low_dw')['rgb_head'].clone();images={};records=[]
    for kind in ('current','nearest','coverage4'):
        net.load_state_dict(initial)
        if kind!='current':
            coverage=4 if kind=='coverage4' else 1
            kernel=torch.ones(3,3) if kind=='coverage4' else torch.zeros(3,3)
            if kind=='nearest':kernel[1,1]=1
            for c,increment in enumerate((16,9,6)):
                divisor=int(kernel.sum())*increment
                mult=(131072*coverage+divisor-1)//divisor
                net.up2_high_dw_m[:,c]=mult
                net.weight['up2_high_dw'][c,0]=kernel*mult/131072
                net.weight['rgb_head'][:,c]/=coverage
        error=float((fragment(net,z,'up2_low_dw')['rgb_head']-constant).abs().max());assert error==0
        spatial=spatial_permutation_audit(net);assert max(v['max_rgb_error'] for v in spatial)==0
        sub=dest/kind;sub.mkdir();blob=net.export();(sub/'coefficients-padded.bin').write_bytes(blob)
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(blob),device='cpu');stats=[]
        for row in rows:
            rgb=m.t.r.read(row['path']);x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)-128
            nodes,_=actual(x,True);ours=net(x/128,all_nodes=True)
            difference=max(float((v-ours[k]).abs().max()) for k,v in nodes.items());assert difference==0
            im=m.t.r.c.rgb_image(nodes['rgb_head'][0]/128);im.save(sub/(row['id']+'.png'));images[kind,row['id']]=im
            codes,membership=nearest_codes(np.array(im),palette);lab=components(codes);area=np.bincount(lab.ravel())
            stats.append(dict(id=row['id'],all_node_error=difference,nearest_palette_small_area_fraction=float(area[area<256].sum()/area.sum()),
                nearest_palette_regions=len(area),**membership))
        if kind!='current':
            saved={k:v for k,v in state.items() if k not in ('optimizer','bounded_training_model','rng','torch_rng','cuda_rng')}
            saved.update(model=copy.deepcopy(net.state_dict()),manual_colour_transport=kind,manual_decoder_change=True,
                training_proxy_requires_update=True,visual_acceptance=False,minimum_area_inference_guarantee=False)
            torch.save(saved,sub/'candidate.pt')
        records.append(dict(kind=kind,flat_32768_error=error,spatial=spatial,stats=stats,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
        print('COLOUR_TRANSPORT',kind,stats,flush=True)
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14);by_id={r['id']:r for r in rows}
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        pics=[Image.open(by_id[key]['target'])]+[images[k,key] for k in ('current','nearest','coverage4')]
        for j,(im,title) in enumerate(zip(pics,('离线目标','原整数颜色插值','不混合已判颜色','边缘4级覆盖'))):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png')
    write_json(dest/'review.json',dict(parent=a.checkpoint,parent_sha256=file_hash(a.checkpoint),records=records,
        changed_classifier=False,added_inference_operations=0,external_filter=False,actual_board=False,visual_acceptance=False,
        area_analysis_uses_nearest_base_palette=True))


if __name__=='__main__':main()
