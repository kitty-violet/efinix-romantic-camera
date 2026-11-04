"""Compare nonnegative ordinal-pair decoding with the radix field compiler.

Thirty existing ordered predicates form fifteen adjacent positive pairs. The
existing integer DW converts each pair into0/1/2. This eliminates subtractive
digit mixing without reducing15/9/9 palette levels or adding operations. The
quantized boundary quality must still be checked; constants alone are not QA.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .probe_b1_line_preserving_vote import install as vote_install
from .build_b1_thirty_binary_regions import code_bits,GROUPS30
from .train_b1_ternary_regions import region_codes
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

GROUPS15=((0,7),(7,11),(11,15))

@torch.no_grad()
def install(net):
    key='up1_low_pw';net.weight[key].zero_();net.bias[key].zero_();getattr(net,key+'_m').fill_(65536)
    for i in range(15):net.weight[key][i,2*i:2*i+2,0,0]=.5
    key='up2_low_dw';net.weight[key].zero_();net.bias[key].zero_();getattr(net,key+'_m').fill_(2064)
    net.weight[key][:15,0,1,1]=2064/131072
    key='up2_low_pw';net.weight[key].zero_();net.bias[key].zero_();getattr(net,key+'_m').fill_(131071)
    for c,(lo,hi) in enumerate(GROUPS15):net.weight[key][c,lo:hi,0,0]=8*131071/131072

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);dest=Path(a.out);dest.mkdir(exist_ok=False);setup=state['binary_coverage720']
    levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float()
    palette=(levels@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])).clamp(-128,127);pk=tuple(map(tuple,(palette+128).tolist()))
    rows=m.load_data(a.data)['validation'];boards=[];records=[]
    for name in ('radix','pair','pair_centre9'):
        net=copy.deepcopy(base)
        if name!='radix':install(net)
        if name=='pair_centre9':vote_install(net,'centre9')
        z=F.pad(code_bits(torch.arange(1215)[:,None,None])*127,(0,0,0,0,0,2))
        assert torch.equal(fragment(net,z,'res3_conv1.pw')['rgb_head'][:,:,0,0],palette)
        raw=net.export();actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');assert len(raw)==20480;ims={};injected={}
        for row in rows:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()-128
            nodes=net(x/128,all_nodes=True);other,_=actual(x,True);assert all(torch.equal(v,other[k]) for k,v in nodes.items())
            im=m.t.r.c.rgb_image(nodes['rgb_head'][0]/128);im.save(dest/f'{name}_{row["id"]}.png');ims[row['id']]=im
            codes=torch.from_numpy(region_codes(row['target'],row['region'],pk).astype(np.int64))[None]
            z=F.pad(net.round(F.avg_pool2d(code_bits(codes)*127,4)),(0,0,0,0,0,2))
            p=fragment(net,z,'up1_low_dw')['rgb_head'];injected[row['id']]=m.t.r.c.rgb_image(p[0]/128)
            target=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()-128
            records.append(dict(variant=name,id=row['id'],source_mae=float((nodes['rgb_head']-target).abs().mean()),injection_mae=float((p-target).abs().mean())))
        boards.append(ims);saved=copy.deepcopy(state);saved.update(model=net.state_dict(),monotone_pair_decoder=name!='radix',ordinal_geometry_variant=name,visual_acceptance=False)
        torch.save(saved,dest/(name+'.pt'));print('MONOTONE_PAIR_DECODER_PASS',name,'1215 constants/6x720 replay/20KiB',flush=True)
        make_board([injected],['target injection, not source'],dest/(name+'_injection.png'))
    make_board(boards,['radix','positive pairs','positive pairs centre9'],dest/'source_crops.png')
    write_json(dest/'review.json',dict(records=records,constant1215_error=0,source_replay_error=0,bytes=20480,
      extra_operators=0,visual_acceptance=False,parent_sha256=file_hash(a.checkpoint)))

def make_board(boards,names,path):
    board=Image.new('RGB',(420*len(boards),1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    for j,(name,ims) in enumerate(zip(names,boards)):
        for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
            board.paste(ims[rid].crop(box),(j*420,i*314+34));draw.text((j*420+3,i*314+4),label+' | '+name,font=font,fill='white')
    board.save(path)

if __name__=='__main__':main()
