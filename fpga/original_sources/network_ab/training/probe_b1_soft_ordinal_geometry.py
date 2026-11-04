"""Retain fractional boundary positions in nested predicates at180p.

Compare box majority with a centre-weighted stencil and a less abrupt ramp.
Only two existing B1 convolution coefficients change. Tests distinguish exact
constant/rank arithmetic from source-image visual acceptance.
"""
import argparse,copy
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .audit_b1_common_coordinate_boundaries import hinges
from .build_b1_thirty_binary_regions import code_bits,GROUPS30
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

VARIANTS={'box9':(9,[[1,1,1],[1,1,1],[1,1,1]],14563),
 'box3':(3,[[1,1,1],[1,1,1],[1,1,1]],14563),
 'weighted5':(5,[[1,2,1],[2,3,2],[1,2,1]],8738),
 'weighted3':(3,[[1,2,1],[2,3,2],[1,2,1]],8738)}

@torch.no_grad()
def install(net,variant):
    gain,kernel,mult=VARIANTS[variant];key='res3_conv2__dw';net.weight[key].zero_();net.bias[key].zero_()
    getattr(net,key+'_m').fill_(mult);net.weight[key][:30,0]=torch.tensor(kernel,device=net.weight[key].device)*mult/131072
    key='res3_conv2__pw';net.weight[key].zero_();net.bias[key].zero_();getattr(net,key+'_m').fill_(131071)
    for i in range(30):net.weight[key][i,i,0,0]=gain*131071/131072
    net.bias[key][:30]=127*(1-gain)//2*131071/131072

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);setup=state['binary_coverage720'];dest=Path(a.out);dest.mkdir(exist_ok=False)
    levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float();pal=(levels@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])).clamp(-128,127)
    code=torch.arange(1215)[:,None,None];constant=F.pad(code_bits(code)*127,(0,0,0,0,0,2));rows=m.load_data(a.data)['validation']
    board=Image.new('RGB',(1680,1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    for j,name in enumerate(VARIANTS):
        net=copy.deepcopy(base);install(net,name)
        assert torch.equal(fragment(net,constant,'res3_conv1.pw')['rgb_head'][:,:,0,0],pal)
        c=torch.rand(2,3,48,64)*torch.tensor([14,8,8])[None,:,None,None]
        z=fragment(net,net.round(hinges(c)),'res3_conv1.pw')['up1_low_dw']
        assert all((torch.diff(z[:,lo:hi],dim=1)<=0).all() for lo,hi in GROUPS30)
        raw=net.export();assert len(raw)==20480;actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');ims={}
        for row in rows:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()-128
            nodes=net(x/128,all_nodes=True);other,_=actual(x,True);assert all(torch.equal(v,other[k]) for k,v in nodes.items())
            im=m.t.r.c.rgb_image(nodes['rgb_head'][0]/128);im.save(dest/(name+'_'+row['id']+'.png'));ims[row['id']]=im
        for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
            board.paste(ims[rid].crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+name,font=font,fill='white')
        saved=copy.deepcopy(state);saved.update(model=net.state_dict(),ordinal_geometry_variant=name,visual_acceptance=False)
        torch.save(saved,dest/(name+'.pt'));print('SOFT_ORDINAL_PASS',name,'1215colours/rank/6x720nodes/20KiB',flush=True)
    board.save(dest/'source_crops.png');write_json(dest/'review.json',dict(variants=VARIANTS,constant_palette_error=0,
      ordinal_rank_error=0,six720_actual_node_error=0,bytes=20480,extra_operators=0,visual_acceptance=False,parent_sha256=file_hash(a.checkpoint)))

if __name__=='__main__':main()
