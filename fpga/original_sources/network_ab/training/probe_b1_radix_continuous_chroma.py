"""Remove only chroma hard decisions while preserving15 luma layers.

Unlike the old7-luma-level tournament experiment, keep this checkpoint's
14 ordered Y fields, six radix Y bits, g11 contour and exact1215 constant
colours. Co/Cg sums use existing PW/DW rows without hard radix carries.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .build_b1_thirty_binary_regions import code_bits
from .audit_b1_colour_fringe_origin import fringe,two_colour_support
from .train_b1_ternary_regions import region_codes
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def install(net):
    key='up1_low_pw';net.weight[key][6:].zero_();net.bias[key][6:].zero_()
    for row,lo,hi in ((6,14,22),(7,22,30)):
        net.weight[key][row,lo:hi,0,0]=1/16;net.up1_low_pw_m[:,row]=8192
        net.weight['up1_high_dw'][row]=net.weight['up1_high_dw'][0];net.up1_high_dw_m[:,row]=net.up1_high_dw_m[:,0]
        net.bias['up1_high_dw'][row]=0
        net.weight['up2_low_dw'][row].zero_();net.weight['up2_low_dw'][row,0,1,1]=1
        net.bias['up2_low_dw'][row]=0;net.up2_low_dw_m[:,row]=65536
    for row,src in ((1,6),(2,7)):
        net.weight['up2_low_pw'][row].zero_();net.weight['up2_low_pw'][row,src,0,0]=1
        net.bias['up2_low_pw'][row]=0;net.up2_low_pw_m[:,row]=65536


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);net=copy.deepcopy(base);install(net)
    out=Path(a.out);out.mkdir(exist_ok=False);data=m.load_data(a.data)
    z=F.pad(code_bits(torch.arange(1215)[:,None,None])*127,(0,0,0,0,0,2))
    expected=fragment(base,z,'up1_low_dw')['rgb_head'];got=fragment(net,z,'up1_low_dw')['rgb_head'];assert torch.equal(got,expected)
    pal=expected[:,:,0,0].numpy()+128;pk=tuple(map(tuple,pal.tolist()));raw=net.export();assert len(raw)==20480
    (out/'coefficients-padded.bin').write_bytes(raw)
    ss={k:v for k,v in state.items() if k not in ('optimizer','rng','torch_rng','cuda_rng')};ss.update(model=copy.deepcopy(net.state_dict()),radix_luma_continuous_chroma=True,
        old_luminance_path_bit_exact=True,constant_palette_preserved=1215,extra_inference_operations=0,visual_acceptance=False)
    torch.save(ss,out/'initial.pt');actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');shots={};records=[]
    for row in data['validation']:
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
        before=base(x,all_nodes=True);after=net(x,all_nodes=True);other,_=actual(x*128,True)
        assert all(torch.equal(v,other[k]) for k,v in after.items()) and torch.equal(before['up2_high_dw'][:,0],after['up2_high_dw'][:,0])
        target=m.t.r.read(row['target']).astype(np.float32);codes=region_codes(row['target'],row['region'],pk).astype(np.int64);ca,cb,valid=two_colour_support(codes,pal)
        images={}
        for name,nodes in (('before',before),('continuous_chroma',after)):
            p=nodes['rgb_head'][0].permute(1,2,0).numpy()+128;images[name]=p
            records.append(dict(id=row['id'],mode=name,mae=float(np.abs(p-target).mean()),fringe=fringe(p,ca,cb,valid)))
        for rid,box,title in m.t.r.c.CROPS:
            if rid==row['id']:shots[rid,title]=[Image.fromarray(z.astype(np.uint8)).crop(box) for z in (target,images['before'],images['continuous_chroma'])]
        print('RADIX_CONTINUOUS_CHROMA',records[-2:],flush=True)
    board=Image.new('RGB',(1260,1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,(rid,box,title) in enumerate(m.t.r.c.CROPS):
        for j,(im,label) in enumerate(zip(shots[rid,title],('目标','当前实际B1','仅解除色度硬判定·实际B1'))):
            board.paste(im,(420*j,314*i+34));draw.text((420*j+2,314*i+4),title+' | '+label,font=font,fill='white')
    board.save(out/'comparison.png');write_json(out/'review.json',dict(records=records,constant_colours_error=0,luminance_coordinate_error=0,all_nodes_integer_error=0,
        bytes=20480,source_sha256=file_hash(__file__),checkpoint_sha256=file_hash(a.checkpoint),extra_inference_operations=0,visual_acceptance=False))


if __name__=='__main__':main()
