"""Use four already-present unused channels for coarse boundary contrast.

The current decoder uses3 of8 final features. Copy its simplified luminance
into4 existing channels, then use the existing720p depthwise ReLU for positive
and negative Sobel responses. A one-level deadband avoids outlining every colour
step. RGB subtracts modest shared ink. No source-photo edge, external filter,
new channel, operation, or hardware change is introduced.
"""
import argparse,copy
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .b1_binary_layout import fields_for_layout
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

@torch.no_grad()
def install(net,strength):
    assert 0<=strength<=8
    assert net.weight['up2_low_pw'][3:].abs().max()==0 and net.weight['rgb_head'][:,3:].abs().max()==0
    sobel=torch.tensor([[-1.,0,1],[-2,0,2],[-1,0,1]],device=net.weight['rgb_head'].device)
    for c,kernel in zip(range(3,7),(sobel,-sobel,sobel.T,-sobel.T)):
        key='up2_low_pw';net.weight[key][c]=net.weight[key][0];net.bias[key][c]=net.bias[key][0]
        getattr(net,key+'_m')[:,c]=getattr(net,key+'_m')[:,0]
        key='up2_high_dw';getattr(net,key+'_m')[:,c]=8192;net.weight[key][c,0]=kernel/16;net.bias[key][c]=-2
        net.weight['rgb_head'][:,c,0,0]=-strength

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--strengths',type=float,nargs='+',default=[0,.5,1,1.5]);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);dest=Path(a.out);dest.mkdir(exist_ok=False);setup=state['binary_coverage720']
    code=torch.arange(1215)[:,None,None];constant=F.pad(fields_for_layout(code,setup)*127,(0,0,0,0,0,1))
    palette=fragment(base,constant,'up1_low_pw')['rgb_head'];rows=m.load_data(a.data)['validation']
    board=Image.new('RGB',(420*len(a.strengths),1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    records=[]
    for j,strength in enumerate(a.strengths):
        net=copy.deepcopy(base);install(net,strength)
        assert torch.equal(fragment(net,constant,'up1_low_pw')['rgb_head'],palette)
        raw=net.export();assert len(raw)==20480;actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');ims={};fractions=[]
        for row in rows:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()-128
            nodes=net(x/128,all_nodes=True);other,_=actual(x,True);assert all(torch.equal(v,other[k]) for k,v in nodes.items())
            fractions.append(float((nodes['up2_high_dw'][:,3:7].sum(1)>0).float().mean()))
            im=m.t.r.c.rgb_image(nodes['rgb_head'][0]/128);im.save(dest/(str(strength)+'_'+row['id']+'.png'));ims[row['id']]=im
        for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
            board.paste(ims[rid].crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | strength '+str(strength),font=font,fill='white')
        saved=copy.deepcopy(state);saved.update(model=net.state_dict(),existing_contour_channels=dict(channels=[3,4,5,6],strength=strength,source='simplified luminance',deadband=2),visual_acceptance=False)
        torch.save(saved,dest/('ink'+str(strength)+'.pt'));records.append(dict(strength=strength,active_fraction=fractions))
        print('EXISTING_CONTOUR_CHANNELS_PASS',strength,'constant1215/6x720nodes/20KiB',flush=True)
    board.save(dest/'source_crops.png');write_json(dest/'review.json',dict(records=records,constant_palette_error=0,
      six720_actual_node_error=0,bytes=20480,extra_operators=0,visual_acceptance=False,parent_sha256=file_hash(a.checkpoint)))

if __name__=='__main__':main()
