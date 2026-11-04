"""Known straight boundaries through real B1 decoder, all four input phases.

This isolates decoder geometry from source classification. It does not show
source predictions or establish a deployable style. No image postfilter is used.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_binary_confidence import set_gain
from .probe_b1_decoder_coverage import fragment
from .b1_binary_layout import fields_for_layout
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def contour_metrics(rgb,a,b,truth_x):
    delta=b-a;alpha=((rgb-a[:,None,None])*delta[:,None,None]).sum(0)/delta.square().sum()
    positions=[];widths=[];backtracks=[]
    for y in range(16,rgb.shape[-2]-16):
        row=alpha[y];crossings=[]
        for level in (.1,.5,.9):
            ids=torch.where((row[:-1]<=level)&(row[1:]>level))[0]
            assert len(ids)>0
            j=int(ids[(ids.float()-truth_x[y]).abs().argmin()])
            crossings.append(j+float((level-row[j])/(row[j+1]-row[j])))
        positions.append(crossings[1]);widths.append(crossings[2]-crossings[0])
        lo=max(0,int(truth_x[y])-12);hi=min(len(row),int(truth_x[y])+12)
        backtracks.append(float(F.relu(-torch.diff(row[lo:hi])).sum()))
    positions=np.array(positions);ys=np.arange(len(positions));line=np.polyval(np.polyfit(ys,positions,1),ys)
    return dict(contour_rms_px=float(np.sqrt(np.mean((positions-line)**2))),
                boundary_width_px=float(np.mean(widths)),backtrack=float(np.mean(backtracks)))


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    torch.set_num_threads(2);state,base=load(args.checkpoint);setup=state['binary_coverage720'];dest=Path(args.out);dest.mkdir(exist_ok=False)
    yy,xx=torch.meshgrid(torch.arange(96),torch.arange(128),indexing='ij');records=[];pictures={}
    pairs=(((5,4,4),(9,4,4)),((5,3,4),(6,6,3)),((2,4,5),(11,4,3)))
    matrix=torch.tensor(setup['matrix']).float();bias=torch.tensor(setup['bias']).float()
    for gain in (4,8,16,64):
        net=copy.deepcopy(base);set_gain(net,gain)
        for pair_index,(ca,cb) in enumerate(pairs):
            a=(matrix@torch.tensor(ca).float()+bias).clamp(-128,127);b=(matrix@torch.tensor(cb).float()+bias).clamp(-128,127)
            for slope in (.2,.5,1.):
                for phase in range(4):
                    truth_x=64+slope*(torch.arange(96)-48)+phase-.5
                    right=xx>truth_x[:,None];code=torch.where(right,cb[0]*81+cb[1]*9+cb[2],ca[0]*81+ca[1]*9+ca[2])[None]
                    score=F.pad(F.avg_pool2d(fields_for_layout(code,setup)*96+16,4),(0,0,0,0,0,1))
                    rgb=fragment(net,score,'up1_low_pw')['rgb_head'][0]
                    metrics=contour_metrics(rgb,a,b,truth_x)
                    lo=torch.minimum(a,b)[:,None,None];hi=torch.maximum(a,b)[:,None,None]
                    metrics['component_overshoot']=float(torch.maximum(F.relu(rgb-hi),F.relu(lo-rgb)).max())
                    records.append(dict(gain=gain,pair=pair_index,slope=slope,phase=phase,**metrics))
                    if pair_index==1 and slope in (.2,1.) and phase==1:pictures[gain,slope]=m.t.r.c.rgb_image(rgb/128)
        subset=[r for r in records if r['gain']==gain]
        print('BINARY_BOUNDARY_PHASE',gain,{k:float(np.mean([r[k] for r in subset])) for k in metrics},flush=True)
    board=Image.new('RGB',(4*256,2*220),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    for j,gain in enumerate((4,8,16,64)):
        for i,slope in enumerate((.2,1.)):
            board.paste(pictures[gain,slope].resize((256,192),Image.Resampling.NEAREST),(256*j,220*i+28))
            draw.text((256*j+3,220*i+3),f'坡度{gain} / 斜率{slope} / 2倍近看',font=font,fill='white')
    board.save(dest/'straight_edges.png');write_json(dest/'review.json',dict(records=records,source_prediction=False,
        means={str(g):{k:float(np.mean([r[k] for r in records if r['gain']==g])) for k in metrics} for g in (4,8,16,64)},
        checkpoint_sha256=file_hash(args.checkpoint)))


if __name__=='__main__':main()
