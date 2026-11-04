"""Analytical colour ladders and subpixel slanted edges, CPU diagnostic.

Tests the codec itself; not a photographic stylization test. A flat result may
have no stair steps, so contour residual, transition width and contrast are all
reported independently. No extra filter is applied to model output.
"""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from . import train_b1_minimum_area as m
from .runtime import configure_workspace, contract_guard, write_json, file_hash


def slant(slope,phase):
    yy,xx=np.mgrid[:256,:384].astype(np.float32)
    line=128+slope*(xx-192)+phase
    cover=np.clip(.5+(yy-line)/np.sqrt(1+slope*slope),0,1)
    a=np.array([48,72,104],np.float32);b=np.array([184,200,224],np.float32)
    return np.rint(a+cover[:,:,None]*(b-a)).astype(np.uint8)


def edge_metrics(rgb,slope,phase):
    a=np.array([48,72,104],np.float32);direction=np.array([136,128,120],np.float32)
    q=((rgb.astype(np.float32)-a)*direction).sum(2)/(direction*direction).sum()
    positions=[];widths=[];contrasts=[];columns=[]
    for x in range(40,344):
        true=128+slope*(x-192)+phase
        if not 40<true<215:continue
        lo,hi=int(true)-20,int(true)+21
        v=q[lo:hi,x];low=np.median(q[lo-12:lo-4,x]);high=np.median(q[hi+4:hi+12,x])
        contrast=high-low;contrasts.append(float(contrast))
        if contrast<.25:continue
        normalized=(v-low)/contrast
        def crossing(level):
            ids=np.flatnonzero((normalized[:-1]<=level)&(normalized[1:]>level))
            if not len(ids):return np.nan
            j=int(ids[np.argmin(np.abs(ids+lo-true))])
            return lo+j+(level-normalized[j])/(normalized[j+1]-normalized[j])
        p=crossing(.5);w=crossing(.9)-crossing(.1)
        if np.isfinite(p) and np.isfinite(w):positions.append(p);widths.append(w/np.sqrt(1+slope*slope));columns.append(x)
    assert len(positions)>30
    pos=np.array(positions);xx=np.array(columns)
    fitted=np.polyval(np.polyfit(xx,pos,1),xx);residue=pos-fitted
    return dict(contour_residual_rms=float(np.sqrt(np.mean(residue**2))),
        contour_residual_p95=float(np.percentile(np.abs(residue),95)),
        transition_width_normal_median=float(np.median(widths)),
        contrast_ratio=float(np.mean(contrasts)),valid_columns=len(pos))


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--integer',action='store_true');a=ap.parse_args()
    configure_workspace();contract_guard();torch.set_num_threads(2)
    dest=Path(a.out);dest.mkdir(exist_ok=False)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(state['model'])
    if 'residual_encoding' in state:net.package['residual_encoding']=copy.deepcopy(state['residual_encoding'])
    net.quantized=a.integer
    if a.integer:
        replay=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu')
    def infer(rgb):
        x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)/128-1
        if a.integer:
            p=replay(x*128)/128
            assert float((p-net(x)[0]).abs().max())==0
        else:p=net(x)[0]
        return np.array(m.t.r.c.rgb_image(p[0]))
    records=[];images=[]
    for slope in (.25,.5,1.,-.5):
        for phase in (.1,.35,.65,.9):
            original=slant(slope,phase);output=infer(original)
            records.append(dict(slope=slope,phase=phase,input=edge_metrics(original,slope,phase),output=edge_metrics(output,slope,phase)))
            if phase==.35:images.append((original,output))
    levels=np.rint(np.linspace(32,224,16)).astype(np.uint8)
    ladder=np.repeat(np.repeat(levels,24)[None,:,None],128,0).repeat(3,2)
    out=infer(ladder);means=[out[32:96,i*24+7:(i+1)*24-7].mean((0,1)).tolist() for i in range(16)]
    gaps=np.diff(np.mean(means,1));assert np.all(np.diff(levels.astype(int))>0)
    result=dict(records=records,level_input=levels.tolist(),level_output_rgb=means,
        distinct_adjacent_levels_over4=int((gaps>4).sum())+1,
        level_monotone=bool((gaps>0).all()),level_mae=float(np.abs(np.array(means)-levels[:,None]).mean()),
        checkpoint_sha256=file_hash(a.checkpoint),integer=a.integer,source_to_style=False,synthetic_diagnostic=True)
    board=Image.new('RGB',(768,300*len(images)+160),'#17202a');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(source,output) in enumerate(images):
        for j,(pic,title) in enumerate(zip((source,output),('解析斜边输入','同B1输出·编解码诊断'))):
            board.paste(Image.fromarray(pic),(384*j,300*i+35));d.text((384*j+4,300*i+4),title,font=font,fill='white')
    board.paste(Image.fromarray(ladder),(0,300*len(images)+30));board.paste(Image.fromarray(out),(384,300*len(images)+30))
    board.save(dest/'comparison.png');write_json(dest/'results.json',result)
    print('SAMPLING_EDGE_AUDIT',dict(levels=result['distinct_adjacent_levels_over4'],level_mae=result['level_mae'],
        input_rms=np.mean([r['input']['contour_residual_rms'] for r in records]),
        output_rms=np.mean([r['output']['contour_residual_rms'] for r in records]),
        width=np.mean([r['output']['transition_width_normal_median'] for r in records])),flush=True)


if __name__=='__main__':main()
