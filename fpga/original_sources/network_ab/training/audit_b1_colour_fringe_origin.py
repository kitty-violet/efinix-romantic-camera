"""Separate palette, fixed rendering, and source prediction colour errors.

Palette/SDF images use known labels solely as diagnostic interventions. Only
the column marked actual B1 is source inference from the20KiB coefficient image.
Test colour fringes outside a common two-colour RGB segment, including fringes
that remain inside every channel's individual range. No inference filter.
"""
import argparse
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_ordered_distance_geometry import distance_fields
from .probe_b1_decoder_coverage import fragment
from .train_b1_ternary_regions import region_codes
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def two_colour_support(codes,palette):
    lo=cv2.erode(codes.astype(np.uint16),np.ones((5,5),np.uint8)).astype(np.int64)
    hi=cv2.dilate(codes.astype(np.uint16),np.ones((5,5),np.uint8)).astype(np.int64)
    h,w=codes.shape;pad=np.pad(codes,2,mode='edge');other=np.zeros((h,w),bool)
    for yy in range(5):
        for xx in range(5):
            z=pad[yy:yy+h,xx:xx+w];other|=(z!=lo)&(z!=hi)
    a,b=palette[lo],palette[hi];valid=(lo!=hi)&~other&(np.linalg.norm(b-a,axis=-1)>=16)
    return a,b,valid


def fringe(p,a,b,valid):
    v=b-a;alpha=np.clip(((p-a)*v).sum(-1)/(v*v).sum(-1).clip(1),0,1)
    error=np.linalg.norm(p-(a+alpha[...,None]*v),axis=-1)
    y=lambda z:(z[...,0]+2*z[...,1]+z[...,2])/4
    white=np.maximum(y(p)-np.maximum(y(a),y(b)),0)
    if not valid.any():return dict(pixels=0)
    return dict(pixels=int(valid.sum()),off_common_rgb_segment_mean=float(error[valid].mean()),
        off_common_rgb_segment_p95=float(np.quantile(error[valid],.95)),
        off_segment_over8_fraction=float((error[valid]>8).mean()),bright_beyond_endpoints_over4_fraction=float((white[valid]>4).mean()))


def colour_stats(p,t,core):
    d=p-t;co=(d[...,0]-d[...,2])/2;cg=(2*d[...,1]-d[...,0]-d[...,2])/4
    y=(d[...,0]+2*d[...,1]+d[...,2])/4
    if not core.any():return dict(pixels=0)
    return dict(pixels=int(core.sum()),mae=float(np.abs(d[core]).mean()),luma_bias=float(y[core].mean()),
        Co_bias=float(co[core].mean()),Cg_bias=float(cg[core].mean()),
        note='Negative Cg means excess R+B relative to G; report per crop, not only full-frame mean.')


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(2);state,net=load(a.checkpoint);data=m.load_data(a.data)
    out=Path(a.out);out.mkdir(exist_ok=False);raw=net.export();actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
    setup=state['binary_coverage720'];lev=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float()
    palette=(lev@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255).numpy()
    pk=tuple(map(tuple,palette.tolist()));records=[];shots={};crop_records=[]
    for row in data['validation']:
        rid=row['id'];target=m.t.r.read(row['target']).astype(np.float32)
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()-128
        nn,_=actual(x,True);nodes=net(x/128,all_nodes=True);assert all(torch.equal(v,nn[k]) for k,v in nodes.items())
        codes=region_codes(row['target'],row['region'],pk).astype(np.int64)
        truth=distance_fields(torch.from_numpy(codes)[None],8)
        rendered=fragment(net,F.pad(net.round(truth),(0,0,0,0,0,2)),'up1_low_dw')['rgb_head']
        stages=dict(target=target,palette_labels=palette[codes],known_sdf_decoder=rendered[0].permute(1,2,0).numpy()+128,
            actual_b1=nn['rgb_head'][0].permute(1,2,0).numpy()+128)
        labels=m.load_region(row['region']);boundary=np.zeros(labels.shape,bool)
        boundary[:,1:]|=labels[:,1:]!=labels[:,:-1];boundary[1:]|=labels[1:]!=labels[:-1]
        core=cv2.distanceTransform((~boundary).astype(np.uint8),cv2.DIST_L2,5)>4
        ca,cb,valid=two_colour_support(codes,palette)
        records.append(dict(id=rid,all_node_error=0,stages={k:dict(core=colour_stats(im,target,core),fringe=fringe(im,ca,cb,valid)) for k,im in stages.items()}))
        for key,box,title in m.t.r.c.CROPS:
            if key!=rid:continue
            xx,yy,xe,ye=box;sl=np.s_[yy:ye,xx:xe]
            crop_records.append(dict(id=rid,crop=title,box=box,stages={k:dict(core=colour_stats(im[sl],target[sl],core[sl]),fringe=fringe(im[sl],ca[sl],cb[sl],valid[sl])) for k,im in stages.items()}))
            for k,im in stages.items():shots[rid,title,k]=Image.fromarray(im.astype(np.uint8)).crop(box)
        print('COLOUR_FRINGE_ORIGIN',rid,records[-1]['stages']['actual_b1'],flush=True)
    board=Image.new('RGB',(1680,1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,(rid,box,title) in enumerate(m.t.r.c.CROPS):
        for j,(k,label) in enumerate((('target','当前目标'),('palette_labels','已知标签·仅色表诊断'),('known_sdf_decoder','已知轮廓·解码诊断'),('actual_b1','实际B1源图预测'))):
            board.paste(shots[rid,title,k],(420*j,314*i+34));draw.text((420*j+3,314*i+4),title+' | '+label,font=font,fill='white')
    board.save(out/'comparison.png')
    write_json(out/'review.json',dict(records=records,crop_records=crop_records,all_node_error=0,bytes=len(raw),
        checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),known_label_columns_not_deployable=True,
        fringe_mask='Exactly two target palette codes in5x5; RGB endpoint contrast>=16. Not semantic and does not cover all fringes.',visual_acceptance=False))
    assert len(raw)==20480 and net.export()==raw
    print('COLOUR_FRINGE_ORIGIN_COMPLETE',flush=True)


if __name__=='__main__':main()
