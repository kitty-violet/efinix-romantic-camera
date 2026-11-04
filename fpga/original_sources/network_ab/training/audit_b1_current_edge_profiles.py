"""Locate Y threshold dispersion versus transition width on straight edges.

Profiles are diagnostic resampling only, never a deployment operation.
Compare prediction with known labels through the SAME frozen integer tail.
"""
import argparse
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .b1_joint_interface_targets import joint_interface_fields
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def crossing(v,level):
    d=v-level;idx=np.flatnonzero(d[:-1]*d[1:]<=0)
    if not len(idx):return None
    loc=idx+(level-v[idx])/(v[idx+1]-v[idx]+1e-10)-20
    return float(loc[np.argmin(np.abs(loc))])


def measures(z,lo,hi):
    q=z[min(lo,hi):max(lo,hi)]/127
    if hi<lo:q=q[:,::-1]
    centres=[];widths=[];back=[]
    for v in q:
        c=crossing(v,63.5/127);a=crossing(v,.1);b=crossing(v,.9)
        if c is not None:centres.append(c)
        if a is not None and b is not None:widths.append(abs(b-a))
        back.append(float(np.maximum(-np.diff(v),0).sum()))
    return dict(active=max(lo,hi)-min(lo,hi),crossings=len(centres),
        centre_span=float(np.ptp(centres)) if len(centres)>1 else 0.,
        centre_std=float(np.std(centres)) if centres else None,
        widths=widths,backtracking=float(np.mean(back)),
        false_bright=float((z[max(lo,hi):]/127).mean()) if max(lo,hi)<14 else 0.)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);_,net=load(a.checkpoint);data=m.load_data(a.data);records=[]
    for row in data['validation']:
        with np.load(row['native_luma_codes']) as zz:code=zz['codes'].astype(np.int64)
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
        nodes=net(x,all_nodes=True)
        truth=joint_interface_fields(torch.from_numpy(code)[None])[:,:14]
        known=fragment(net,F.pad(net.round(truth),(0,0,0,0,0,2)),'up1_low_pw')
        h,w=code.shape;yy=code//81
        arrays={name:cv2.resize(n['up2_low_dw'][0,:14].numpy().transpose(1,2,0),(w,h),interpolation=cv2.INTER_LINEAR).transpose(2,0,1) for name,n in [('actual',nodes),('known',known)]}
        selected=[]
        for axis in (0,1):
            lab=yy if axis==1 else yy.T
            ys,xs=np.nonzero(np.abs(np.diff(lab,axis=1))>=3)
            for y,x0 in zip(ys,xs):
                x0+=1
                if x0<21 or x0>=lab.shape[1]-21 or y<3 or y>=lab.shape[0]-3:continue
                lo=int(lab[y,x0-1]);hi=int(lab[y,x0])
                if not ((lab[y,x0-12:x0]==lo).all() and (lab[y,x0:x0+12]==hi).all()):continue
                if not ((lab[y-2:y+3,x0-8:x0-4]==lo).all() and (lab[y-2:y+3,x0+4:x0+8]==hi).all()):continue
                if any(ax==axis and abs(y-py)<6 and abs(x0-px)<6 for ax,py,px,_,_ in selected):continue
                selected.append((axis,int(y),int(x0),lo,hi))
        for axis,y,x0,lo,hi in selected:
            rec=dict(id=row['id'],axis=axis,xy=[x0,y],levels=[lo,hi])
            for name,z in arrays.items():
                arr=z if axis==1 else z.transpose(0,2,1)
                rec[name]=measures(arr[:,y,x0-20:x0+21],lo,hi)
            records.append(rec)
        print('CURRENT_EDGE_PROFILES',row['id'],len(selected),flush=True)
    summary={}
    for name in ('actual','known'):
        rr=[r[name] for r in records];width=np.array([w for r in rr for w in r['widths']]);span=np.array([r['centre_span'] for r in rr])
        summary[name]=dict(samples=len(rr),width_p50=float(np.median(width)),width_p90=float(np.quantile(width,.9)),
            span_p50=float(np.median(span)),span_p90=float(np.quantile(span,.9)),
            missing_crossing_fraction=1-sum(r['crossings'] for r in rr)/sum(r['active'] for r in rr),
            backtracking=float(np.mean([r['backtracking'] for r in rr])))
    write_json(a.out,dict(records=records,summary=summary,node='up2_low_dw actual decisions; diagnostic bilinear profiles',checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),diagnostic_only=True))
    print('CURRENT_EDGE_SUMMARY',summary,flush=True)


if __name__=='__main__':main()
