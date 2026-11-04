"""Fit a monotone, separable Y/Co/Cg palette using existing B1 channels.

Unrestricted RGB hinges can fold the chroma axis. Here Cg interval slopes
are constrained positive and Y/Co remain affine. No change to geometry.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .calibrate_b1_radix_chroma import add_hinges,constants,fit_head
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .b1_joint_interface_targets import joint_interface_fields
from .train_b1_ternary_regions import region_codes
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def bounded_fit(x,y,w,prior,lo,hi,anchor_x,ridge=.01):
    g=x.T@(x*w[:,None]);rhs=x.T@(y*w)
    ag=anchor_x.T@anchor_x/len(anchor_x);g+=ridge*ag;rhs+=ridge*ag@prior
    step=1/torch.linalg.eigvalsh(g).max();s=prior.clone()
    for _ in range(3000):s=(s-step*(g@s-rhs)).maximum(lo).minimum(hi)
    return s


def rgb_solution(y,co,cg):
    # Cg = intercept + six positive slopes over [0,2],[2,3],...,[6,8].
    # Re-express the slopes with the five existing hinge features (8*code).
    cy=torch.zeros(9,dtype=torch.float64);cc=cy.clone();cgvec=cy.clone()
    cy[0]=y[1]/8;cy[8]=y[0]
    cc[1]=co[1]/8;cc[8]=co[0]
    cgvec[2]=cg[1]/8;cgvec[3:8]=(cg[2:]-cg[1:-1])/8;cgvec[8]=cg[0]
    result=torch.stack((cy+cc-cgvec,cy+cgvec,cy-cc-cgvec),1);result[8]-=128
    return result


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False);state,base=load(a.checkpoint);net=copy.deepcopy(base);add_hinges(net)
    data=m.load_data(a.data);target=[];weight=[];counts={}
    for r in data['training']:counts[r['video']]=counts.get(r['video'],0)+1
    for r in data['training']:
        rgb=m.t.r.read(r['target']).astype(np.float64);lab=m.load_region(r['region']).astype(np.int64);n=np.bincount(lab.ravel());keep=n>0
        means=np.stack([np.bincount(lab.ravel(),weights=rgb[...,c].ravel(),minlength=len(n))/n.clip(1) for c in range(3)],1)[keep]
        w=np.sqrt(n[keep]);w/=w.sum()*counts[r['video']];target.append(torch.from_numpy(means));weight.append(torch.from_numpy(w))
    t=torch.cat(target);w=torch.cat(weight);w/=w.sum();y=(t[:,0]+2*t[:,1]+t[:,2])/4;co=(t[:,0]-t[:,2])/2;cg=(2*t[:,1]-t[:,0]-t[:,2])/4
    lev=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).double();one=torch.ones(1215,dtype=torch.float64)
    dy=torch.stack((one,lev[:,0]),1);dc=torch.stack((one,lev[:,1]),1)
    dg=torch.stack([one]+[(lev[:,2]-s).clamp(0,e-s) for s,e in ((0,2),(2,3),(3,4),(4,5),(5,6),(6,8))],1)
    py=torch.tensor([16.,16.],dtype=torch.float64);pc=torch.tensor([-36.,9.],dtype=torch.float64);pg=torch.tensor([-24.,6.,6.,6.,6.,6.,6.],dtype=torch.float64)
    records=[];best=float('inf');best_state=None
    for step in range(17):
        _,pal=constants(net);pal+=128;assigned=[]
        for part in t.split(2048):assigned.append((part.square().sum(1)[:,None]+pal.square().sum(1)[None]-2*part@pal.T).argmin(1))
        ids=torch.cat(assigned);err=float(((pal[ids]-t).abs().mean(1)*w).sum());record=dict(step=step,mae=err,unique_rgb=len(torch.unique(pal,dim=0)));records.append(record)
        if err<best:best=err;best_state=copy.deepcopy(net.state_dict());best_step=step
        print('MONOTONE_CHROMA',record,flush=True)
        sy=bounded_fit(dy[ids],y,w,py,py.new_tensor([0,12]),py.new_tensor([32,20]),dy)
        sc=bounded_fit(dc[ids],co,w,pc,pc.new_tensor([-50,6]),pc.new_tensor([-20,12]),dc)
        sg=bounded_fit(dg[ids],cg,w,pg,pg.new_tensor([-40,1,1,1,1,1,1]),pg.new_tensor([-10,12,12,12,12,12,12]),dg)
        solution=rgb_solution(sy,sc,sg);fit_head(net,solution)
        record['slopes']=sg[1:].tolist();record['y']=sy.tolist();record['co']=sc.tolist()
    net.load_state_dict(best_state);raw=net.export();assert len(raw)==20480;_,pal=constants(net);pal=(pal+128).float();pk=tuple(map(tuple,pal.tolist()))
    saved={k:v for k,v in state.items() if k not in ('optimizer','rng','torch_rng','cuda_rng')};saved.update(model=copy.deepcopy(best_state),radix_colour_calibration='monotone_ycocg_hinges',calibration_training_ids=[r['id'] for r in data['training']],
        constant_palette_requires_runtime=True,five_existing_cg_hinges=True,requires_new_code_assignment_training=True,visual_acceptance=False)
    torch.save(saved,out/'initial.pt');(out/'coefficients-padded.bin').write_bytes(raw)
    actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');shots={};checks=[]
    for row in data['validation']:
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
        nn=net(x,all_nodes=True);other,_=actual(x*128,True);assert all(torch.equal(v,other[k]) for k,v in nn.items())
        codes=region_codes(row['target'],row['region'],pk).astype(np.int64);truth=joint_interface_fields(torch.from_numpy(codes)[None]);gt=fragment(net,F.pad(net.round(truth),(0,0,0,0,0,2)),'up1_low_dw')['rgb_head']
        target=m.t.r.read(row['target']);palette=pal.numpy()[codes];decoded=gt[0].permute(1,2,0).numpy()+128
        checks.append(dict(id=row['id'],all_node_error=0,palette_mae=float(np.abs(palette-target).mean()),known_geometry_mae=float(np.abs(decoded-target).mean())))
        for rid,box,title in m.t.r.c.CROPS:
            if rid==row['id']:
                shots[rid,title]=[Image.fromarray(z.astype(np.uint8)).crop(box) for z in (target,palette,decoded)]
    board=Image.new('RGB',(1260,1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,(rid,box,title) in enumerate(m.t.r.c.CROPS):
        for j,(im,label) in enumerate(zip(shots[rid,title],('目标','单调新色表·已知标签','已知共同轮廓·非源预测'))):
            board.paste(im,(420*j,314*i+34));draw.text((420*j+2,314*i+4),title+' | '+label,font=font,fill='white')
    board.save(out/'diagnostic.png')
    write_json(out/'review.json',dict(records=records,selected_step=best_step,selected_mae=best,checks=checks,bytes=20480,source_sha256=file_hash(__file__),parent_sha256=file_hash(a.checkpoint),train_only=True,target_columns_not_source_predictions=True,visual_acceptance=False))


if __name__=='__main__':main()
