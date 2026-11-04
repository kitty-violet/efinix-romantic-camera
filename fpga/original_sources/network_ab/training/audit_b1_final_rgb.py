"""Native image quality and byte-exact execution for unconstrained B1 RGB.

No obsolete fixed palette/latent-label assertions are applied to a decoder
whose ordinary coefficients are being learned. Numerical PASS is not visual
or temporal acceptance.
"""
import argparse
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .audit_b1_colour_fringe_origin import fringe,two_colour_support
from .audit_b1_target_chain import region_means,adjacency,lab
from .audit_b1_rim_tolerance import rim_terms
from .b1_complete_rgb_profiles import complete_rgb_profiles
from .probe_b1_monotone_pair_decoder import make_board
from .runtime import write_json,file_hash,contract_guard
from . import train_b1_minimum_area as m


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--reserved',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2);contract_guard();state,net=load(a.checkpoint);out=Path(a.out);out.mkdir(exist_ok=False)
    raw=net.export();assert len(raw)==20480;rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
    data=m.load_data(a.data);rows=data['validation']+(data['reserved_test'] if a.reserved else [])
    records=[];boards=[{},{}];tensor=lambda z:torch.from_numpy(z.transpose(2,0,1).copy())[None].float()/128-1
    for row in rows:
        for key,sha in (('path','sha256'),('target','target_sha256'),('region','region_sha256')):assert file_hash(row[key])==row[sha]
        src=m.t.r.read(row['path']);target=m.t.r.read(row['target']);labels=m.load_region(row['region']).astype(np.int64)
        x=tensor(src);t=tensor(target);ll=torch.from_numpy(labels)[None];nodes=net(x,all_nodes=True);actual,_=rt(net.round(x*128),True)
        assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
        p=actual['rgb_head']/128;rgb=actual['rgb_head'][0].permute(1,2,0).numpy()+128;d=rgb-target.astype(np.float32)
        rim,_=rim_terms(p,t,ll,2);profile,_=complete_rgb_profiles(p,t,ll)
        means,count=region_means(target,labels);predicted,_=region_means(rgb,labels);ca,cb,valid=two_colour_support(labels,means)
        aa,bb,length=adjacency(labels);wanted=lab(means)[aa]-lab(means)[bb];got=lab(predicted)[aa]-lab(predicted)[bb]
        strong=np.linalg.norm(wanted,axis=1)>=5;den=max(1,length[strong].sum())
        edge=torch.zeros_like(ll,dtype=torch.bool);edge[...,1:]|=ll[...,1:]!=ll[...,:-1];edge[:,1:]|=ll[:,1:]!=ll[:,:-1]
        core=(F.max_pool2d(edge[:,None].float(),5,1,2)==0)[0,0].numpy();roi=[]
        for rid,(x0,y0,x1,y1),title in m.t.r.c.CROPS:
            if rid!=row['id']:continue
            dd=d[y0:y1,x0:x1][core[y0:y1,x0:x1]]
            roi.append(dict(region=title,pixels=len(dd),bias_Co=float(((dd[:,0]-dd[:,2])/2).mean()),bias_Cg=float(((2*dd[:,1]-dd[:,0]-dd[:,2])/4).mean())))
        rec=dict(id=row['id'],split=row['split'],mae=float(np.abs(d).mean()),tight_rim=float(rim),complete_profile=float(profile),
            original_region_colour_fringe=fringe(rgb,ca,cb,valid),roi_chroma_bias=roi,
            collapsed_fraction=float(length[(np.linalg.norm(got,axis=1)<2)&strong].sum()/den),reversed_fraction=float(length[((wanted*got).sum(1)<0)&strong].sum()/den))
        records.append(rec);im=Image.fromarray(rgb.astype(np.uint8));im.save(out/(row['id']+'_integer720.png'))
        boards[0][row['id']]=Image.fromarray(target);boards[1][row['id']]=im
        write_json(out/'review.json',dict(records=records,complete=False,actual_all_nodes_error=0,bytes=len(raw),checkpoint_sha256=file_hash(a.checkpoint),
            source_sha256=file_hash(__file__),data_sha256=file_hash(a.data),fixed_palette_assumed=False,local_fit_only=state.get('local_fit_only',False),
            reserved_previously_used_for_reconstruction=a.reserved,visual_acceptance=False,temporal_acceptance=False))
        print('FINAL_RGB_AUDIT',row['id'],rec['mae'],rec['tight_rim'],rec['complete_profile'],flush=True)
    make_board(boards,['目标','实际20KiB重载'],out/'comparison.png')
    import json
    review=json.loads((out/'review.json').read_text(encoding='utf-8'));review['complete']=True;write_json(out/'review.json',review)


if __name__=='__main__':main()
