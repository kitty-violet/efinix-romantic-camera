"""Compare source versus stylized within-cell phase targets on TRAIN only.

Image and feature gradients are diagnostics, not a claim that Adam's final
quantized parameter update has the same direction.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .train_b1_trainable_luma_readout import ReadoutB1
from .b1_geometry_first_objective import geometry_first
from .train_b1_contour_coherence import contour_coherence
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--mining',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);net.__class__=ReadoutB1
    heads={}
    for key,ni,no in (('phase2',16,4),('phase4',32,16)):
        h=torch.nn.Conv2d(ni,no,1);h.load_state_dict({k:state['heads'][key+'.'+k] for k in ('weight','bias')});heads[key]=h
    rows={r['id']:r for r in m.load_data(a.data)['training']};mined=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];records=[]
    for item in (mined[1],mined[5],mined[6]):
        row=rows[item['row']['id']];xx,yy=item['xy'];w,h=item['wh'];cut=lambda z:z[...,40:-40,40:-40]
        tensor=lambda k:torch.from_numpy(m.t.r.read(row[k])[yy:yy+h,xx:xx+w].transpose(2,0,1).copy())[None].float()/128-1
        x=tensor('path');t=tensor('target');lab=torch.from_numpy(m.load_region(row['region'])[yy:yy+h,xx:xx+w].astype(np.int64))[None]
        nodes=net(x,all_nodes=True);p=nodes['rgb_head']/128
        mainloss=geometry_first(cut(p),cut(t),cut(lab))[0]+.5*contour_coherence(cut(p),cut(t))
        values={}
        for scale,node,key in ((2,'down1.pw','phase2'),(4,'down2.pw','phase4')):
            feature=nodes[node];primary=torch.autograd.grad(mainloss,feature,retain_graph=True)[0]
            q=heads[key](feature/64-1);q=q-q.mean(1,keepdim=True);margin=40//scale;q=q[...,margin:-margin,margin:-margin]
            gradients={}
            for name,ref in (('source',x),('stylized',t)):
                y=(ref[:,:1]+2*ref[:,1:2]+ref[:,2:3])/4
                target=F.pixel_unshuffle(y,scale);target=target-target.mean(1,keepdim=True);target=target[...,margin:-margin,margin:-margin]
                part=F.smooth_l1_loss(q,target,beta=2/128);g=torch.autograd.grad(part,feature,retain_graph=True)[0]
                gradients[name]=g
                values[key+'_'+name]=dict(loss=float(part.detach()),cosine_to_geometry=float((g*primary).sum()/(g.norm()*primary.norm()).clamp_min(1e-12)),
                    residual_mae_bytes=float((q-target).detach().abs().mean()*128),gradient_norm=float(g.norm()))
            values[key+'_source_vs_stylized_cosine']=float((gradients['source']*gradients['stylized']).sum()/(gradients['source'].norm()*gradients['stylized'].norm()).clamp_min(1e-12))
        records.append(dict(id=row['id'],values=values));print('PHASE_CONFLICT',records[-1],flush=True)
    out=Path(a.out);out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'results.json',dict(records=records,checkpoint_sha256=file_hash(a.checkpoint),existing_heads_trained_on_source=True,
        limitation='Stylized head is not refitted here; head mismatch and feature objective conflict are not separated. TRAIN diagnostics only.'))


if __name__=='__main__':main()
