"""Whole720 real-source integer review of explicit research checkpoints."""
import argparse,json,shutil
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .b1_geometry_first_objective import geometry_first
from .b1_relative_rgb_profiles import relative_profiles
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',nargs='+',required=True)
    ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);data=m.load_data(a.data);rows=data['validation'];out=Path(a.out);out.mkdir(exist_ok=False)
    images={};records=[];paths=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    for checkpoint in a.checkpoints:
        _,net=load(checkpoint);name=Path(checkpoint).parent.name+'_'+Path(checkpoint).stem;folder=out/name;folder.mkdir();paths.append(folder)
        raw=net.export();assert len(raw)==20480;runtime=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
        rec=[]
        for row in rows:
            src=m.t.r.read(row['path']);target=m.t.r.read(row['target']);x=torch.from_numpy(src.transpose(2,0,1).copy())[None].float()/128-1
            t=torch.from_numpy(target.transpose(2,0,1).copy())[None].float()/128-1;lab=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
            actual,_=runtime(net.round(x*128),True);ours=net(x,all_nodes=True)
            assert all(torch.equal(v,actual[k]) for k,v in ours.items())
            p=actual['rgb_head']/128;loss,terms=geometry_first(p,t,lab);rel,_=relative_profiles(p,t,lab)
            im=m.t.r.c.rgb_image(p[0]);im.save(folder/(row['id']+'_output.png'));images[name,row['id']]=im
            r=dict(id=row['id'],mae=float((p-t).abs().mean()*128),geometry_loss=float(loss),relative_shape=float(rel),all_node_error=0,**{k:float(v) for k,v in terms.items()})
            rec.append(r)
        mean={k:float(np.mean([r[k] for r in rec])) for k in rec[0] if k!='id'}
        records.append(dict(checkpoint=checkpoint,checkpoint_sha256=file_hash(checkpoint),folder=str(folder),records=rec,mean=mean))
        write_json(out/'results.json',dict(models=records,data_sha256=file_hash(a.data),whole720=True,actual_integer=True,visual_acceptance=False))
        print('ACTUAL_CANDIDATE_REVIEW',name,mean,flush=True)
    byid={r['id']:r for r in rows};board=Image.new('RGB',(420*(len(records)+1),314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        opts=[('固定目标',Image.open(byid[key]['target']))]+[(Path(r['folder']).name,images[Path(r['folder']).name,key]) for r in records]
        for j,(name,im) in enumerate(opts):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+name,font=font,fill='white')
    board.save(out/'comparison.png');shutil.copy2(__file__,out/Path(__file__).name)


if __name__=='__main__':main()
