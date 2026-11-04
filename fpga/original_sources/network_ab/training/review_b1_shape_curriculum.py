"""Hold-out procedural geometry and full-frame natural review, no updates."""
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .train_b1_shape_curriculum_native import shape_batch,shape_loss
from .b1_complete_rgb_profiles import complete_rgb_profiles
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--base',required=True)
    ap.add_argument('--data',required=True);ap.add_argument('--mining',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(99117)
    state,net=load(a.checkpoint);_,base=load(a.base);dest=Path(a.out);dest.mkdir(exist_ok=False)
    setup=state['synthetic_shape_curriculum'];models={'before':base,'after':net}
    results={};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    source,target,labels,specs=shape_batch(setup['colours'],setup['immutable_warm_anchors'],random.Random(8717),16)
    predictions={};cut=lambda v:v[...,40:-40,40:-40]
    for name,n in models.items():
        predictions[name]=[];res=[]
        for i in range(len(source)):
            x=source[i:i+1];nodes=n(x,all_nodes=True);raw=n.export()
            actual,_=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')(n.round(x*128),True)
            assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            p=actual['rgb_head']/128;loss,parts=shape_loss(cut(p),cut(target[i:i+1]),cut(labels[i:i+1]))
            res.append(dict(spec=specs[i],score=float(loss),mae=float((cut(p)-cut(target[i:i+1])).abs().mean()*128),**{k:float(v) for k,v in parts.items()}))
            predictions[name].append(p[0])
        results[name]=res
    selected=[]
    for kind in range(5):
        found=[i for i,s in enumerate(specs) if s['kind']==kind]
        if found:selected.append(found[0])
    board=Image.new('RGB',(4*256,len(selected)*218),'#17202a');draw=ImageDraw.Draw(board)
    for j,i in enumerate(selected):
        for k,(name,z) in enumerate(zip(('合成输入','固定几何目标','训练前实际B1','训练后实际B1'),(source[i],target[i],predictions['before'][i],predictions['after'][i]))):
            board.paste(m.t.r.c.rgb_image(z),(256*k,j*218+26));draw.text((256*k+3,j*218+2),name,font=font,fill='white')
    board.save(dest/'synthetic_comparison.png')
    rows={r['id']:r for r in m.load_data(a.data)['training']};mined=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records']
    natural=[];board=Image.new('RGB',(4*320,3*348),'#17202a');draw=ImageDraw.Draw(board)
    for j,s in enumerate([mined[i] for i in (1,5,6)]):
        row=rows[s['row']['id']];source_np=m.t.r.read(row['path']);target_np=m.t.r.read(row['target'])
        x=torch.from_numpy(source_np.transpose(2,0,1).copy())[None].float()/128-1
        xx,yy=s['xy'];w,h=s['wh'];box=(xx,yy,xx+w,yy+h);ims=[Image.fromarray(source_np).crop(box),Image.fromarray(target_np).crop(box)];values={}
        for name,n in models.items():
            p=n(x)[0];im=m.t.r.c.rgb_image(p[0]);ims.append(im.crop(box))
            t=torch.from_numpy(target_np.transpose(2,0,1).copy())[None].float()/128-1
            lab=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
            profile,_=complete_rgb_profiles(p,t,lab)
            values[name]=dict(mae=float((p-t).abs().mean()*128),profile=float(profile))
        natural.append(dict(id=row['id'],values=values,full720_inference=True,training_scene=True))
        for k,(name,im) in enumerate(zip(('原片','固定目标','训练前实际B1','训练后实际B1'),ims)):
            board.paste(im,(320*k,j*348+28));draw.text((320*k+3,j*348+3),name,font=font,fill='white')
    board.save(dest/'natural_comparison.png');Path(dest/'executed_review.py').write_bytes(Path(__file__).read_bytes())
    write_json(dest/'review.json',dict(synthetic=results,natural=natural,checkpoint_sha256=file_hash(a.checkpoint),base_sha256=file_hash(a.base),source_sha256=file_hash(__file__),synthetic_test_seed=8717,synthetic_all_nodes_error=0,natural_images_are_train=True,visual_acceptance=False))
    print('SHAPE_REVIEW_PASS',dict(synthetic_means={name:float(np.mean([v['score'] for v in r])) for name,r in results.items()},natural=natural),flush=True)


if __name__=='__main__':main()
