"""Replay the full frozen label recipe, then change only outside B1 support.

Counterfactual diagnosis only: these constructed images are not natural-video
accuracy bounds. Original data/labels remain unchanged. Every baseline stage
must reproduce its frozen RGB/labels before remote-context results are valid.
"""
import argparse,json,shutil,time
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .probe_b1_encoder_gauge import load as load_b1
from .audit_b1_spatial_target_scope import dependency
from .publish_global_blocks import load
from .publish_coarse_people import probability
from .coarse_people import refined_mask,composite
from .coarse_people_blocks import person_surface
from .gentle_block_boundaries import process
from .refine_block_separation import PARAMETERS
from .minimum_area_blocks import make_target,boundary_mask
from .refine_b1_hierarchical_targets import refine
from .runtime import configure_workspace,file_hash,write_json


def aa(rgb,labels):
    edge=cv2.dilate(boundary_mask(labels),np.ones((3,3),np.uint8))
    return np.where(edge[...,None],cv2.GaussianBlur(rgb.astype(np.float32),(3,3),.65),rgb).round().clip(0,255).astype(np.uint8)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','mining','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();configure_workspace();torch.set_num_threads(2);cv2.setNumThreads(2)
    out=Path(a.out);out.mkdir(exist_ok=False)
    d=m.load_data(a.data);base=json.loads((m.OUT.parent/'b1-style-recovery-20261005/data.json').read_text('utf-8'))
    files=dict(base['recipe_bindings'])
    for name in ('minimum_area_blocks.py','refine_b1_hierarchical_targets.py','add_b1_region_boundary_coverage.py',Path(__file__).name):
        p=Path(__file__).with_name(name);files[str(p)]=file_hash(p)
    assert all(file_hash(p)==h for p,h in files.items())
    snap=out/'source';snap.mkdir()
    for i,p in enumerate(files):shutil.copy2(p,snap/(f'{i:02d}_'+Path(p).name))
    cfg,teacher,b,direction,paint,adapter,clip,state,seg=load()
    _,student=load_b1(a.checkpoint);student.cuda();span=dependency(student.plan['nodes'])
    byid={r['id']:r for r in d['training']};review=json.loads(Path(a.mining).read_text('utf-8'))['records'];specs=[review[k] for k in (1,5,6)]
    def generate(x):
        pred=np.array(m.t.r.c.rgb_image(teacher(m.t.r.tensor(x)[None])[0]))
        surface=direction.surface(paint.tone(pred,adapter),direction.PRESETS['direction4'])
        background=b.transform_surface(surface,'balanced')
        mask=refined_mask(probability(seg,x),x)
        people=person_surface(pred,adapter,paint,'strongblocks',x)
        parent=process(composite(background,people,mask),PARAMETERS)
        initial,labels,_=make_target(parent,256)
        hard,labels2,_=refine(parent,labels)
        return dict(neural=pred,parent=parent,minimum=initial,hard=hard,target=aa(hard,labels2)),labels2
    records=[];baseline=[];started=time.monotonic();font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,spec in enumerate(specs):
        row=byid[spec['row']['id']];x=m.t.r.read(row['path']);stages,label=generate(x)
        checks={k:int(np.abs(stages[k].astype(np.int16)-m.t.r.read(row[path]).astype(np.int16)).max())
                for k,path in (('parent','parent_target'),('minimum','previous_target'),('hard','hard_region_target'),('target','target'))}
        checks['region']=int(np.max(np.abs(label.astype(np.int32)-m.load_region(row['region']).astype(np.int32))))
        write_json(out/(row['id']+'_replay.json'),checks);assert max(checks.values())==0,checks
        baseline.append(dict(id=row['id'],checks=checks));print('FULL_TARGET_REPLAY',row['id'],checks,flush=True)
        p=student(m.t.r.tensor(x)[None])[0]
        donor=m.t.r.read(byid[specs[(i+1)%len(specs)]['row']['id']]['path'])
        left,top=spec['xy']
        for offset in (64,160,256):
            cx,cy=left+offset,top+160;half=32
            xa,xb=span('rgb_head',cx-half)[0],span('rgb_head',cx+half-1)[1]+1
            ya,yb=span('rgb_head',cy-half)[0],span('rgb_head',cy+half-1)[1]+1
            yy,xx=np.indices(x.shape[:2]);distance=np.maximum.reduce((xa-xx,xx-(xb-1),ya-yy,yy-(yb-1),np.zeros_like(xx)))
            alpha=np.clip(distance/64.,0,1)[...,None]
            altered=np.round(x*(1-alpha)+donor*alpha).astype(np.uint8)
            assert np.array_equal(altered[ya:yb,xa:xb],x[ya:yb,xa:xb])
            q=student(m.t.r.tensor(altered)[None])[0]
            ys=slice(cy-half,cy+half);xs=slice(cx-half,cx+half)
            e=float((p[...,ys,xs]-q[...,ys,xs]).abs().max()*128);assert e==0
            changed,other_label=generate(altered)
            stats={}
            for name in stages:
                delta=np.abs(stages[name][ys,xs].astype(float)-changed[name][ys,xs].astype(float))
                stats[name]=dict(mae=float(delta.mean()),p95=float(np.percentile(delta,95)),max=float(delta.max()),pixel_over8=float((delta.max(2)>8).mean()))
            rec=dict(id=row['id'],centre=[cx,cy],preserved_box=[xa,ya,xb,yb],b1_rgb_error=e,stages=stats)
            records.append(rec);write_json(out/'partial.json',dict(baseline=baseline,records=records))
            board=Image.new('RGB',(640,152),'#17202a');draw=ImageDraw.Draw(board)
            choices=[('局部输入完全相同',x),('原完整目标',stages['target']),('只改远场后的目标',changed['target']),('原神经教师',stages['neural']),('只改远场后的神经教师',changed['neural'])]
            for j,(title,ar) in enumerate(choices):
                board.paste(Image.fromarray(ar[ys,xs]).resize((128,128),Image.Resampling.NEAREST),(128*j,24));draw.text((128*j+2,2),title,font=font,fill='white')
            board.save(out/f'context_{i}_{offset}.png')
            print('FULL_TARGET_REMOTE',i,offset,stats,'student0',flush=True)
    assert all(file_hash(p)==h for p,h in files.items())
    write_json(out/'results.json',dict(baseline=baseline,records=records,seconds=time.monotonic()-started,code=files,
        checkpoint_sha256=file_hash(a.checkpoint),data_sha256=file_hash(a.data),target_modified=False,
        scope='Constructed counterfactuals with 64-pixel feathered remote context. Not natural-distribution error bound or impossibility proof.'))


if __name__=='__main__':main()
