"""Paired landscape-teacher supervision from existing source-isolated footage."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .runtime import configure_workspace, file_hash, write_json
from .animegan_teacher import AnimeTeacher, MODEL_SHA
from .prepare_coherent_shared_data import contact


def main():
    root=configure_workspace();run=root/'runs/clear-style-control-20261003';out=run/'paired_data';out.mkdir(exist_ok=True)
    master=root/'runs/coherent-shared-20261002/data/manifest.json';data=json.loads(master.read_text(encoding='utf-8'))
    excluded={9:'moving foreground hand',10:'moving foreground hand',11:'hand motion blur',20:'strong exit glare',
              47:'entering blurred hand',51:'near hand gesture/motion',73:'near hand occlusion',80:'large veiling glare',81:'large veiling glare'}
    net=AnimeTeacher(root);cv2.setNumThreads(4);pairs=[];targets={};reviews=[]
    for i,p in enumerate(data['pairs']):
        if i in excluded:continue
        assert data['source_splits'][p['source']]['split']=='train'
        q=dict(p);q['parent_pair_index']=i
        for key in ['current','next']:
            path=Path(p[key]);sha=file_hash(path);assert sha==p[key+'_sha256']
            dst=out/(sha[:20]+'_teacher.png')
            if sha not in targets:
                rgb=np.array(Image.open(path).convert('RGB'));y=net(rgb);Image.fromarray(y).save(dst)
                targets[sha]={'input':str(path),'input_sha256':sha,'target':str(dst),'target_sha256':file_hash(dst)}
            q[key+'_target']=str(dst);q[key+'_target_sha256']=targets[sha]['target_sha256']
        assert file_hash(p['flow'])==p['flow_sha256']
        with np.load(p['flow']) as f:flow=f['backward'].astype(np.float32);mask=f['mask'].astype(bool)
        hh,ww=mask.shape;yy,xx=np.mgrid[:hh,:ww].astype(np.float32)
        a,b,ta,tb=[cv2.resize(np.array(Image.open(q[k])),(ww,hh),interpolation=cv2.INTER_AREA).astype(np.float32)
                   for k in ['current','next','current_target','next_target']]
        delta=(tb-b)-cv2.remap(ta-a,xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_LINEAR)
        q['teacher_style_temporal_mae_rgb255']=float(np.abs(delta)[mask].mean())
        pairs.append(q);reviews.append({'path':q['current_target'],'scene':f'parent {i:03d} '+Path(p['source']).name})
        if len(pairs)%10==0:print('TEACHER_PAIRS',len(pairs),flush=True)
    record={'schema':'clear-style-paired-data-v1','status':'REQUIRES_TEACHER_VISUAL_REVIEW','teacher_sha256':MODEL_SHA,
            'parent_manifest':str(master),'parent_manifest_sha256':file_hash(master),'source_splits':data['source_splits'],
            'pairs':pairs,'unique_teacher_frames':len(targets),'excluded':[{'parent_pair_index':i,'reason':r} for i,r in excluded.items()],
            'new_source_holdout':'VID_20260925_065914.mp4','note':'95 temporal anchors in 7 source videos are not 95 independent scenes; no validation/test source used for training'}
    write_json(out/'manifest.draft.json',record);contact(reviews,out,'teacher_review','path')
    write_json(out/'temporal_rank.json',sorted([{'index':p['parent_pair_index'],'mae':p['teacher_style_temporal_mae_rgb255']} for p in pairs],key=lambda x:-x['mae']))
    print('PREPARED',len(pairs),len(targets),flush=True)


if __name__=='__main__':main()
