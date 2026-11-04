"""Build native720 corrected-teacher targets from source-isolated raw videos."""
from collections import defaultdict
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from .runtime import configure_workspace, file_hash, write_json, contract_guard
from .media import decode_frames
from .fpga_video_resize import area_3to2, verify
from .trainable_animegan import TrainableAnimeGAN
from .smooth_surface_guard import correct_surface, CONFIG
from .prepare_coherent_shared_data import pair_flow, contact
from .train_clear_style import tensor

TEACHER_SHA = 'c86b9b99e901c0ad6be7ce74bd35388a5028dc8c12228c579489025fa6b2f7a5'


def rgb_output(net, rgb):
    with torch.inference_mode():
        y = net(tensor(rgb))
        # Preserve accepted renderer's truncation before fixed correction.
        raw = (y[0].permute(1,2,0).cpu().numpy()*128+128).clip(0,255).astype(np.uint8)
    return correct_surface(rgb, raw)


def main():
    root = configure_workspace(); contract_guard()
    run = root/'runs/b1-distillation-20261003'; out = run/'data'; out.mkdir(parents=True,exist_ok=False)
    write_json(run/'resize_verification.json', verify())
    torch.set_num_threads(4); cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    teacher = root/'runs/texture-recovery-20261003/final/refined_style_fp32.pt'
    assert file_hash(teacher)==TEACHER_SHA
    net = TrainableAnimeGAN(root).cuda().eval()
    net.load_state_dict(torch.load(teacher,map_location='cpu',weights_only=True)['generator'])
    parent = root/'runs/clear-style-control-20261003/paired_data/manifest.json'
    data = json.loads(parent.read_text(encoding='utf-8'))
    groups = defaultdict(list)
    for p in data['pairs']:
        assert data['source_splits'][p['source']]['split']=='train'
        groups[p['source']].append(p)
    pairs = []; images = []
    for source, rows in sorted(groups.items()):
        assert file_hash(source)==rows[0]['source_sha256']
        indices=sorted({p[k] for p in rows for k in ['current_index','next_index']})
        saved={}; print('SOURCE',Path(source).name,len(indices),flush=True)
        for index,pts,rgb in decode_frames(source,indices=indices,hlg_exposure_stops=-1.):
            assert rgb.shape==(1080,1920,3)
            x=area_3to2(rgb); target=rgb_output(net,x)
            stem=rows[0]['source_sha256'][:12]+f'_{index:06d}'
            ip=out/(stem+'_source.png');tp=out/(stem+'_target.png')
            Image.fromarray(x).save(ip);Image.fromarray(target).save(tp)
            saved[index]={'source':str(ip),'target':str(tp),'source_sha256':file_hash(ip),'target_sha256':file_hash(tp),'pts':pts}
            images.append({'path':str(tp),'scene':Path(source).stem+f' f{index}'})
        for p in rows:
            q={k:p[k] for k in ['source','source_sha256','current_index','next_index','parent_pair_index']}
            for k,idx in [('current',p['current_index']),('next',p['next_index'])]:
                for key,suffix in [('source',''),('target','_target')]:
                    q[k+suffix]=saved[idx][key];q[k+suffix+'_sha256']=saved[idx][key+'_sha256']
                q[k+'_pts']=saved[idx]['pts']
            a,b=[np.array(Image.open(q[k])) for k in ['current','next']]
            flow,mask=pair_flow(a,b);fp=out/(Path(q['current']).stem+'_flow.npz')
            np.savez_compressed(fp,backward=flow,mask=mask)
            q.update(flow=str(fp),flow_sha256=file_hash(fp),valid_fraction=float(mask.mean()),split='train')
            pairs.append(q)
        write_json(out/'preparation_progress.json',{'pairs':len(pairs),'source_complete':source})
    contact(images,out,'targets','path')
    # Previously viewed source-isolated regressions, explicitly not blind test.
    clips=json.loads((root/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips']
    validation=[]; reviews=[]
    for clip in clips:
        assert clip['source'] not in groups and file_hash(clip['clip'])==clip['clip_sha256']
        cap=cv2.VideoCapture(clip['clip']);wanted=[120,210] if clip['id']=='new_hallway' else [90,210]
        for index in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,bgr=cap.read();assert ok
            x=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));target=rgb_output(net,x)
            prefix=out/f'val_{clip["id"]}_{index:03d}'
            ip=prefix.with_name(prefix.name+'_source.png');tp=prefix.with_name(prefix.name+'_target.png')
            Image.fromarray(x).save(ip);Image.fromarray(target).save(tp)
            validation.append(dict(id=prefix.name,source=str(ip),target=str(tp),source_sha256=file_hash(ip),target_sha256=file_hash(tp),
                                   raw_source=clip['source'],clip=clip['clip'],frame=index,role='reviewed_regression_not_blind'))
            reviews.extend([{'path':str(ip),'scene':prefix.name+' input'},{'path':str(tp),'scene':prefix.name+' teacher'}])
        cap.release()
    contact(reviews,out,'validation','path',per_page=12)
    manifest={'schema':'b1-corrected-teacher-pairs-v1','status':'PREPARED_REQUIRES_REVIEW','pairs':pairs,'validation':validation,
              'source_splits':data['source_splits'],'teacher':str(teacher),'teacher_sha256':TEACHER_SHA,
              'guard':CONFIG,'parent_manifest':str(parent),'parent_sha256':file_hash(parent),
              'preprocess':'native1080 fixed HLG SDR -1EV -> exact integer FPGA area720',
              'target_policy':'whole-frame720 accepted teacher then fixed smooth surface correction, before any crop',
              'source_binding':{str(p):file_hash(p) for p in [Path(__file__),Path(__file__).with_name('fpga_video_resize.py'),Path(__file__).with_name('smooth_surface_guard.py'),Path(__file__).with_name('trainable_animegan.py')]}}
    write_json(out/'manifest.json',manifest)
    print('DATA_READY',len(pairs),'pairs',len(validation),'regressions',flush=True)


if __name__=='__main__':main()
