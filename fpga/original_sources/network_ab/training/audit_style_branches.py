"""Causal path interventions, separate style/structure/temporal measurements.

Gates are diagnostic only, not advertised as physically pruned networks.
Unmodified historical training code/checkpoints remain immutable.
"""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
import torch
from .importance_pruned_teacher import build_group_pruned, conv_name
from .prune_animation_teacher import TEACHER_SHA
from .trainable_animegan import TrainableAnimeGAN
from .train_b1_distillation import as_tensor, to_rgb, read_image
from .fpga_video_resize import area_3to2
from .smooth_surface_guard import correct_surface
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .engine import set_reproducible

RUN='branch-function-audit-20261004'
REPAIR='style-retention-repair-20261003/repair/pruned_teacher_fp32.pt'
REPAIR_SHA='d36389b595e52d216871fb3792f70f9e3c91343dbc38e71ee8d7b3176b438599'


def load(root,kind):
    path=root/'runs'/(REPAIR if kind=='repair' else 'texture-recovery-20261003/final/refined_style_fp32.pt')
    assert file_hash(path)==(REPAIR_SHA if kind=='repair' else TEACHER_SHA)
    state=torch.load(path,map_location='cpu',weights_only=True)
    net=build_group_pruned(root,state['indices']) if kind=='repair' else TrainableAnimeGAN(root)
    net.load_state_dict(state['generator']);return net.cuda().eval()


def paths():
    result={f'{branch}_lade{i}':dict(kind='statistics',module=conv_name(branch,2*i+1),block=i)
            for branch,n in [('base',7),('main',6)] for i in range(n)}
    for name,suffix in [('skip180','3'),('skip360','10'),('skip720','17')]:
        result[name]=dict(kind='merge',module='generator.generator_1/main/add_'+suffix,operand=1)
    result['attention']=dict(kind='merge',module='generator.generator_1/main/External_attention/add_1',operand=0)
    return result


def intervene(net,name,alpha):
    spec=paths()[name];m=net.get_submodule(spec['module'])
    if spec['kind']=='merge':
        def before(module,args):
            if alpha==1:return args
            values=list(args);i=spec['operand'];values[i]=values[i]*alpha;return tuple(values)
        return m.register_forward_pre_hook(before)
    def after(module,args,output):
        # LADE's learned 1x1 statistics path consumes its main convolution.
        # alpha=0 uses that feature's own mean/std, hence bypasses learned
        # statistical remapping without zeroing the entire feature tensor.
        assert args[0].shape==output.shape
        if alpha==1:return output
        return output*alpha+args[0]*(1-alpha)
    return m.register_forward_hook(after)


def predict(net,source,kind):
    with torch.inference_mode():y=to_rgb(net(as_tensor(source)[None].cuda())[0])
    return correct_surface(source,y) if kind=='teacher' else y


def gray(x):return cv2.cvtColor(x,cv2.COLOR_RGB2GRAY).astype(np.float32)
def hp(x):return x-cv2.blur(x,(9,9),borderType=cv2.BORDER_REPLICATE)
def grad(x):
    return np.hypot(cv2.Sobel(x,cv2.CV_32F,1,0,ksize=3)/8,cv2.Sobel(x,cv2.CV_32F,0,1,ksize=3)/8)


def spatial(p,t,s):
    p,t,s=[x[8:-8,8:-8] for x in (p,t,s)]
    gp,gt,gs=map(gray,(p,t,s));bp,bt,bs=map(hp,(gp,gt,gs))
    ep,et=grad(gp),grad(gt);smooth=(np.abs(bt)<3)&(et<2)
    structural=et>5;wanted=bt-bs;actual=bp-bs
    return dict(rgb_mae=float(np.abs(p.astype(float)-t).mean()),
        fine_error=float(np.abs(bp-bt).mean()),
        excess_detail=float(np.maximum(np.abs(bp)-np.abs(bt),0)[smooth].mean()),
        false_edge_fraction=float((ep[smooth]>5).mean()),
        structural_edge_error=float(np.abs(ep-et)[structural].mean()),
        structural_edge_ratio=float(ep[structural].mean()/et[structural].mean()),
        change_projection=float((actual*wanted).sum()/(wanted.square().sum() if hasattr(wanted,'square') else (wanted*wanted).sum()+1e-9)),
        source_change=float(np.abs(p.astype(float)-s).mean()))


def motion(a,b):
    ga,gb=gray(a),gray(b);aa=cv2.resize(ga,(640,360));bb=cv2.resize(gb,(640,360))
    back=cv2.calcOpticalFlowFarneback(bb,aa,None,.5,3,21,3,5,1.2,0)
    forward=cv2.calcOpticalFlowFarneback(aa,bb,None,.5,3,21,3,5,1.2,0)
    f=cv2.resize(back,(1280,720))*2;fw=cv2.resize(forward,(1280,720))*2
    yy,xx=np.mgrid[:720,:1280].astype(np.float32);mx,my=xx+f[:,:,0],yy+f[:,:,1]
    sample=lambda x:cv2.remap(x,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    valid=(np.abs(gb-sample(ga))<8)&(np.linalg.norm(f+sample(fw),axis=2)<1.5)&(mx>=8)&(mx<1272)&(my>=8)&(my<712)
    assert valid.mean()>.05
    return mx,my,valid


def prepare(root,run):
    run.mkdir(exist_ok=False);(run/'assets').mkdir()
    manifest=json.loads((root/'runs/style-retention-repair-20261003/data_manifest.json').read_text(encoding='utf-8'))
    groups={}
    for row in manifest['pairs']:groups.setdefault(row['source'],[]).append(row)
    rows=[]
    for i,source in enumerate(sorted(groups)):
        row=groups[source][len(groups[source])//2]
        for side in ['current','next']:
            rows.append(dict(id=f'train{i}_{side}',source=row[side],target=row[side+'_target'],split='train',origin=source,index=row[side+'_index']))
    rows.extend(dict(id=r['id'],source=r['source'],target=r['target'],split='seen_regression') for r in manifest['validation'])
    clips=json.loads((root/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips']
    teacher=load(root,'teacher')
    for clip in clips:
        assert file_hash(clip['clip'])==clip['clip_sha256']
        cap=cv2.VideoCapture(clip['clip']);start=330 if clip['id']=='plaza' else 90
        cap.set(cv2.CAP_PROP_POS_FRAMES,start)
        for j in range(12):
            assert abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(start+j))<.1
            ok,bgr=cap.read();assert ok
            s=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));t=predict(teacher,s,'teacher')
            name=f"seq_{clip['id']}_{start+j:03d}";sp=run/'assets'/(name+'_s.png');tp=run/'assets'/(name+'_t.png')
            Image.fromarray(s).save(sp);Image.fromarray(t).save(tp)
            rows.append(dict(id=name,source=str(sp),target=str(tp),split='continuous_diagnostic',sequence=clip['id'],index=start+j,origin=clip['clip'],origin_sha256=clip['clip_sha256']))
        cap.release()
    del teacher;torch.cuda.empty_cache()
    for row in rows:
        row['source_sha256']=file_hash(row['source']);row['target_sha256']=file_hash(row['target'])
    write_json(run/'assets.json',dict(rows=rows,sequence_scope='Three 12-frame windows, 33 genuinely adjacent 30fps pairs; diagnosis only, not full-video acceptance'))
    write_json(run/'protocol.json',dict(paths=paths(),levels=[1,.75,0],teacher_guard='Same accepted fixed R48 correction for all teacher interventions; raw student has none',
        metrics='Separate style/detail, major-edge preservation, and motion-compensated teacher-relative temporal error. No scalar weighted ranking.',
        source=str(Path(__file__)),source_sha256=file_hash(__file__),repair_sha256=REPAIR_SHA,teacher_sha256=TEACHER_SHA))
    print('PREPARED',len(rows),flush=True)


def audit(root,run,kind):
    protocol=json.loads((run/'protocol.json').read_text());assert file_hash(__file__)==protocol['source_sha256']
    rows=json.loads((run/'assets.json').read_text())['rows'];assets=[];flows={};previous={}
    for row in rows:
        assert file_hash(row['source'])==row['source_sha256'] and file_hash(row['target'])==row['target_sha256']
        s,t=read_image(row['source']),read_image(row['target']);assets.append((row,s,t))
        if row.get('sequence'):
            seq=row['sequence']
            if seq in previous:
                last,a=previous[seq];assert row['index']==last['index']+1;flows[row['id']]=motion(a,s)
            previous[seq]=(row,s)
    net=load(root,kind);out=run/kind;out.mkdir(exist_ok=False)
    # Diagnostic instrumentation must be exactly transparent at alpha=1.
    checks=[]
    for row,s,t in assets[:2]+assets[14:16]:
        base=predict(net,s,kind)
        handles=[intervene(net,k,1) for k in paths()]
        again=predict(net,s,kind)
        for h in handles:h.remove()
        error=int(np.abs(base.astype(int)-again.astype(int)).max());assert error==0;checks.append(error)
    write_json(out/'preflight.json',dict(status='PASS',all17_hooks_identity_pixel_errors=checks,fp32=all(p.dtype==torch.float32 for p in net.parameters())))
    variants=[('baseline',None,1)]+[(f'{k}_{a:g}',k,a) for k in paths() for a in (.75,0)]
    records=[]
    for name,key,alpha in variants:
        before=time.perf_counter();handle=intervene(net,key,alpha) if key else None
        folder=out/name;folder.mkdir();scores=[];seqpast={};temporals=[]
        for row,s,t in assets:
            p=predict(net,s,kind);scores.append(dict(id=row['id'],split=row['split'],**spatial(p,t,s)))
            if row['split']=='seen_regression':Image.fromarray(p).save(folder/(row['id']+'.png'))
            seq=row.get('sequence')
            if seq:
                effect=gray(p)-gray(s);error=gray(p)-gray(t)
                if seq in seqpast:
                    old_effect,old_error=seqpast[seq];mx,my,mask=flows[row['id']]
                    warp=lambda v:cv2.remap(v,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                    temporals.append(dict(id=row['id'],sequence=seq,valid_fraction=float(mask.mean()),
                        effect_residual=float(np.abs(effect-warp(old_effect))[mask].mean()),
                        teacher_relative_residual=float(np.abs(error-warp(old_error))[mask].mean())))
                seqpast[seq]=(effect,error)
        if handle:handle.remove()
        means={split:{k:float(np.mean([x[k] for x in scores if x['split']==split])) for k in scores[0] if k not in ('id','split')} for split in sorted(set(x['split'] for x in scores))}
        tm={seq:{k:float(np.mean([x[k] for x in temporals if x['sequence']==seq])) for k in ('effect_residual','teacher_relative_residual','valid_fraction')} for seq in seqpast}
        result=dict(name=name,path=key,alpha=alpha,means=means,temporal=tm,seconds=time.perf_counter()-before)
        write_json(folder/'measurements.json',dict(summary=result,spatial=scores,adjacent_pairs=temporals));records.append(result)
        write_json(out/'summary.json',dict(status='COMPLETE' if len(records)==len(variants) else 'RUNNING',records=records))
        print('AUDIT',kind,name,round(result['seconds'],1),json.dumps(means['train']),flush=True)
    print('AUDIT_COMPLETE',kind,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','audit']);ap.add_argument('--model',choices=['repair','teacher']);args=ap.parse_args()
    root=configure_workspace();contract_guard();set_reproducible(2026100401);torch.set_num_threads(4);cv2.setNumThreads(2);run=root/'runs'/RUN
    if args.action=='prepare':prepare(root,run)
    else:audit(root,run,args.model)


if __name__=='__main__':main()
