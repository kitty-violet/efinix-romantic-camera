"""Render the approved balanced still recipe over all 450 frames of plaza."""
import argparse,ast,importlib,json,shutil,time
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from .block_video_runtime import load_frozen,local_contrast_exact,STILLS
from .stroke_color_input_net import StrokeColorInputNet
from .train_native_contours import tensor,load_rgb
from .train_b1_distillation import to_rgb
from .runtime import configure_workspace,file_hash,write_json
from .engine import set_reproducible
from .fpga_video_resize import area_3to2,bilinear_2to3
from .publish_moderate_contours import encode
from .render_wavelet_game import close_pipe,video_check
from .publish_region_classes import motion_map

OUT=STILLS.parent/'block-separation-video-20261004-v1'


def setup():
    workspace=configure_workspace();set_reproducible(2026100492)
    torch.set_num_threads(2);cv2.setNumThreads(2)
    cfg=json.loads((STILLS/'experiment.json').read_text(encoding='utf-8'))
    for n,h in cfg['code'].items():assert file_hash(STILLS/'source_code'/n)==h
    assert file_hash(cfg['neural_weight_source'])==cfg['neural_weight_sha256']
    state=torch.load(cfg['neural_weight_source'],map_location='cpu',weights_only=True)
    assert all(file_hash(p)==h for p,h in state['binding']['code'].items())
    net=StrokeColorInputNet(False).cuda().eval();net.load_state_dict(state['model'])
    b,d,p=load_frozen();b.local_contrast=local_contrast_exact
    adapter=np.load(STILLS/'color_adapter.npz')['lab_chroma_affine']
    clips=json.loads((workspace/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips']
    clip=next(c for c in clips if c['id']=='plaza');assert file_hash(clip['clip'])==clip['clip_sha256']
    return cfg,net,b,d,p,adapter,clip,state


def pipeline(rgb,net,b,d,p,adapter):
    source=area_3to2(rgb);pred=to_rgb(net(tensor(source)[None].cuda())[0])
    base=d.surface(p.tone(pred,adapter),d.PRESETS['direction4'])
    current=b.transform_surface(base,'balanced');previous=d.render(base,d.PRESETS['direction4'])[0]
    return source,pred,previous,current


@torch.no_grad()
def preflight():
    cfg,net,b,d,p,adapter,clip,state=setup();OUT.mkdir(exist_ok=False)
    files={str(STILLS/'source_code'/n):h for n,h in cfg['code'].items()}
    for n in ['block_video_runtime.py',Path(__file__).name,'fpga_video_resize.py','publish_moderate_contours.py','render_wavelet_game.py','publish_region_classes.py','engine.py']:
        path=Path(__file__).with_name(n);ast.parse(path.read_text(encoding='utf-8'));files[str(path)]=file_hash(path)
    files.update(state['binding']['code']);snapshot=OUT/'source_code';snapshot.mkdir()
    for i,(path,sha) in enumerate(files.items()):
        dest=snapshot/(str(i).zfill(2)+'_'+Path(path).name);shutil.copy2(path,dest);assert file_hash(dest)==sha
    # All six static cases must stay byte-exact after computational skipping.
    exact=[]
    for row in cfg['inputs']:
        key=row['id'];base=load_rgb(STILLS/(key+'_surface.png'));result=b.transform_surface(base,'balanced')
        error=int(np.abs(result.astype(int)-load_rgb(STILLS/(key+'_balanced.png')).astype(int)).max());assert error==0
        exact.append(dict(id=key,max_error=error))
    cap=cv2.VideoCapture(clip['clip'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);selected={0,90,210};measure=[]
    for n in range(211):
        ok,bgr=cap.read();assert ok
        if n not in selected:continue
        start=time.perf_counter();source,pred,old,current=pipeline(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),net,b,d,p,adapter)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        if n in (90,210):
            key=f'val_plaza_{n:03d}';assert np.array_equal(current,load_rgb(STILLS/(key+'_balanced.png')))
            assert np.array_equal(pred,load_rgb(Path(cfg['previous_package'])/(key+'_student.png')))
        Image.fromarray(current).save(OUT/f'preflight_{n:03d}.png')
        measure.append(dict(frame=n,seconds=elapsed));print('BLOCK_PREFLIGHT',n,round(elapsed,3),flush=True)
    cap.release()
    write_json(OUT/'pipeline.json',dict(still_package=str(STILLS),preset='balanced',frames=450,fps=30,seconds=15,
        source_clip={k:v for k,v in clip.items() if k!='frames'},weight=cfg['neural_weight_source'],weight_sha256=cfg['neural_weight_sha256'],
        live_code_hashes=files,adapter=str(STILLS/'color_adapter.npz'),adapter_sha256=file_hash(STILLS/'color_adapter.npz'),
        optimization='skip mathematically zero anchor contributions only; six still outputs exact',
        chain='1920x1080 SDR -> integer area1280x720 -> real FP32 student -> fixed tone -> direction4 surface -> balanced block separation -> integer bilinear1920x1080 -> H264 CRF16',
        temporal_blending=False,frame_interpolation=False,teacher_inference=False,new_training=False,board_validated=False))
    write_json(OUT/'preflight.json',dict(six_still_replay=exact,actual_pipeline_timing=measure,matched_native_frames=[90,210],AST_pass=True))
    write_json(OUT/'STATUS.json',dict(status='READY_TO_RENDER',frames=0,total=450))
    print('BLOCK_PREFLIGHT_DONE',OUT,flush=True)


def verify_frozen(cfg):
    assert all(file_hash(p)==h for p,h in cfg['live_code_hashes'].items())
    assert file_hash(cfg['adapter'])==cfg['adapter_sha256']


@torch.no_grad()
def render():
    frozen=json.loads((OUT/'pipeline.json').read_text(encoding='utf-8'));verify_frozen(frozen)
    cfg,net,b,d,p,adapter,clip,_=setup()
    current_path=OUT/'plaza_balanced_15s_1080p.mp4';previous_path=OUT/'plaza_previous_15s_1080p.mp4'
    assert not current_path.exists() and not previous_path.exists()
    cap=cv2.VideoCapture(clip['clip'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
    pipes=[encode(current_path),encode(previous_path)];previous=None;motion=[];dups=0;n=0;start=time.perf_counter();matched=[]
    write_json(OUT/'STATUS.json',dict(status='RENDERING',frames=0,total=450))
    try:
        while n<450:
            ok,bgr=cap.read();assert ok
            source,pred,old,current=pipeline(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),net,b,d,p,adapter)
            output=bilinear_2to3(current);pipes[0].stdin.write(output.tobytes());pipes[1].stdin.write(bilinear_2to3(old).tobytes())
            if n in (0,90,210,330,449):
                Image.fromarray(current).save(OUT/f'frame_{n:03d}_native720.png')
                if n==90:Image.fromarray(output).save(OUT/'poster_1080p.png')
            if n in (90,210):
                assert np.array_equal(current,load_rgb(STILLS/(f'val_plaza_{n:03d}_balanced.png')));matched.append(n)
            if previous is not None:
                mx,my,valid=motion_map(previous[0],source);record=dict(frame=n,valid_fraction=float(valid.mean()))
                for label,a,q in [('original',previous[0],source),('previous',previous[1],old),('current',previous[2],current)]:
                    warp=cv2.remap(a.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                    err=np.abs(q.astype(np.float32)-warp).mean(2);v=err[valid]
                    record[label]=dict(mean=float(v.mean()),p95=float(np.percentile(v,95)),fraction_over12=float((v>12).mean()))
                motion.append(record);dups+=int(np.array_equal(previous[2],current))
            previous=(source,old,current);n+=1
            if n%10==0:
                seconds=time.perf_counter()-start
                write_json(OUT/'STATUS.json',dict(status='RENDERING',frames=n,total=450,elapsed_seconds=seconds,estimated_remaining_seconds=seconds/n*(450-n)))
                print('BLOCK_VIDEO',n,round(seconds,1),flush=True)
    except BaseException as exc:
        write_json(OUT/'STATUS.json',dict(status='INCOMPLETE',frames=n,total=450,error=repr(exc)));raise
    finally:
        cap.release()
        for pipe in pipes:close_pipe(pipe)
    assert n==450 and matched==[90,210] and dups==0;verify_frozen(frozen)
    videos={k:video_check(path,450) for k,path in [('current',current_path),('previous',previous_path)]}
    report=dict(videos=videos,generated_frames=n,identical_adjacent_native_frames=dups,matched_still_frames=matched,
        elapsed_seconds=time.perf_counter()-start,motion=motion,
        motion_mean={k:{m:float(np.mean([r[k][m] for r in motion])) for m in ['mean','p95','fraction_over12']} for k in ['original','previous','current']},
        temporal_acceptance=False,note='Source-flow residual is a diagnostic; visual playback is needed. No temporal blending or duplicated frame padding.')
    write_json(OUT/'video_verification.json',report);write_json(OUT/'STATUS.json',dict(status='RENDER_COMPLETE_PENDING_VIEW',frames=450,total=450))
    print('BLOCK_VIDEO_DONE',OUT,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','render']);args=parser.parse_args()
    if args.mode=='preflight':preflight()
    else:render()
