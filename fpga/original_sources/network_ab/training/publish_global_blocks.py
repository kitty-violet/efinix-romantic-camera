"""Recompute all native720 objects and apply one fixed colour-plane map.

Source1080 -> area720 -> FP32 student and frozen appearance processing ->
global colour planes -> bilinear1080. No encoded-video background is reused.
"""
import argparse,ast,json,shutil,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np,torch
from PIL import Image
from .publish_block_video import setup
from .publish_coarse_people import probability
from .probe_coarse_people import WEIGHT
from .coarse_people import refined_mask,composite
from .coarse_people_blocks import person_surface
from .global_colour_blocks import prepare_surface,PRESETS
from .painting_parameters import as_rgb
from .fpga_video_resize import area_3to2,bilinear_2to3
from .train_native_contours import tensor
from .train_b1_distillation import to_rgb
from .publish_region_classes import motion_map
from .publish_moderate_contours import encode
from .render_wavelet_game import close_pipe,video_check
from .runtime import file_hash,write_json
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

ROOT=Path(r'E:\fpga\model-training-workspace\runs\region-detail-deploy-20261004')
OUT=ROOT/'global-colour-blocks-video-20261005-v1'
PREVIOUS=ROOT/'person-colour-blocks-video-20261005-v1/plaza_coarse_people_15s_1080p.mp4'
PREVIOUS_HASH='4d632fd5f4413723446fb149e51b0292af1e36da2b479487364939a3a1b59916'


def colour_planes(lab,previous=None,flow=None):
    steps=np.array([18.,12.,12.],np.float32)
    unit=lab/steps;bins=np.floor(unit+.5)
    held=np.zeros(lab.shape[:2],bool)
    if previous is not None:
        mx,my,valid=flow
        prior=cv2.remap(previous['bins'],mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_REPLICATE)
        guide=cv2.remap(previous['lab'],mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        reliable=valid&(np.abs(lab-guide).max(2)<4.)
        keep=reliable[:,:,None]&(np.abs(unit-prior)<.56)
        held=np.any(keep&(bins!=prior),axis=2)
        bins=np.where(keep,prior,bins)
    return as_rgb(bins*steps),bins.astype(np.float32),held


def load():
    cfg,net,b,d,p,adapter,clip,state=setup()
    assert file_hash(PREVIOUS)==PREVIOUS_HASH and WEIGHT.is_file()
    seg=lraspp_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=21)
    seg.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True));seg=seg.cuda().eval()
    return cfg,net,b,d,p,adapter,clip,state,seg


@torch.inference_mode()
def process(raw,net,b,d,p,adapter,seg,pool,previous):
    source=area_3to2(raw);pred=to_rgb(net(tensor(source)[None].cuda())[0])
    base=d.surface(p.tone(pred,adapter),d.PRESETS['direction4'])
    # One GPU worker. Overlap independent CPU background contrast with GPU
    # person processing in this same process, not with another training run.
    background_job=pool.submit(b.transform_surface,base,'balanced')
    prob=probability(seg,source);flow=None
    if previous is not None:
        flow=motion_map(previous['source'],source);mx,my,valid=flow
        warped=cv2.remap(previous['prob'],mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        weight=.30*valid*np.exp(-((prob-warped)/.20)**2);prob=prob*(1-weight)+warped*weight
    mask=refined_mask(prob,source)
    people=person_surface(pred,adapter,p,'strongblocks',source)
    before=composite(background_job.result(),people,mask)
    lab=prepare_surface(before,'strong')
    out,bins,held=colour_planes(lab,previous,flow)
    state=dict(source=source,prob=prob,lab=lab,bins=bins,result=out,before=before)
    record=dict(held_bin_fraction=float(held.mean()),changed_fraction=float(np.any(out!=before,axis=2).mean()))
    if previous is not None:
        mx,my,valid=flow
        for name,a,current in [('before',previous['before'],before),('current',previous['result'],out)]:
            warp=cv2.remap(a.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            values=np.abs(current.astype(np.float32)-warp).mean(2)[valid]
            record[name]=dict(mean=float(values.mean()),p95=float(np.percentile(values,95)))
    return out,before,state,record


def bindings(cfg,state):
    files=dict(state['binding']['code'])
    for n in [Path(__file__).name,'global_colour_blocks.py','coarse_people.py','coarse_people_blocks.py','coarse_surface_fill.py','contour_guided_fill.py','painting_parameters.py','publish_block_video.py','block_video_runtime.py','publish_coarse_people.py','fpga_video_resize.py','publish_moderate_contours.py','render_wavelet_game.py','publish_region_classes.py']:
        path=Path(__file__).with_name(n);ast.parse(path.read_text(encoding='utf-8'));files[str(path)]=file_hash(path)
    from .block_video_runtime import STILLS
    for name,sha in cfg['code'].items():files[str(STILLS/'source_code'/name)]=sha
    files[str(STILLS/'color_adapter.npz')]=file_hash(STILLS/'color_adapter.npz')
    return files


@torch.inference_mode()
def run(mode):
    cfg,net,b,d,p,adapter,clip,state,seg=load()
    if mode=='preflight':
        OUT.mkdir(exist_ok=False);snap=OUT/'source_code';snap.mkdir();files=bindings(cfg,state)
        for i,(path,sha) in enumerate(files.items()):
            dest=snap/f'{i:02d}_{Path(path).name}';shutil.copy2(path,dest);assert file_hash(dest)==sha
        write_json(OUT/'pipeline.json',dict(source={k:v for k,v in clip.items() if k!='frames'},student=cfg['neural_weight_source'],student_sha256=cfg['neural_weight_sha256'],segmenter=str(WEIGHT),segmenter_sha256=file_hash(WEIGHT),code=files,
            previous=str(PREVIOUS),previous_sha256=PREVIOUS_HASH,global_surface=PRESETS['strong'],colour_steps=[18,12,12],hard_palette=True,
            hysteresis_halfwidth=.56,guide_reliability_Lab=4,teacher_inference=False,new_training=False,board_validated=False,
            chain='source SDR1080 -> integer area720 -> actual FP32 student -> frozen direction4/background blocks + person blocks -> full-frame TV and fixed Lab palette -> integer bilinear1080 -> H264 CRF16',
            encoded_background_reused=False,global_region_exemption=False,output_RGB_temporal_blending=False))
    frozen=json.loads((OUT/'pipeline.json').read_text(encoding='utf-8'));assert all(file_hash(path)==sha for path,sha in frozen['code'].items())
    cap=cv2.VideoCapture(clip['clip'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);assert cap.isOpened()
    previous=None;records=[];start=time.perf_counter();pipe=None;n=0;dups=0;selected={0,90,210,330,449}
    if mode=='preflight':
        wanted={90,210,330};end=331
    elif mode=='short':
        wanted=set(range(320,350));end=350;pipe=encode(OUT/'short_check_1s.mp4')
    else:
        assert json.loads((OUT/'short_verification.json').read_text(encoding='utf-8'))['complete']
        wanted=set(range(450));end=450;assert not (OUT/'global_blocks_15s_1080p.mp4').exists();pipe=encode(OUT/'global_blocks_15s_1080p.mp4')
    write_json(OUT/'STATUS.json',dict(status=mode,frames=0,total=len(wanted)))
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            for i in range(end):
                ok,bgr=cap.read();assert ok
                if i not in wanted:continue
                if mode=='preflight':previous=None
                tick=time.perf_counter();out,before,new_state,row=process(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),net,b,d,p,adapter,seg,pool,previous)
                row.update(frame=i,seconds=time.perf_counter()-tick);records.append(row)
                if previous is not None:dups+=int(np.array_equal(previous['result'],out))
                previous=new_state;n+=1
                output=bilinear_2to3(out)
                if pipe:pipe.stdin.write(output.tobytes())
                save=mode=='preflight' or (mode=='short' and i in (320,330,349)) or (mode=='render' and i in selected)
                if save:
                    prefix=mode if mode!='render' else 'frame'
                    Image.fromarray(out).save(OUT/f'{prefix}_{i:03d}_native720.png');Image.fromarray(output).save(OUT/f'{prefix}_{i:03d}_1080p.png')
                    Image.fromarray(before).save(OUT/f'{prefix}_{i:03d}_before720.png')
                if n%10==0 or mode=='preflight':
                    elapsed=time.perf_counter()-start
                    write_json(OUT/'STATUS.json',dict(status=mode,frames=n,total=len(wanted),elapsed_seconds=elapsed,remaining_seconds=elapsed/n*(len(wanted)-n)))
                    print('GLOBAL_BLOCKS',mode,n,'/',len(wanted),round(elapsed,1),round(row['seconds'],2),flush=True)
    except BaseException as exc:
        write_json(OUT/'STATUS.json',dict(status='INCOMPLETE',mode=mode,frames=n,error=repr(exc)));raise
    finally:
        cap.release()
        if pipe:close_pipe(pipe)
    assert n==len(wanted) and dups==0
    check=video_check(OUT/('short_check_1s.mp4' if mode=='short' else 'global_blocks_15s_1080p.mp4'),n) if pipe else None
    assert all(file_hash(path)==sha for path,sha in frozen['code'].items())
    write_json(OUT/(mode+'_verification.json'),dict(complete=True,frames=n,video=check,rows=records,seconds=time.perf_counter()-start,adjacent_duplicate_frames=dups,temporal_acceptance=False))
    write_json(OUT/'STATUS.json',dict(status=mode+'_COMPLETE',frames=n,total=n))
    print('GLOBAL_DONE',mode,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','short','render']);run(parser.parse_args().mode)
