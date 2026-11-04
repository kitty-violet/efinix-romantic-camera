"""A 15-second person-coarsening preview, reusing the approved video background.

People are re-inferred from original SDR footage at native 720p; the background
is the decoded previous 1080p delivery, so this is an honest composite preview,
not a fresh full-chain render or a newly trained/FPGA-ready model.
"""
import argparse,ast,json,shutil,time
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from .publish_block_video import setup
from .coarse_people import person_surface,refined_mask,composite,PRESETS
from .probe_coarse_people import WEIGHT
from .fpga_video_resize import area_3to2,bilinear_2to3,_up_axis
from .train_native_contours import tensor
from .train_b1_distillation import to_rgb
from .publish_moderate_contours import encode
from .publish_region_classes import motion_map
from .render_wavelet_game import close_pipe,video_check
from .runtime import file_hash,write_json

ROOT=Path(r'E:\fpga\model-training-workspace\runs\region-detail-deploy-20261004')
OUT=ROOT/'coarse-people-video-20261005-v2'
OLD=ROOT/'block-separation-video-20261004-v1/plaza_balanced_15s_1080p.mp4'
OLD_HASH='218ba88dfd6e5ef6cc46b017973822f6592974a3c0fb675e8ea00447503d3801'
SELECTED=(0,90,210,330,449)


def prepare():
    cfg,net,b,d,p,adapter,clip,state=setup()
    assert file_hash(OLD)==OLD_HASH and WEIGHT.is_file()
    seg=lraspp_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=21)
    seg.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True));seg=seg.cuda().eval()
    return cfg,net,p,adapter,clip,state,seg


@torch.inference_mode()
def probability(model,source):
    x=cv2.resize(source,(768,432),interpolation=cv2.INTER_AREA)
    x=torch.from_numpy(x.copy()).permute(2,0,1)[None].cuda().float()/255
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    p=model((x-mean)/std)['out'].softmax(1)[0,15].cpu().numpy()
    return cv2.resize(p,(1280,720),interpolation=cv2.INTER_LINEAR)


def up_mask(mask):
    q=np.uint8(np.round(mask*255))[:,:,None]
    return _up_axis(_up_axis(q,1),0)[:,:,0].astype(np.float32)/255


@torch.inference_mode()
def frame(source1080,background,net,p,adapter,seg,previous):
    source=area_3to2(source1080);pred=to_rgb(net(tensor(source)[None].cuda())[0])
    prob=probability(seg,source);flow=None
    if previous is not None:
        mx,my,valid=motion_map(previous['source'],source);flow=(mx,my,valid)
        warped=cv2.remap(previous['prob'],mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        # Stabilize only the semantic probability, never blend or warp faces.
        weight=.30*valid*np.exp(-((prob-warped)/.20)**2)
        prob=prob*(1-weight)+warped*weight
    mask=refined_mask(prob,source)
    people=person_surface(pred,adapter,p,'coarse',source)
    alpha=up_mask(mask);final=composite(background,bilinear_2to3(people),alpha)
    outside=alpha==0
    assert np.array_equal(final[outside],background[outside])
    record=dict(person_fraction=float((mask>.5).mean()),background_zero_alpha_max_error=0)
    if flow is not None:
        mx,my,valid=flow
        region=(mask>.8)&valid
        current_small=area_3to2(final);old_small=area_3to2(background)
        for k,prev,current in [('current',previous['final720'],current_small),('previous',previous['old720'],old_small)]:
            warp=cv2.remap(prev.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            values=np.abs(current.astype(np.float32)-warp).mean(2)[region]
            record[k]=dict(mean=float(values.mean()) if len(values) else None,p95=float(np.percentile(values,95)) if len(values) else None)
    state=dict(source=source,prob=prob,final720=area_3to2(final),old720=area_3to2(background))
    return final,people,mask,state,record


def open_capture(path):
    cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==450
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))==(1920,1080)
    assert abs(cap.get(cv2.CAP_PROP_FPS)-30)<1e-6
    return cap


@torch.inference_mode()
def render(short=False):
    cfg,net,p,adapter,clip,state,seg=prepare()
    if short:
        OUT.mkdir(exist_ok=False);snapshot=OUT/'source_code';snapshot.mkdir()
        files=dict(state['binding']['code'])
        for n in [Path(__file__).name,'coarse_people.py','coarse_surface_fill.py','contour_guided_fill.py','fpga_video_resize.py','publish_block_video.py','probe_coarse_people.py','publish_moderate_contours.py','publish_region_classes.py','render_wavelet_game.py']:
            path=Path(__file__).with_name(n);ast.parse(path.read_text(encoding='utf-8'));files[str(path)]=file_hash(path)
        for n,path in enumerate(files):shutil.copy2(path,snapshot/f'{n:02d}_{Path(path).name}')
        write_json(OUT/'pipeline.json',dict(source_clip={k:v for k,v in clip.items() if k!='frames'},student=cfg['neural_weight_source'],student_sha256=cfg['neural_weight_sha256'],
            background_video=str(OLD),background_sha256=OLD_HASH,segmenter=str(WEIGHT),segmenter_sha256=file_hash(WEIGHT),person_class=15,
            preset=PRESETS['coarse'],live_code_hashes=files,skin_min_tv_strength=.014,
            preview_chain='Original SDR1080 -> integer area720 -> actual FP32 student -> adaptive TV colour planes + fixed tone/continuous levels -> integer bilinear1080 -> composite over decoded previous delivery using automatic person mask -> H264 CRF16',
            semantic_mask_temporal_weight=.30,rgb_temporal_blending=False,geometry_warp=False,teacher_inference=False,new_training=False,fpga_validated=False,
            background_is_reused_encoded_video=True,background_reencoding=True))
        # Invariants specific to a masked colour-only change.
        a=np.full((4,6,3),30,np.uint8);b=np.full_like(a,200)
        assert np.array_equal(composite(a,b,np.zeros((4,6),np.float32)),a)
        assert np.array_equal(composite(a,b,np.ones((4,6),np.float32)),b)
        assert all(q>0 for q in [PRESETS['coarse']['strength'],.014])
    frozen=json.loads((OUT/'pipeline.json').read_text(encoding='utf-8'))
    assert all(file_hash(q)==h for q,h in frozen['live_code_hashes'].items())
    if not short:
        pre=json.loads((OUT/'short_verification.json').read_text(encoding='utf-8'))
        assert pre['frames']==30 and pre['complete']
    path=OUT/('short_check_1s.mp4' if short else 'plaza_coarse_people_15s_1080p.mp4');assert not path.exists()
    cap=open_capture(clip['clip']);back=open_capture(OLD);pipe=encode(path);previous=None;records=[];n=0;dups=0;start=time.perf_counter()
    begin,end=(320,350) if short else (0,450)
    write_json(OUT/'STATUS.json',dict(status='SHORT_CHECK' if short else 'RENDERING',frames=0,total=end-begin))
    try:
        for i in range(end):
            ok,raw=cap.read();ok2,bg=back.read();assert ok and ok2
            if i<begin:continue
            source=cv2.cvtColor(raw,cv2.COLOR_BGR2RGB);background=cv2.cvtColor(bg,cv2.COLOR_BGR2RGB)
            final,people,mask,new_state,record=frame(source,background,net,p,adapter,seg,previous)
            pipe.stdin.write(final.tobytes());record['frame']=i;records.append(record)
            if previous is not None:dups+=int(np.array_equal(previous['final720'],new_state['final720']))
            previous=new_state;n+=1
            if (short and i in (320,330,349)) or (not short and i in SELECTED):
                prefix='short_' if short else 'frame_'
                Image.fromarray(final).save(OUT/f'{prefix}{i:03d}_1080p.png')
                Image.fromarray(people).save(OUT/f'{prefix}{i:03d}_people720.png')
                Image.fromarray(np.uint8(mask*255)).save(OUT/f'{prefix}{i:03d}_mask720.png')
                if not short:Image.fromarray(background).save(OUT/f'old_{i:03d}_1080p.png')
            if n%10==0:
                elapsed=time.perf_counter()-start
                write_json(OUT/'STATUS.json',dict(status='SHORT_CHECK' if short else 'RENDERING',frames=n,total=end-begin,elapsed_seconds=elapsed,remaining_seconds=elapsed/n*(end-begin-n)))
                print('COARSE_PEOPLE',n,'/',end-begin,round(elapsed,1),flush=True)
    except BaseException as exc:
        write_json(OUT/'STATUS.json',dict(status='INCOMPLETE',frames=n,total=end-begin,error=repr(exc)));raise
    finally:
        cap.release();back.release();close_pipe(pipe)
    check=video_check(path,end-begin);assert n==end-begin and dups==0
    assert all(file_hash(q)==h for q,h in frozen['live_code_hashes'].items())
    summary={k:{metric:float(np.mean([r[k][metric] for r in records if k in r and r[k][metric] is not None])) for metric in ('mean','p95')} for k in ('previous','current')}
    write_json(OUT/('short_verification.json' if short else 'video_verification.json'),dict(complete=True,frames=n,video=check,seconds=time.perf_counter()-start,records=records,person_flow_diagnostic=summary,identical_adjacent_frames=dups,temporal_acceptance=False))
    write_json(OUT/'STATUS.json',dict(status='SHORT_CHECK_COMPLETE' if short else 'COMPLETE_PENDING_VISUAL_CHECK',frames=n,total=end-begin))
    print('COARSE_PEOPLE_DONE',path,summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['short','render']);args=parser.parse_args();render(args.mode=='short')
