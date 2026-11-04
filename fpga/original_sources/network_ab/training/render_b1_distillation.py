"""Independent B1 FP32 export and native720 full-video teacher comparisons."""
import argparse
import copy
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from network_ab.contract_torch import build_fp32_model
from .runtime import configure_workspace, file_hash, write_json, contract_guard
from .fpga_video_resize import area_3to2, bilinear_2to3
from .train_b1_distillation import as_tensor, to_rgb
from .trainable_animegan import TrainableAnimeGAN
from .prepare_b1_distillation import rgb_output, TEACHER_SHA
from .prepare_coherent_shared_data import pair_flow
from .render_temporal_native_detail import encode_pipe
from .render_wavelet_game import close_pipe, video_check


def export(root,run,step):
    path=run/f'train/checkpoint_{step:06d}.pt'
    state=torch.load(path,map_location='cpu',weights_only=True)
    contract,plan=contract_guard(); assert state['step']==step
    data=json.loads((run/'data/manifest.json').read_text(encoding='utf-8'))
    for name,value in state['binding']['code'].items():assert file_hash(name)==value
    out=run/f'final-{step}';out.mkdir(exist_ok=False)
    saved=out/'b1_style_fp32.pt'
    payload={'schema':'b1-distilled-style-fp32-v1','contract_canonical_sha256':plan['contract_sha256'],
             'contract_file_sha256':file_hash(Path(__file__).parents[1]/'model_b_contract.json'),
             'model_id':plan['model_id'],'step':step,'model':state['ema'],'teacher_sha256':TEACHER_SHA,
             'training_binding':state['binding'],'checkpoint_sha256':file_hash(path),
             'training_config':state['config'],'target_guard_config':data['guard'],
             'dtype':'FP32','quantized':False,'postprocess':'none; FPGA area-down and integer bilinear-up only'}
    torch.save(payload,saved)
    write_json(out/'model_contract.json',contract)
    write_json(out/'inference.json',{'model_id':plan['model_id'],'checkpoint':saved.name,'weight_key':'model',
        'input':'NCHW float32 RGB, (u8-128)/128; 720x1280',
        'output':'NCHW float32 clamped [-1,127/128]; round(y*128+128) to uint8',
        'preprocess':'fpga_video_resize.area_3to2 on uint8 full1080 RGB',
        'display':'fpga_video_resize.bilinear_2to3 on native720 uint8 output',
        'network_postprocess':'none','trained_target_guard_not_required_at_runtime':True,
        'quantized':False,'hardware_release':False})
    reference=build_fp32_model().cuda().eval();reference.load_state_dict(state['ema'])
    # Separately deserialize compact export into a new contract instance.
    net=build_fp32_model().cuda().eval();net.load_state_dict(torch.load(saved,map_location='cpu',weights_only=True)['model'])
    folded=copy.deepcopy(net)
    for seq in folded.operators.values():
        for i,layer in enumerate(seq):
            if isinstance(layer,torch.nn.BatchNorm2d):
                seq[i-1]=torch.nn.utils.fuse_conv_bn_eval(seq[i-1],layer);seq[i]=torch.nn.Identity()
    records=[]
    with torch.inference_mode():
        for row in data['validation']:
            x=as_tensor(np.array(Image.open(row['source'])))[None].cuda();a=reference(x);b=net(x);c=folded(x)
            error=float((a-b).abs().max());fold=float((b-c).abs().max())
            assert error==0 and fold<1e-4
            records.append({'case':row['id'],'reload_max':error,'fold_max':fold})
    write_json(out/'verification.json',{'status':'PASS','records':records,'path':str(saved),'sha256':file_hash(saved),
                                      'contract':plan['contract_sha256'],'checkpoint':str(path),'step':step,'not_int8':True})
    print('EXPORTED',str(saved),flush=True)


def temporal(previous,current,pa,pb,flow,mask):
    h,w=mask.shape;yy,xx=np.mgrid[:h,:w].astype(np.float32)
    s0,s1,o0,o1=[cv2.resize(a,(w,h),interpolation=cv2.INTER_AREA).astype(np.float32) for a in [previous,current,pa,pb]]
    a=(o0-s0);b=(o1-s1)
    error=np.abs(b-cv2.remap(a,xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_LINEAR))
    valid=mask.astype(bool)
    return float(error[valid].mean()) if valid.any() else None


def render(root,run,step):
    statepath=run/f'final-{step}/b1_style_fp32.pt';state=torch.load(statepath,map_location='cpu',weights_only=True)
    assert state['contract_canonical_sha256']==contract_guard()[1]['contract_sha256']
    net=build_fp32_model().cuda().eval();net.load_state_dict(state['model'])
    teacher_path=root/'runs/texture-recovery-20261003/final/refined_style_fp32.pt';assert file_hash(teacher_path)==TEACHER_SHA
    teacher=TrainableAnimeGAN(root).cuda().eval();teacher.load_state_dict(torch.load(teacher_path,map_location='cpu',weights_only=True)['generator'])
    clips=json.loads((root/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips']
    out=run/f'videos-{step}';out.mkdir(exist_ok=False);records=[]
    train_sources={r['source'] for r in json.loads((run/'data/manifest.json').read_text(encoding='utf-8'))['pairs']}
    for clip in clips:
        assert clip['source'] not in train_sources and file_hash(clip['clip'])==clip['clip_sha256']
        name=clip['id'];cap=cv2.VideoCapture(clip['clip']);n=0;stats=[];prev=None;samples=[]
        tp=out/f'{name}_teacher_1080p.mp4';sp=out/f'{name}_b1_1080p.mp4'
        tpipe=encode_pipe(tp,1920,1080,30);spipe=encode_pipe(sp,1920,1080,30)
        while True:
            ok,bgr=cap.read()
            if not ok:break
            source=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));begin=time.perf_counter()
            with torch.inference_mode():student=to_rgb(net(as_tensor(source)[None].cuda())[0])
            target=rgb_output(teacher,source)
            ts=bilinear_2to3(target);ss=bilinear_2to3(student)
            tpipe.stdin.write(ts.tobytes());spipe.stdin.write(ss.tobytes())
            row={'frame':n,'teacher_error_rgb255':float(np.abs(student.astype(np.float32)-target).mean()),
                 'source_change_rgb255':float(np.abs(student.astype(np.float32)-source).mean()),
                 'teacher_source_change_rgb255':float(np.abs(target.astype(np.float32)-source).mean()),
                 'seconds_teacher_student_and_encoding':time.perf_counter()-begin}
            if prev is not None and n%3==0:
                flow,mask=pair_flow(prev[0],source);flow=flow.astype(np.float32)
                row.update(student_temporal_style_mae=temporal(prev[0],source,prev[1],student,flow,mask),
                           teacher_temporal_style_mae=temporal(prev[0],source,prev[2],target,flow,mask),mask_fraction=float(mask.mean()))
            if n in [0,90,120,210,270,359]:
                for suffix,a in [('source720',source),('teacher720',target),('b1_720',student),('b1_preencode1080',ss)]:Image.fromarray(a).save(out/f'{name}_{n:03d}_{suffix}.png')
                panel=Image.new('RGB',(1920,388),'#101010');draw=ImageDraw.Draw(panel)
                for j,(a,label) in enumerate([(source,'SOURCE / FPGA DOWN'),(target,'ACCEPTED TEACHER + CORRECTION'),(student,f'B1 FP32 / STEP {step}')]):
                    panel.paste(Image.fromarray(a).resize((640,360)),(j*640,28));draw.text((j*640+8,7),label,fill='white')
                panel.save(out/f'{name}_{n:03d}_comparison.jpg',quality=95);samples.append(n)
            stats.append(row);prev=(source,student,target);n+=1
            if n%90==0:print('RENDER',name,n,'mean teacher error',np.mean([r['teacher_error_rgb255'] for r in stats]),flush=True)
        cap.release();close_pipe(tpipe);close_pipe(spipe)
        assert n==clip['validation']['frames']
        record={'id':name,'source':clip['clip'],'raw_source':clip['source'],'clip_sha256':clip['clip_sha256'],
                'teacher':str(tp),'student':str(sp),'teacher_validation':video_check(tp,n),'student_validation':video_check(sp,n),
                'samples':samples,'stats':stats,'model_calls':n,'teacher_calls':n}
        records.append(record)
        write_json(out/'report.json',{'status':'RENDERING' if len(records)<len(clips) else 'COMPLETE_REQUIRES_VISUAL_REVIEW',
            'records':records,'student_weights_sha256':file_hash(statepath),'teacher_weights_sha256':TEACHER_SHA,
            'pipeline':'1080 source -> exact area720 -> B1 FP32 -> exact integer bilinear1080 -> H264 CRF16',
            'student_postprocess':'none','repeated_frames':False,'frames_interpolated':False,
            'note':'Source-isolated regression clips previously reviewed. Temporal metric samples every third adjacent pair; diagnostic, not aesthetic pass.'})
    print('ALL_VIDEOS_COMPLETE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['export','render']);ap.add_argument('--step',type=int,default=2000);a=ap.parse_args()
    root=configure_workspace();run=root/'runs/b1-distillation-20261003';torch.set_num_threads(4);cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    (export if a.mode=='export' else render)(root,run,a.step)


if __name__=='__main__':main()
