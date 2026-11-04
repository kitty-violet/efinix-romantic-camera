"""Real 12--15 s before/after video with explicit generator graph metadata."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from .b1_capacity_component_net import CapacityComponentNet
from .clean_skip_capacity_net import CleanSkipNet
from .clean_rgb_skip_net import CleanRGBNet
from .contour_region_net import ContourRegionNet
from .component_region_net import decode
from .train_luma_regions import ROOT,RUN
from .train_b1_distillation import as_tensor,to_rgb
from .fpga_video_resize import area_3to2,bilinear_2to3
from .publish_region_classes import encode,motion_map
from .render_wavelet_game import close_pipe,video_check
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--name',required=True);ap.add_argument('--clip',default='stairs');args=ap.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100462);cv2.setNumThreads(2);base=ROOT/'runs'/RUN;out=base/args.name;out.mkdir(exist_ok=False)
    nets={};weights={}
    after_schema=torch.load(args.checkpoint,map_location='cpu',weights_only=True)['config'].get('graph_schema')
    before=base/('direct-teacher-rgb-v1/fp32_checkpoint_002000.pt' if after_schema=='contour-region-bottleneck-v1' else 'b1-capacity-control-v2/capacity/fp32_checkpoint_002000.pt')
    for key,path in [('before',before),('after',args.checkpoint)]:
        s=torch.load(path,map_location='cpu',weights_only=True);assert s['scales'] is None and all(file_hash(k)==v for k,v in s['binding']['code'].items())
        schema=s['config'].get('graph_schema');is_rgb=schema in ('direct-teacher-rgb-skip-v1','contour-region-bottleneck-v1')
        cls=ContourRegionNet if schema=='contour-region-bottleneck-v1' else CleanRGBNet if is_rgb else CleanSkipNet if schema=='clean-skip-capacity-v1' else CapacityComponentNet
        net=cls().cuda().eval();net.load_state_dict(s['model']);nets[key]=net;weights[key]=dict(path=str(path),sha256=file_hash(path),step=s['step'],parameters=sum(p.numel() for p in net.parameters()),graph=cls.__name__,output='RGB' if is_rgb else 'C3_palette')
    clip=next(c for c in json.loads((ROOT/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips'] if c['id']==args.clip)
    assert file_hash(clip['clip'])==clip['clip_sha256'];data=json.loads((args.checkpoint.parent/'data.json').read_text(encoding='utf-8'));assert clip['source'] not in {r['source'] for r in data['pairs']}
    paths={k:out/(args.clip+'_'+k+'_1080p.mp4') for k in nets};pipes={k:encode(p) for k,p in paths.items()};cap=cv2.VideoCapture(clip['clip']);previous=None;rows=[];identical={k:0 for k in nets};n=0;start=time.perf_counter()
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            source=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));x=as_tensor(source)[None].cuda();images={'source':source}
            for key,net in nets.items():
                raw=net(x);v=to_rgb((raw if weights[key]['output']=='RGB' else decode(raw))[0]);images[key]=v;full=bilinear_2to3(v);pipes[key].stdin.write(full.tobytes())
                if n in (90,120,210):Image.fromarray(v).save(out/f'{key}_{n:03d}_native720.png');Image.fromarray(full).save(out/f'{key}_{n:03d}_preencode1080.png')
            if previous:
                mx,my,valid=motion_map(previous['source'],source);r=dict(frame=n,valid_fraction=float(valid.mean()))
                for key,v in images.items():
                    old=cv2.remap(previous[key].astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE);e=np.abs(v.astype(np.float32)-old).mean(2)[valid]
                    r[key]=dict(mean=float(e.mean()),p95_pixel=float(np.percentile(e,95)),over8=float((e>8).mean())) if len(e) else None
                    if key in identical:identical[key]+=int(np.array_equal(v,previous[key]))
                rows.append(r)
            previous=images;n+=1
            if n%60==0:print('CLEAN_VIDEO',args.clip,n,round(time.perf_counter()-start,1),flush=True)
    finally:
        cap.release()
        for pipe in pipes.values():close_pipe(pipe)
    assert n==clip['validation']['frames'];videos={k:video_check(p,n) for k,p in paths.items()};motion={k:{metric:float(np.mean([r[k][metric] for r in rows if r[k] is not None])) for metric in ('mean','p95_pixel','over8')} for k in previous}
    write_json(out/'motion.json',dict(records=rows,summary=motion,not_temporal_acceptance=True));write_json(out/'report.json',dict(status='FP32_METHOD_COMPARISON_NOT_ACCEPTANCE',weights=weights,videos=videos,frames=n,duration=n/30,motion=motion,identical_adjacent=identical,source=clip['clip'],source_sha256=clip['clip_sha256'],inference=[1280,720],output=[1920,1080],postfilter=False,frame_blend=False,teacher_deployed=False,critic_deployed=False,fpga_validated=False,script_sha256=file_hash(__file__)))
    url='../../'+Path(clip['clip']).relative_to(ROOT/'runs').as_posix()
    teacher_records=json.loads((ROOT/'runs/b1-distillation-20261003/videos-2000/report.json').read_text(encoding='utf-8'))['records']
    teacher=next(r for r in teacher_records if r['id']==args.clip)
    assert teacher['clip_sha256']==clip['clip_sha256'] and file_hash(teacher['teacher'])==teacher['teacher_validation']['sha256']
    assert teacher['teacher_validation']['frames']==n and teacher['teacher_validation']['size']==[1920,1080]
    teacher_url='../../'+Path(teacher['teacher']).relative_to(ROOT/'runs').as_posix()
    html=f'''<!doctype html><meta charset="utf-8"><title>块内清洁训练：真实视频对照</title><style>body{{margin:20px;background:#151b23;color:#edf2f7;font:16px system-ui}}h1{{font-size:24px}}video{{display:block;width:min(100%,1280px);max-height:78vh}}button{{padding:10px;margin:4px}}</style><h1>块内清洁训练：{args.clip}，{n/30:g}秒</h1><p>同一原片。改前{weights['before']['parameters']}参数，改后{weights['after']['parameters']}参数（{weights['after']['graph']}）。1080→720网络→1080，30帧/秒。判别器和特征教师仅用于训练。无后滤波、锐化或混帧；这是待验收的FP32结果。</p><button onclick="change('{paths['after'].name}','改后')">改后</button><button onclick="change('{paths['before'].name}','改前')">改前</button><button onclick="change('{url}','原片')">原片</button><button onclick="v.paused?v.play():v.pause()">播放 / 暂停</button><span id="label">改后</span><video id="v" controls preload="auto" src="{paths['after'].name}"></video><script>function change(src,title){{const t=v.currentTime,playing=!v.paused;v.src=src;label.textContent=title;v.onloadedmetadata=()=>{{v.currentTime=Math.min(t,v.duration);if(playing)v.play();}};v.load();}}</script>'''
    html=html.replace('<button onclick="v.paused?',f'<button onclick="change(\'{teacher_url}\',\'原教师参考\')">教师参考</button><button onclick="v.paused?')
    (out/'preview.html').write_text(html,encoding='utf-8')
    write_json(out/'teacher_reference.json',dict(path=teacher['teacher'],validation=teacher['teacher_validation'],clip_sha256=teacher['clip_sha256'],same_frame_count=True))
    print('CLEAN_VIDEO_DONE',args.clip,motion,flush=True)


if __name__=='__main__':main()
