"""Continuous full-resolution matched capacity/control videos from real weights."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from .b1_capacity_component_net import CapacityComponentNet
from .component_region_net import ComponentNet,decode
from .train_luma_regions import ROOT,RUN
from .train_b1_distillation import as_tensor,to_rgb
from .fpga_video_resize import area_3to2,bilinear_2to3
from .publish_region_classes import encode,motion_map
from .render_wavelet_game import close_pipe,video_check
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True);p.add_argument('--step',type=int,default=2000);p.add_argument('--name',required=True);p.add_argument('--clip',default='stairs');a=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100458);cv2.setNumThreads(2);out=ROOT/'runs'/RUN/a.name;out.mkdir(exist_ok=False)
    nets={};weights={}
    for variant,cls in [('capacity',CapacityComponentNet),('small-control',ComponentNet)]:
        checkpoint=a.training/variant/f'fp32_checkpoint_{a.step:06d}.pt';s=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert s['scales'] is None and s['step']==a.step and all(file_hash(k)==v for k,v in s['binding']['code'].items())
        net=cls().cuda().eval();net.load_state_dict(s['model']);nets[variant]=net;weights[variant]=dict(checkpoint=str(checkpoint),sha256=file_hash(checkpoint),parameters=sum(v.numel() for v in net.parameters()))
    clip=next(c for c in json.loads((ROOT/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips'] if c['id']==a.clip)
    assert file_hash(clip['clip'])==clip['clip_sha256'];data=json.loads((a.training/'capacity/data.json').read_text(encoding='utf-8'));assert clip['source'] not in {r['source'] for r in data['pairs']}
    paths={name:out/f'{a.clip}_{name}_fp32_1080p.mp4' for name in nets};encoders={k:encode(v) for k,v in paths.items()};cap=cv2.VideoCapture(clip['clip']);previous=None;rows=[];same={k:0 for k in nets};n=0;start=time.perf_counter()
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            source=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));x=as_tensor(source)[None].cuda();images={'source':source}
            for variant,net in nets.items():
                pred=to_rgb(decode(net(x))[0]);images[variant]=pred;full=bilinear_2to3(pred);encoders[variant].stdin.write(full.tobytes())
                if n in (90,120,210):Image.fromarray(pred).save(out/f'{variant}_{n:03d}_native720.png');Image.fromarray(full).save(out/f'{variant}_{n:03d}_preencode1080.png')
            if previous:
                mx,my,valid=motion_map(previous['source'],source);row=dict(frame=n,valid_fraction=float(valid.mean()))
                for key,v in images.items():
                    old=cv2.remap(previous[key].astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE);row[key]=float(np.abs(v.astype(np.float32)-old).mean(2)[valid].mean()) if valid.any() else None
                    if key in same:same[key]+=int(np.array_equal(v,previous[key]))
                rows.append(row)
            previous=images;n+=1
            if n%60==0:print('CAPACITY_REAL_VIDEO',n,round(time.perf_counter()-start,1),flush=True)
    finally:
        cap.release()
        for encoder in encoders.values():close_pipe(encoder)
    assert n==clip['validation']['frames'];videos={k:video_check(v,n) for k,v in paths.items()};motion={k:dict(mean=float(np.mean([r[k] for r in rows if r[k] is not None])),p95=float(np.percentile([r[k] for r in rows if r[k] is not None],95))) for k in previous}
    write_json(out/'motion.json',dict(records=rows,summary=motion,not_temporal_acceptance=True));write_json(out/'report.json',dict(status='FP32_CAPACITY_CONTROL_RENDERED_NOT_ACCEPTED',videos=videos,paths={k:str(v) for k,v in paths.items()},weights=weights,frames=n,duration=n/30,motion=motion,identical_adjacent=same,source=clip['clip'],source_sha256=clip['clip_sha256'],native_inference=[1280,720],output=[1920,1080],postfilter=False,frame_blend=False,fpga=False,script_sha256=file_hash(__file__)))
    source_url='../../'+Path(clip['clip']).relative_to(ROOT/'runs').as_posix();title=f'B1容量扩容：{a.step}步真实视频对照'
    html=f'''<!doctype html><meta charset="utf-8"><title>{title}</title><style>body{{margin:20px;background:#151b23;color:#edf2f7;font:16px system-ui}}h1{{font-size:24px}}video{{display:block;width:min(100%,1280px);max-height:78vh}}button{{padding:10px;margin:4px}}p{{max-width:1100px}}</style><h1>{title}</h1><p>同一原片、同样续训次数。1080→720网络处理→1080，{n/30:g}秒、30帧/秒。未增加去噪、锐化或混帧。画质与部署均待验收。</p><button onclick="switchVideo('{paths['capacity'].name}','扩容网络：10,115参数')">扩容网络</button><button onclick="switchVideo('{paths['small-control'].name}','小网络对照：3,107参数')">小网络对照</button><button onclick="switchVideo('{source_url}','原片')">原片</button><button onclick="v.paused?v.play():v.pause()">播放 / 暂停</button><span id="label">扩容网络：10,115参数</span><video id="v" controls preload="auto" src="{paths['capacity'].name}"></video><script>function switchVideo(src,title){{const t=v.currentTime,playing=!v.paused;v.src=src;label.textContent=title;v.onloadedmetadata=()=>{{v.currentTime=Math.min(t,v.duration);if(playing)v.play();}};v.load();}}</script>'''
    (out/'preview.html').write_text(html,encoding='utf-8');print('CAPACITY_REAL_VIDEO_DONE',motion,flush=True)


if __name__=='__main__':main()
