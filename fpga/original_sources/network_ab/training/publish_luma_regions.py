"""Continuous 1080->720 model->1080 evaluation, no postfilter or time mixing."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np
from PIL import Image
import torch
from .train_luma_regions import LumaNet,display,ROOT,RUN
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .train_b1_distillation import as_tensor,to_rgb
from .publish_region_classes import encode,motion_map
from .render_wavelet_game import close_pipe,video_check
from .fpga_video_resize import area_3to2,bilinear_2to3

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--clip',action='append');p.add_argument('--context',action='store_true');p.add_argument('--native',action='store_true');p.add_argument('--coarse',action='store_true');a=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100427);cv2.setNumThreads(2)
    root=ROOT/'runs'/RUN;out=root/a.name;assert out.parent==root;out.mkdir(exist_ok=False)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);assert all(file_hash(p)==h for p,h in state['binding']['code'].items())
    cls=LumaNet
    if a.context:
        from .context_region_net import ContextNet
        cls=ContextNet
    if a.native:
        from .skip_filter_net import SkipFilterNet
        cls=SkipFilterNet
    if a.coarse:
        from .coarse_skip_net import CoarseSkipNet
        cls=CoarseSkipNet
    render=display
    if state['model']['layers.rgb.0.weight'].shape[0]==3:
        from .component_region_net import ComponentNet,decode
        cls=ComponentNet;render=decode
    net=cls().cuda().eval();net.load_state_dict(state['model']);net.scales=state['scales']
    assert (state['mode']=='qat')==(net.scales is not None)
    clips=json.loads((ROOT/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips'];records=[]
    data=json.loads((root/'luma/data.json').read_text(encoding='utf-8'));train={r['source'] for r in data['pairs']}
    for clip in clips:
        if a.clip and clip['id'] not in a.clip:continue
        assert clip['source'] not in train and file_hash(clip['clip'])==clip['clip_sha256']
        path=out/(clip['id']+'_luma_'+state['mode']+'_1080p.mp4');encoder=encode(path);cap=cv2.VideoCapture(clip['clip']);n=0;prev=None;logs=[];same=0;start=time.perf_counter()
        try:
            while True:
                ok,bgr=cap.read()
                if not ok:break
                source=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                with torch.no_grad():pred=to_rgb(render(net(as_tensor(source)[None].cuda()))[0])
                full=bilinear_2to3(pred);encoder.stdin.write(full.tobytes())
                if n in (90,120,210):
                    Image.fromarray(pred).save(out/f'{clip["id"]}_{n:03d}_native720.png');Image.fromarray(full).save(out/f'{clip["id"]}_{n:03d}_preencode1080.png')
                if prev:
                    mx,my,mask=motion_map(prev[0],source);row=dict(frame=n,valid_fraction=float(mask.mean()))
                    for key,pa,ca in [('source',prev[0],source),('network',prev[1],pred)]:
                        w=cv2.remap(pa.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                        row[key]=float(np.abs(ca.astype(np.float32)-w).mean(2)[mask].mean()) if mask.any() else None
                    same+=int(np.array_equal(pred,prev[1]));logs.append(row)
                prev=(source,pred);n+=1
                if n%60==0:print('LUMA_VIDEO',clip['id'],n,round(time.perf_counter()-start,1),flush=True)
        finally:cap.release();close_pipe(encoder)
        assert n==clip['validation']['frames'];check=video_check(path,n)
        motion={k:dict(mean=float(np.mean([r[k] for r in logs if r[k] is not None])),p95=float(np.percentile([r[k] for r in logs if r[k] is not None],95))) for k in ('source','network')}
        write_json(out/(clip['id']+'_motion.json'),dict(records=logs,summary=motion,method='source optical-flow RGB diagnostic, NOT temporal acceptance'))
        records.append(dict(id=clip['id'],frames=n,duration=n/30,path=str(path),sha256=file_hash(path),check=check,motion=motion,identical_adjacent_outputs=same))
        write_json(out/'report.json',dict(status='RENDERED_NOT_ACCEPTED',records=records,mode=state['mode'],steps=state['step'],checkpoint=str(a.checkpoint),checkpoint_sha256=file_hash(a.checkpoint),
            pipeline='1080 area integer->720 model->1080 bilinear integer',postfilter=False,time_blending=False,duplicate_padding=False,fpga=False))

if __name__=='__main__':main()
