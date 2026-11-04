"""Matched full720 adjacent-frame probes, actual exported integer models."""
import argparse,json,shutil
from pathlib import Path
import cv2,numpy as np,torch
from .runtime import write_json,file_hash,configure_workspace
from .probe_b1_encoder_gauge import load
from .render_b1_region_trial import region_temporal_diagnostics
from .publish_region_classes import motion_map
from .fpga_video_resize import area_3to2
from . import train_b1_minimum_area as m

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(2);root=configure_workspace();out=Path(a.out);out.mkdir(exist_ok=False)
    nets={};hashes={}
    for item in a.models:
        name,path=item.split('=',1);s,n=load(path);raw=n.export();nets[name]=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cuda');hashes[name]=file_hash(path)
    clips=json.loads((root/'runs/clear-style-control-20261003/video_clips.json').read_text('utf-8'))['clips'];records=[]
    for clip in clips:
        if clip['id'] not in ('stairs','plaza','new_hallway'):continue
        cap=cv2.VideoCapture(clip['clip'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);assert cap.isOpened()
        for frame in (90,120,210,330):
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame);xx=[]
            for _ in range(2):
                ok,bgr=cap.read();assert ok;xx.append(area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
            mx,my,valid=motion_map(*xx);sample=dict(scene=clip['id'],frame=frame,models={})
            for name,net in nets.items():
                pp=[]
                for x in xx:
                    y=net(torch.tensor(x.transpose(2,0,1)[None].copy(),device='cuda',dtype=torch.float32)-128)
                    pp.append((y[0].permute(1,2,0).cpu().numpy()+128).astype(np.uint8))
                warp=cv2.remap(pp[0].astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                delta=np.abs(warp-pp[1]).mean(2)[valid];rec=region_temporal_diagnostics(*pp,mx,my,valid)
                rec['warp_mean']=float(delta.mean());rec['warp_over16']=float((delta>16).mean());sample['models'][name]=rec
            records.append(sample);print('QUADRANT_MOTION',clip['id'],frame,{k:round(v['warp_mean'],3) for k,v in sample['models'].items()},flush=True)
        cap.release()
    summary={}
    for name in nets:
        summary[name]={}
        for scene in ('stairs','plaza','new_hallway'):
            rr=[r['models'][name] for r in records if r['scene']==scene]
            summary[name][scene]=dict(warp_mean=float(np.mean([r['warp_mean'] for r in rr])),interior_mean=float(np.mean([r['interior']['mean'] for r in rr if r['interior']])),contour_mean=float(np.mean([r['contours']['mean_distance720'] for r in rr if r['contours']])))
    shutil.copy2(__file__,out/Path(__file__).name);write_json(out/'review.json',dict(args=vars(a),records=records,summary=summary,checkpoint_sha256=hashes,whole720_frames=True,limited_temporal_sample=True,not_a_video_acceptance=True))
    print('QUADRANT_MOTION_SUMMARY',summary,flush=True)

if __name__=='__main__':main()
