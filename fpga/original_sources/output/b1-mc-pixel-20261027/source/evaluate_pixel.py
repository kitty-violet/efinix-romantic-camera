"""Native decoded-coefficient B1 evaluation. Optical flow is metrics only."""
from core import *
from pixel_targets import target,panel
import evaluation_utils as ev
from network_ab.training.media import decode_frames
from network_ab.training.fpga_video_resize import area_3to2
import imageio_ffmpeg, argparse

def pixel_stats(a):
    blocks=a.reshape(a.shape[0]//4,4,a.shape[1]//4,4,3)
    extent=blocks.max((1,3)).astype(np.int16)-blocks.min((1,3)).astype(np.int16)
    dx=np.abs(np.diff(a.astype(np.int16),axis=1)).mean(2)
    dy=np.abs(np.diff(a.astype(np.int16),axis=0)).mean(2)
    gx=np.arange(1,a.shape[1])%4!=0;gy=np.arange(1,a.shape[0])%4!=0
    return dict(cell_max_rgb_extent=int(extent.max()),flat_cells_fraction=float((extent.max(2)==0).mean()),
        gradient_outside_grid_mean=float((dx[:,gx].mean()+dy[gy].mean())/2),
        distinct_rgb=int(len(np.unique(a.reshape(-1,3),axis=0))))

def mean_metrics(records,key):
    values=[r[key] for r in records if key in r]
    return {k:float(np.mean([x[k] for x in values if x[k] is not None]))
        for k in values[0] if any(x[k] is not None for x in values)} if values else {}

@torch.inference_mode()
def main(args):
    setup();disk_guard();started=time.perf_counter()
    exp=OUT/'exports/mc-pixel-v1';raw=(exp/'coefficients-padded.bin').read_bytes()
    engine=IntegerEngine(raw);old=IntegerEngine(DEPLOY.read_bytes());n=load_model(exp/'selected.pt')
    verification=json.loads((exp/'verification.json').read_text('utf-8'))
    assert sha(exp/'coefficients-padded.bin')==verification['sha256_padded']
    dest=OUT/'evaluation';dest.mkdir(exist_ok=True);images=[];records=[]
    for scene in ['outdoor','indoor']:
        a=read(OUT/f'targets/{scene}_source.png');t=target(a);x=tensor(a)[None]
        q=engine(x*128);y=rgb(q/128);v14=rgb(old(x*128)/128)
        assert torch.equal(q,n(x)*128)
        assert pixel_stats(y)['flat_cells_fraction']==1.
        for kind,im in [('student',y),('old_v14',v14)]:Image.fromarray(im).save(dest/f'{scene}_{kind}.png')
        panel([a,t,y],['原图','算法目标 / 非B1','学生 / 纯B1整数'],dest/f'{scene}_target_student.jpg',width=640)
        crop=(slice(192,384),slice(448,768))
        panel([a[crop],t[crop],y[crop]],['原图局部 ×2','目标局部 ×2','学生局部 ×2 / 纯B1整数'],dest/f'{scene}_detail.png',width=640)
        records.append(dict(scene=scene,mae_target=float(np.abs(y.astype(float)-t).mean()),
            source=pixel_stats(a),target=pixel_stats(t),student=pixel_stats(y),old_v14=pixel_stats(v14)))
    save_json(dest/'static.json',dict(records=records,coeff_sha=sha(exp/'coefficients-padded.bin'),
        inference='decoded ordinary B1 INT8 weights, INT32 MAC, Q17 round/saturate; no post filter',
        full_frame_cell_phase='4x4 from two stride2/nearest2 stages',new_geometry=False))
    print('STATIC_DONE',json.dumps(records,ensure_ascii=False),flush=True)
    if not args.video:return
    manifest=json.loads((ROOT/'output/b1-ink-wash-20261026/independent_test/manifest.json').read_text('utf-8'))
    clips=[r for r in manifest['clips'] if r['id'] in ['outdoor','station']]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    completed=[]
    for clip in clips:
        scene='indoor' if clip['id']=='station' else 'outdoor'
        cache=OUT/'cache'/scene;cache.mkdir(parents=True,exist_ok=False)
        frameids=[];pts=[];purepaths=[];comparepaths=[];metrics=[];prev=None
        hashes={k:[] for k in ['source','target','old_v14','student']}
        for index,timestamp,full in decode_frames(clip['source'],seconds=3,hlg_exposure_stops=clip['exposure_stops']):
            if timestamp>=3:break
            if time.perf_counter()-started>900:raise TimeoutError('15 minute evaluation host limit')
            if frameids:assert index==frameids[-1]+1 and timestamp>pts[-1]
            a=area_3to2(full);x=tensor(a)[None]
            y=rgb(engine(x*128)/128);v14=rgb(old(x*128)/128);t=target(a)
            values=dict(source=a,target=t,old_v14=v14,student=y)
            for k,v in values.items():hashes[k].append(hashlib.sha256(v.tobytes()).hexdigest())
            m=dict(frame=index,pts=timestamp,target_mae=float(np.abs(y.astype(float)-t).mean()),
                flat_cells=pixel_stats(y)['flat_cells_fraction'])
            assert m['flat_cells']==1.
            if prev is not None and index%5==0:
                mx,my,valid,static,_=ev.flow_mask(prev['source'],a)
                for k in values:m[k]=ev.temporal(prev[k],values[k],mx,my,valid,static)
            if index in [0,30,60]:
                panel([a,t,y],['原图','目标 / 非B1','学生 / 纯B1整数'],dest/f'{scene}_{index:04d}.jpg')
            p=cache/f'pure_{index:04d}.png';Image.fromarray(y).save(p,compress_level=1);purepaths.append(p)
            canvas=Image.new('RGB',(1920,394),'#111722');draw=ImageDraw.Draw(canvas)
            for j,(label,im) in enumerate([('原图',a),('旧V14 / 纯B1整数',v14),('MC学生 / 纯B1整数',y)]):
                canvas.paste(Image.fromarray(im).resize((640,360),Image.Resampling.NEAREST),(640*j,34))
                draw.text((640*j+8,6),label,font=font,fill='white')
            p=cache/f'compare_{index:04d}.png';canvas.save(p,compress_level=1);comparepaths.append(p)
            frameids.append(index);pts.append(timestamp);metrics.append(m);prev=values
            if index%30==0:print('VIDEO',scene,index,round(time.perf_counter()-started,1),flush=True);disk_guard()
        assert len(frameids)>85 and frameids==list(range(len(frameids)))
        ff=imageio_ffmpeg.get_ffmpeg_exe()
        videos={k:ev.encode_vfr(cache,paths,pts,dest/f'{scene}_{k}.mp4',ff)
            for k,paths in [('pure_B1_integer',purepaths),('source_v14_student',comparepaths)]}
        r=dict(id=scene,source=clip['source'],source_sha256=sha(clip['source']),frames=len(frameids),frame_ids=frameids,
            pts_seconds=pts,duration_seconds=pts[-1]-pts[0]+float(np.median(np.diff(pts))),
            coeff_sha256=sha(exp/'coefficients-padded.bin'),old_v14_sha256=sha(DEPLOY),
            videos=videos,metrics=metrics,temporal_summary={k:mean_metrics(metrics,k) for k in hashes},
            raw_unique_frames={k:len(set(v)) for k,v in hashes.items()},
            consecutive_duplicates={k:sum(a==b for a,b in zip(v,v[1:])) for k,v in hashes.items()},
            mean_target_mae=float(np.mean([m['target_mae'] for m in metrics])),
            all_cells_exactly_flat=True,history_or_flow_inference=False,post_filter=False,
            fps_is_software_playback_not_fpga_throughput=True)
        save_json(dest/f'{scene}_report.json',r)
        # Only own per-clip temporary frame cache is removed, after video decode/PTS checks.
        resolved=cache.resolve();assert resolved.parent==(OUT/'cache').resolve() and resolved.name in ['indoor','outdoor']
        shutil.rmtree(resolved)
        completed.append({k:v for k,v in r.items() if k not in ['metrics','frame_ids','pts_seconds']})
        save_json(dest/'video_summary.json',dict(clips=completed,all_two_complete=len(completed)==2,
            elapsed_seconds=time.perf_counter()-started,disk=disk_guard()))
        print('CLIP_DONE',scene,len(frameids),flush=True)
    print('PIXEL_EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--video',action='store_true');main(ap.parse_args())
