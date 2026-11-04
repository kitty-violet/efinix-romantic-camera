"""Same native frames/PTS for original,V14,R3 and three user curves."""
from core import *
import evaluation_utils as ev
from evaluate_pixel import mean_metrics
from evaluate_r3_r4 import content,controls
from r3_palette_blocks import geometry
from network_ab.training.media import decode_frames
from network_ab.training.fpga_video_resize import area_3to2
import imageio_ffmpeg

@torch.inference_mode()
def main():
    setup();dest=OUT/'pixel-three-curves-review';dest.mkdir(exist_ok=True);disk_guard();begin=time.perf_counter()
    paths={letter:OUT/f'exports/pixel-user-curve-{letter}/coefficients-padded.bin' for letter in 'ABC'}
    paths.update(original_pixel=OUT/'exports/flat-r3-user-selected/coefficients-padded.bin',v14=DEPLOY)
    engines={key:IntegerEngine(p.read_bytes()) for key,p in paths.items()}
    save_json(dest/'binding.json',dict(coefficients={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},
        drawn_final_tone_curves=True,contrast_readout=1.2,contrast_inverse_compensated=True,saturation=1.3,
        post_filter=False,inference_flow=False,scripts={p.name:sha(p) for p in [Path(__file__),OUT/'source/try_user_pixel_curves.py',OUT/'source/core.py']}))
    save_json(dest/'colour_controls.json',controls(engines))
    manifest=json.loads((ROOT/'output/b1-ink-wash-20261026/independent_test/manifest.json').read_text('utf-8'))
    clips=sorted(manifest['clips'],key=lambda c:['outdoor','station','dusk','corridor','park','foliage'].index(c['id']))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);reports=[]
    for clip in clips:
        scene=clip['id'];duration=3 if scene in ['outdoor','station'] else 1
        cache=OUT/'cache'/('three-curves-'+scene);cache.mkdir(parents=True,exist_ok=False)
        pictures={k:[] for k in ['A','B','C','comparison']};pts=[];ids=[];records=[];previous=None;hashes={k:[] for k in ['source',*engines]}
        for index,stamp,full in decode_frames(clip['source'],seconds=duration,hlg_exposure_stops=clip['exposure_stops']):
            if stamp>=duration:break
            if time.perf_counter()-begin>1500:raise TimeoutError('25 minute full review budget')
            if ids:assert index==ids[-1]+1 and stamp>pts[-1]
            source=area_3to2(full);x=tensor(source)[None]*128;images=dict(source=source,**{k:rgb(n(x)/128) for k,n in engines.items()})
            for key,z in images.items():hashes[key].append(hashlib.sha256(z.tobytes()).hexdigest())
            row=dict(frame=index,pts=stamp)
            if previous is not None and index%5==0:
                mx,my,valid,static,_=ev.flow_mask(previous['source'],source)
                for key,z in images.items():row[key]=ev.temporal(previous[key],z,mx,my,valid,static)
            if index%30==0:
                row['geometry']={key:geometry(images[key]) for key in 'ABC'};row['content']={key:content(source,images[key]) for key in 'ABC'}
            canvas=Image.new('RGB',(1920,788),'#111722');draw=ImageDraw.Draw(canvas)
            tiles=[('source','原图'),('v14','旧V14 / 纯B1整数'),('original_pixel','认可R3原版 / C100 S100'),
                ('A','曲线A / 实际B1 / S130'),('B','曲线B / 实际B1 / S130'),('C','曲线C / 实际B1 / S130')]
            for j,(key,label) in enumerate(tiles):
                xx=(j%3)*640;yy=(j//3)*394;canvas.paste(Image.fromarray(images[key]).resize((640,360),Image.Resampling.NEAREST),(xx,yy+34));draw.text((xx+8,yy+6),label,font=font,fill='white')
            for key in 'ABC':
                p=cache/f'{key}_{index:04d}.png';Image.fromarray(images[key]).save(p,compress_level=1);pictures[key].append(p)
            p=cache/f'compare_{index:04d}.png';canvas.save(p,compress_level=1);pictures['comparison'].append(p)
            if index in [0,30,60]:canvas.save(dest/f'{scene}_{index:04d}.jpg',quality=93)
            ids.append(index);pts.append(stamp);records.append(row);previous=images
            if index%30==0:print('THREE_CURVES_FRAME',scene,index,round(time.perf_counter()-begin,1),flush=True);disk_guard()
        assert ids==list(range(len(ids))) and len(ids)>duration*28
        assert all(len(set(v))==len(ids) for v in hashes.values())
        videos={key:ev.encode_vfr(cache,p,pts,dest/f'{scene}_{key}.mp4',imageio_ffmpeg.get_ffmpeg_exe()) for key,p in pictures.items()}
        report=dict(id=scene,source=clip['source'],source_sha256=sha(clip['source']),frames=len(ids),
            seconds=pts[-1]-pts[0]+float(np.median(np.diff(pts))),raw_unique_frames={k:len(set(v)) for k,v in hashes.items()},
            temporal={key:mean_metrics(records,key) for key in images},video=videos,
            content=[r['content'] for r in records if 'content' in r],geometry=[r['geometry'] for r in records if 'geometry' in r],
            post_filter=False,inference_flow=False,original_frame_rate_preserved=True,no_drop_or_repeated_frames=True)
        save_json(dest/f'{scene}_report.json',dict(summary=report,frame_ids=ids,pts_seconds=pts,records=records));reports.append(report)
        resolved=cache.resolve();assert resolved.parent==(OUT/'cache').resolve() and resolved.name=='three-curves-'+scene
        shutil.rmtree(resolved);save_json(dest/'summary.json',dict(clips=reports,all_six_complete=len(reports)==6,elapsed=time.perf_counter()-begin,disk=disk_guard()))
        print('THREE_CURVES_CLIP_DONE',scene,len(ids),flush=True)
    print('THREE_CURVES_REVIEW_COMPLETE',flush=True)

if __name__=='__main__':main()
