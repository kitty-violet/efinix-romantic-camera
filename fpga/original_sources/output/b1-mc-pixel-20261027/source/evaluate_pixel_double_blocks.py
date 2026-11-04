"""Final doubling attempt: actual BIN comparison; do not call it an 8x8 result."""
from core import *
import evaluation_utils as ev
from evaluate_pixel import mean_metrics
from evaluate_r3_r4 import content
from try_pixel_double_blocks import grid_stats
from network_ab.training.media import decode_frames
from network_ab.training.fpga_video_resize import area_3to2
import imageio_ffmpeg

@torch.inference_mode()
def main():
    setup();disk_guard();begin=time.perf_counter();d=OUT/'pixel-r40-video-review';d.mkdir(exist_ok=True)
    paths=dict(selected_R39=OUT/'exports/pixel-A-sharp-medium-r39/coefficients-padded.bin',
               candidate_R40=OUT/'exports/pixel-A-last-coarsening-r40/coefficients-padded.bin')
    engines={k:IntegerEngine(p.read_bytes()) for k,p in paths.items()}
    save_json(d/'binding.json',dict(models={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},
        selected_by_user='R39 medium',attempt='last requested doubling attempt',actual_grid=4,target_grid=8,
        strict8_success=False,reference8_not_in_videos=True,post_filter=False,flow_only_for_evaluation=True))
    manifest=json.loads((ROOT/'output/b1-ink-wash-20261026/independent_test/manifest.json').read_text('utf-8'))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);reports=[]
    for scene in ['corridor','outdoor']:
        clip=next(c for c in manifest['clips'] if c['id']==scene)
        cache=OUT/'cache'/('pixel-r40-'+scene);cache.mkdir(exist_ok=False)
        pictures={k:[] for k in ['selected','candidate','comparison']};pts=[];rows=[];previous=None
        hashes={k:set() for k in ['source',*engines]}
        for index,stamp,frame in decode_frames(clip['source'],seconds=3,hlg_exposure_stops=clip['exposure_stops']):
            if stamp>=3:break
            assert index==len(pts)
            if time.perf_counter()-begin>600:raise TimeoutError('ten minute total review budget')
            a=area_3to2(frame);x=tensor(a)[None]*128
            images={'source':a,**{k:rgb(e(x)/128) for k,e in engines.items()}}
            for k,z in images.items():hashes[k].add(hashlib.sha256(z.tobytes()).hexdigest())
            row=dict(frame=index,pts=stamp)
            if previous is not None and index%5==0:
                mx,my,valid,static,_=ev.flow_mask(previous['source'],a)
                row.update({k:ev.temporal(previous[k],z,mx,my,valid,static) for k,z in images.items()})
            if index%30==0:
                row['content']={k:content(a,images[k]) for k in engines}
                row['grids']={k:{str(cell):grid_stats(images[k],cell) for cell in [4,8]} for k in engines}
            canvas=Image.new('RGB',(1280,790),'#111722');draw=ImageDraw.Draw(canvas)
            for col,(k,label) in enumerate([('selected_R39','已选 R39中强 / 4×4'),('candidate_R40','最后合并试验 / 实际仍4网格')]):
                im=Image.fromarray(images[k]);canvas.paste(im.resize((640,360),Image.Resampling.NEAREST),(col*640,34))
                draw.text((col*640+8,5),label,font=font,fill='white')
                canvas.paste(im.crop((320,96,960,452)),(col*640,434))
                draw.text((col*640+8,401),'下方：同位置原生像素局部',font=font,fill='#becddd')
            for key,im in [('selected',Image.fromarray(images['selected_R39'])),('candidate',Image.fromarray(images['candidate_R40'])),('comparison',canvas)]:
                p=cache/f'{key}_{index:04d}.png';im.save(p,compress_level=1);pictures[key].append(p)
                if index in [0,30,60]:im.save(d/f'{scene}_{key}_{index:04d}.png')
            pts.append(stamp);rows.append(row);previous=images
            if index%30==0:print('FINAL_DOUBLE_VIDEO',scene,index,round(time.perf_counter()-begin,1),flush=True)
        assert len(pts)==91 and all(len(v)==91 for v in hashes.values())
        videos={k:ev.encode_vfr(cache,v,pts,d/f'{scene}_{k}.mp4',imageio_ffmpeg.get_ffmpeg_exe()) for k,v in pictures.items()}
        report=dict(scene=scene,frames=91,seconds=pts[-1]+float(np.median(np.diff(pts))),pts=pts,
            temporal={k:mean_metrics(rows,k) for k in images},records=rows,videos=videos,
            unique_raw_frames={k:len(v) for k,v in hashes.items()})
        reports.append(report);save_json(d/f'{scene}_report.json',report)
        assert cache.resolve().parent==(OUT/'cache').resolve() and cache.name=='pixel-r40-'+scene
        files=list(cache.iterdir());assert all(p.is_file() and p.suffix in ['.png','.ffconcat'] for p in files)
        for p in files:p.unlink()
        cache.rmdir();save_json(d/'summary.json',dict(complete=False,reports=reports))
    encoded=[]
    for scene in ['corridor','outdoor']:
        for variant in ['selected','candidate']:
            p=d/f'{scene}_{variant}.mp4';cap=cv2.VideoCapture(str(p));hs=[];shapes=set()
            while True:
                ok,bgr=cap.read()
                if not ok:break
                hs.append(hashlib.sha256(bgr.tobytes()).hexdigest());shapes.add(tuple(bgr.shape))
            cap.release();assert len(hs)==len(set(hs))==91 and shapes=={(720,1280,3)}
            encoded.append(dict(scene=scene,variant=variant,frames=91,unique_frames=91,resolution=[1280,720]))
    save_json(d/'summary.json',dict(complete=True,reports=reports,encoded=encoded,seconds=time.perf_counter()-begin,
        strict8_succeeded=False,selected_R39_preserved=True,subjective_player_review=False,disk=disk_guard()))
    print('FINAL_DOUBLE_VIDEO_COMPLETE',flush=True)

if __name__=='__main__':main()
