"""Same-frame temporal/content audit of pure pixel boundary cleanup."""
from core import *
import evaluation_utils as ev
from evaluate_pixel import mean_metrics
from evaluate_r3_r4 import content
from network_ab.training.media import decode_frames
from network_ab.training.fpga_video_resize import area_3to2
import imageio_ffmpeg

@torch.inference_mode()
def main():
    setup(271201);begin=time.perf_counter();d=OUT/'pixel-r36-video-review';d.mkdir(exist_ok=True);disk_guard()
    paths=dict(original_A=OUT/'exports/pixel-user-curve-A/coefficients-padded.bin',
        r28=OUT/'exports/pixel-A-fringe-r28/coefficients-padded.bin',
        flat_only=OUT/'pixel-r36-crisp-edges/flat/candidate.bin',
        crisp_r36=OUT/'exports/pixel-A-crisp-r36/coefficients-padded.bin')
    engines={k:IntegerEngine(p.read_bytes()) for k,p in paths.items()}
    manifest=json.loads((ROOT/'output/b1-ink-wash-20261026/independent_test/manifest.json').read_text('utf-8'))
    clips=manifest['clips'];ids={c['id'] for c in clips}
    names=[n for n in ['corridor','outdoor','dusk','park','foliage'] if n in ids]
    assert names[:2]==['corridor','outdoor']
    save_json(d/'binding.json',dict(models={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},
        MC_route_closed=True,training=False,curve='A',saturation=1.3,inference_postprocessing=False,
        optical_flow_is_evaluation_only=True,inference_history=False,script_sha256=sha(__file__)))
    shutil.copy2(__file__,d/'source_used.py');font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20);reports=[]
    for scene in names:
        full=scene in ['corridor','outdoor'];seconds=3 if full else 1
        clip=next(c for c in clips if c['id']==scene);cache=OUT/'cache'/('pixel-r36-'+scene);cache.mkdir(exist_ok=True)
        assert not list(cache.iterdir());pictures={'native':[],'comparison':[]};pts=[];rows=[];previous=None
        unique={k:set() for k in ['source',*engines]}
        for index,stamp,frame in decode_frames(clip['source'],seconds=seconds,hlg_exposure_stops=clip['exposure_stops']):
            if stamp>=seconds:break
            assert index==len(pts)
            if time.perf_counter()-begin>900:raise TimeoutError('15 minute bounded pixel review')
            a=area_3to2(frame);x=tensor(a)[None]*128;images={'source':a,**{k:rgb(n(x)/128) for k,n in engines.items()}}
            for k,z in images.items():unique[k].add(hashlib.sha256(z.tobytes()).hexdigest())
            row={'frame':index,'pts':stamp}
            if previous is not None and index%5==0:
                mx,my,valid,static,_=ev.flow_mask(previous['source'],a)
                row.update({k:ev.temporal(previous[k],z,mx,my,valid,static) for k,z in images.items()})
            if index%30==0:
                row['content']={k:content(a,images[k]) for k in engines}
            if full:
                panel=Image.new('RGB',(1280,790),'#111722');draw=ImageDraw.Draw(panel)
                for j,(k,title) in enumerate([('original_A','原曲线A / 原版块面'),('crisp_r36','R36 / 清理边缘过渡与杂色')]):
                    panel.paste(Image.fromarray(images[k]).resize((640,360),Image.Resampling.NEAREST),(j*640,34))
                    draw.text((j*640+8,6),title,font=font,fill='white')
                    # 1:1 native detail, no interpolated enlargement.
                    crop=images[k][96:452,320:960]
                    panel.paste(Image.fromarray(crop),(j*640,434))
                    draw.text((j*640+8,401),'下方：同位置原生像素局部',font=font,fill='#c4d6e8')
                for k,im in [('native',Image.fromarray(images['crisp_r36'])),('comparison',panel)]:
                    p=cache/f'{k}_{index:04d}.png';im.save(p,compress_level=1);pictures[k].append(p)
                if index in [0,30,60]:
                    panel.save(d/f'{scene}_comparison_{index:04d}.jpg',quality=95)
                    Image.fromarray(images['crisp_r36']).save(d/f'{scene}_native_{index:04d}.png')
            previous=images;pts.append(stamp);rows.append(row)
            if index%30==0:print('PIXEL_CRISP_VIDEO',scene,index,round(time.perf_counter()-begin,1),flush=True);disk_guard()
        assert len(pts)>=(90 if full else 29)
        assert all(len(v)==len(pts) for v in unique.values()),(scene,{k:len(v) for k,v in unique.items()},len(pts))
        videos={k:ev.encode_vfr(cache,v,pts,d/f'{scene}_{k}.mp4',imageio_ffmpeg.get_ffmpeg_exe()) for k,v in pictures.items()} if full else {}
        report=dict(scene=scene,source=clip['source'],source_sha256=sha(clip['source']),frames=len(pts),
            seconds=pts[-1]+float(np.median(np.diff(pts))),pts=pts,unique_frames={k:len(v) for k,v in unique.items()},
            temporal={k:mean_metrics(rows,k) for k in images},content=[r['content'] for r in rows if 'content' in r],
            records=rows,videos=videos,all_frames_preserved=True)
        save_json(d/f'{scene}_report.json',report);reports.append(report)
        owned=list(cache.iterdir());assert cache.resolve().parent==(OUT/'cache').resolve() and cache.name=='pixel-r36-'+scene
        assert all(p.is_file() and p.resolve().parent==cache.resolve() and p.suffix in ['.png','.ffconcat'] for p in owned)
        for p in owned:p.unlink()
        cache.rmdir();save_json(d/'summary.json',dict(complete=False,reports=reports,seconds=time.perf_counter()-begin))
    # Audit the delivered native encoded files, independently of raw tensor hashes.
    encoded=[]
    for scene in ['corridor','outdoor']:
        p=d/f'{scene}_native.mp4';cap=cv2.VideoCapture(str(p));hashes=[];shapes=set()
        while True:
            ok,bgr=cap.read()
            if not ok:break
            hashes.append(hashlib.sha256(bgr.tobytes()).hexdigest());shapes.add(tuple(bgr.shape))
            if len(hashes)==1:Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(d/f'{scene}_encoded_first.png')
        cap.release();expected=next(r['frames'] for r in reports if r['scene']==scene)
        assert len(hashes)==len(set(hashes))==expected and shapes=={(720,1280,3)}
        encoded.append(dict(scene=scene,frames=expected,unique_encoded_frames=expected,resolution=[1280,720]))
    save_json(d/'summary.json',dict(complete=True,reports=reports,encoded=encoded,seconds=time.perf_counter()-begin,
        subjective_motion_player_review=False,MC_route_closed=True,disk=disk_guard()))
    print('PIXEL_CRISP_VIDEO_COMPLETE',flush=True)

if __name__=='__main__':main()
