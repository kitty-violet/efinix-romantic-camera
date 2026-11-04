"""Minimal chroma-fragment edit: same source timestamps, actual BIN only."""
from core import *
import evaluation_utils as ev
from evaluate_pixel import mean_metrics
from evaluate_r3_r4 import content
from network_ab.training.media import decode_frames
from network_ab.training.fpga_video_resize import area_3to2
import imageio_ffmpeg

@torch.inference_mode()
def main():
    setup(271175); begin=time.perf_counter(); d=OUT/'pixel-r28-video-review'; d.mkdir(exist_ok=True)
    paths={k:OUT/'exports'/v/'coefficients-padded.bin' for k,v in
        [('original_A','pixel-user-curve-A'),('candidate_r28','pixel-A-fringe-r28')]}
    engines={k:IntegerEngine(p.read_bytes()) for k,p in paths.items()}
    clips=json.loads((ROOT/'output/b1-ink-wash-20261026/independent_test/manifest.json').read_text('utf-8'))['clips']
    save_json(d/'binding.json',dict(models={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},
        training=False,inference_postprocessing=False,previous_A_still_selected=True,script_sha256=sha(__file__)))
    reports=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    for scene in ['corridor','outdoor']:
        clip=next(c for c in clips if c['id']==scene); cache=OUT/'cache'/('pixel-r28-'+scene);cache.mkdir(exist_ok=True)
        assert not list(cache.iterdir()); paths_out={'native':[],'comparison':[]}; pts=[];rows=[];previous=None
        unique={k:set() for k in ['source',*engines]};differences=[]
        for index,stamp,frame in decode_frames(clip['source'],seconds=3,hlg_exposure_stops=clip['exposure_stops']):
            if stamp>=3:break
            if time.perf_counter()-begin>900:raise TimeoutError('bounded 15 minute review')
            assert index==len(pts)
            a=area_3to2(frame);images={'source':a,**{k:rgb(n(tensor(a)[None]*128)/128) for k,n in engines.items()}}
            for k,z in images.items():unique[k].add(hashlib.sha256(z.tobytes()).hexdigest())
            row={'frame':index,'pts':stamp}
            if previous is not None and index%5==0:
                mx,my,valid,static,_=ev.flow_mask(previous['source'],a)
                row.update({k:ev.temporal(previous[k],z,mx,my,valid,static) for k,z in images.items()})
            if index%30==0:row['content']={k:content(a,images[k]) for k in engines}
            delta=np.abs(images['original_A'].astype(float)-images['candidate_r28'])
            differences.append(dict(rgb_mae=float(delta.mean()),changed_fraction=float(np.any(delta>0,2).mean())))
            panel=Image.new('RGB',(1280,750),'#151a21');draw=ImageDraw.Draw(panel)
            for j,(k,title) in enumerate([('original_A','原曲线A / 原块面'),('candidate_r28','R28 / 仅清理孤立杂色，未定稿')]):
                panel.paste(Image.fromarray(images[k]).resize((640,360),Image.Resampling.NEAREST),(j*640,34))
                draw.text((j*640+8,6),title,font=font,fill='white')
                crop=images[k][120:476,384:1024]
                panel.paste(Image.fromarray(crop),(j*640,394))
            for k,im in [('native',Image.fromarray(images['candidate_r28'])),('comparison',panel)]:
                p=cache/f'{k}_{index:04d}.png';im.save(p,compress_level=1);paths_out[k].append(p)
            if index==0:panel.save(d/f'{scene}_comparison.jpg',quality=95)
            previous=images;pts.append(stamp);rows.append(row)
            if index%30==0:print('PIXEL_R28_REVIEW',scene,index,flush=True);disk_guard()
        assert len(pts)>=90 and all(len(v)==len(pts) for v in unique.values())
        videos={k:ev.encode_vfr(cache,v,pts,d/f'{scene}_{k}.mp4',imageio_ffmpeg.get_ffmpeg_exe()) for k,v in paths_out.items()}
        report=dict(scene=scene,frames=len(pts),pts=pts,source=clip['source'],source_sha256=sha(clip['source']),
            temporal={k:mean_metrics(rows,k) for k in images},content=[r['content'] for r in rows if 'content' in r],
            differences=differences,unique_frames={k:len(v) for k,v in unique.items()},videos=videos,all_frames_preserved=True)
        reports.append(report);save_json(d/f'{scene}_report.json',report)
        owned=list(cache.iterdir());assert cache.resolve().parent==(OUT/'cache').resolve()
        assert all(p.is_file() and p.resolve().parent==cache.resolve() and p.suffix in ['.png','.ffconcat'] for p in owned)
        for p in owned:p.unlink()
        cache.rmdir()
    save_json(d/'summary.json',dict(complete=True,reports=reports,seconds=time.perf_counter()-begin,disk=disk_guard()))
    print('PIXEL_R28_COMPLETE',flush=True)

if __name__=='__main__':main()
