"""Same-frame 3s MC regression review; ordinary decoded B1 BIN outputs only."""
from core import *
import evaluation_utils as ev
from evaluate_pixel import mean_metrics
from evaluate_r3_r4 import content
from network_ab.training.media import decode_frames
from network_ab.training.fpga_video_resize import area_3to2
import imageio_ffmpeg

@torch.inference_mode()
def main():
    setup(271160);begin=time.perf_counter();d=OUT/'mc-latest-r25-review';d.mkdir(exist_ok=True);disk_guard()
    paths=dict(early_r9=OUT/'r9-cuboid-volume/dark-large16/initial.bin',
        previous_r20=OUT/'exports/mc-structure-support-r20/coefficients-padded.bin',
        current_r25=OUT/'exports/mc-geometry-only-r25/coefficients-padded.bin')
    for key in ['previous_r20','current_r25']:
        v=json.loads((paths[key].parent/'verification.json').read_text('utf-8'))
        assert sha(paths[key])==v['sha256_padded']
        assert all(a['max_error']==0 for a in v['full720_qat_vs_decoded_integer_nodes'].values())
    engines={key:IntegerEngine(p.read_bytes()) for key,p in paths.items()}
    save_json(d/'binding.json',dict(models={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},
        native_resolution=[1280,720],inference_extra_filters=False,inference_flow=False,
        inference_history=False,MC_style_accepted=False,purpose='visual regression diagnosis, not a new trained model',
        source_script_sha256=sha(__file__)))
    clips=json.loads((ROOT/'output/b1-ink-wash-20261026/independent_test/manifest.json').read_text('utf-8'))['clips']
    selection=json.loads((OUT/'data-quality/selection.json').read_text('utf-8'))
    assert selection['indoor_review_id']=='corridor' and selection['exclude_old_station_as_primary_review']
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);reports=[]
    for scene in ['corridor','outdoor']:
        clip=next(c for c in clips if c['id']==scene);cache=OUT/'cache'/('latest-mc-'+scene);cache.mkdir(exist_ok=True)
        assert cache.resolve().parent==(OUT/'cache').resolve() and not cache.is_symlink()
        assert not list(cache.iterdir()),'Preserve an interrupted render instead of deleting unknown files'
        pictures=dict(current_r25=[],comparison=[]);pts=[];ids=[];rows=[];hashes={k:[] for k in ['source',*engines]};previous=None
        for index,stamp,frame in decode_frames(clip['source'],seconds=3,hlg_exposure_stops=clip['exposure_stops']):
            if stamp>=3:break
            if time.perf_counter()-begin>900:raise TimeoutError('15 minute bounded two-clip MC review')
            assert index==len(ids)
            a=area_3to2(frame);x=tensor(a)[None]*128
            images=dict(source=a,**{k:rgb(n(x)/128) for k,n in engines.items()})
            for k,z in images.items():hashes[k].append(hashlib.sha256(z.tobytes()).hexdigest())
            row=dict(frame=index,pts=stamp)
            if previous is not None and index%5==0:
                mx,my,valid,static,_=ev.flow_mask(previous['source'],a)
                row.update({k:ev.temporal(previous[k],z,mx,my,valid,static) for k,z in images.items()})
            if index%30==0:
                row['content']={k:content(a,images[k]) for k in engines}
            canvas=Image.new('RGB',(1280,788),'#111722');draw=ImageDraw.Draw(canvas)
            tiles=[('source','原图'),('early_r9','早期R9 / 亮度挤出'),
                ('previous_r20','前版R20 / 结构支持'),('current_r25','当前R25 / 几何支路约束，仍未达标')]
            for j,(k,label) in enumerate(tiles):
                xx=j%2*640;yy=j//2*394
                canvas.paste(Image.fromarray(images[k]).resize((640,360),Image.Resampling.NEAREST),(xx,yy+34))
                draw.text((xx+7,yy+6),label,font=font,fill='white')
            p=cache/f'native_{index:04d}.png';Image.fromarray(images['current_r25']).save(p,compress_level=1);pictures['current_r25'].append(p)
            p=cache/f'comparison_{index:04d}.png';canvas.save(p,compress_level=1);pictures['comparison'].append(p)
            if index in [0,30,60]:
                canvas.save(d/f'{scene}_comparison_{index:04d}.jpg',quality=95)
                Image.fromarray(images['current_r25']).save(d/f'{scene}_native_{index:04d}.png')
            previous=images;rows.append(row);ids.append(index);pts.append(stamp)
            if index%30==0:print('LATEST_MC_REVIEW_FRAME',scene,index,round(time.perf_counter()-begin,1),flush=True);disk_guard()
        assert len(ids)>=90 and all(len(set(v))==len(ids) for v in hashes.values())
        videos={k:ev.encode_vfr(cache,v,pts,d/f'{scene}_{k}.mp4',imageio_ffmpeg.get_ffmpeg_exe()) for k,v in pictures.items()}
        report=dict(scene=scene,source=clip['source'],source_sha256=sha(clip['source']),frames=len(ids),
            seconds=pts[-1]+float(np.median(np.diff(pts))),frame_ids=ids,pts_seconds=pts,
            raw_unique_frames={k:len(set(v)) for k,v in hashes.items()},
            temporal={k:mean_metrics(rows,k) for k in images},content=[r['content'] for r in rows if 'content' in r],
            videos=videos,records=rows,MC_style_pass=False,postfilter=False,all_frames_preserved=True)
        save_json(d/f'{scene}_report.json',report);reports.append(report)
        owned=list(cache.iterdir());resolved=cache.resolve()
        assert resolved.parent==(OUT/'cache').resolve() and resolved.name=='latest-mc-'+scene
        assert all(p.is_file() and p.resolve().parent==resolved and p.name.startswith(('native_','comparison_',scene+'_')) and p.suffix in ['.png','.ffconcat'] for p in owned)
        for p in owned:p.unlink()
        resolved.rmdir();save_json(d/'summary.json',dict(clips=reports,complete=len(reports)==2,elapsed=time.perf_counter()-begin,disk=disk_guard()))
        print('LATEST_MC_CLIP_DONE',scene,len(ids),flush=True)
    print('LATEST_MC_REVIEW_COMPLETE',flush=True)

if __name__=='__main__':main()
