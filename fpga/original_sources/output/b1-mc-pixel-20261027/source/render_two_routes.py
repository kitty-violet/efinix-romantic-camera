"""Same original continuous frames: original/V14/enlargement/local-block trials."""
from core import *
import evaluation_utils as ev
from network_ab.training.media import decode_frames
from network_ab.training.fpga_video_resize import area_3to2
from evaluate_pixel import mean_metrics
import imageio_ffmpeg

@torch.inference_mode()
def main():
    setup();disk_guard();dest=OUT/'two-routes-video';dest.mkdir(exist_ok=True)
    coeffs=dict(large_pixel=OUT/'exports/flat-pixel-v2-trial/coefficients-padded.bin',
                local_blocks=OUT/'exports/mc-local-blocks-trial/coefficients-padded.bin')
    engines={k:IntegerEngine(p.read_bytes()) for k,p in coeffs.items()};engines['v14']=IntegerEngine(DEPLOY.read_bytes())
    for p in coeffs.values():
        v=json.loads((p.parent/'verification.json').read_text('utf-8'));assert sha(p)==v['sha256_padded']
    manifest=json.loads((ROOT/'output/b1-ink-wash-20261026/independent_test/manifest.json').read_text('utf-8'))
    clips=[r for r in manifest['clips'] if r['id'] in ['outdoor','station']]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);begin=time.perf_counter();reports=[]
    for clip in clips:
        scene='indoor' if clip['id']=='station' else 'outdoor';cache=OUT/'cache'/('two-'+scene);cache.mkdir(parents=True,exist_ok=False)
        paths={k:[] for k in ['large_pixel','local_blocks','comparison']};hashes={k:[] for k in ['source',*engines]}
        ids=[];pts=[];metrics=[];previous=None
        for index,stamp,frame in decode_frames(clip['source'],seconds=3,hlg_exposure_stops=clip['exposure_stops']):
            if stamp>=3:break
            if time.perf_counter()-begin>900:raise TimeoutError('15 minute preview host budget')
            if ids:assert index==ids[-1]+1 and stamp>pts[-1]
            a=area_3to2(frame);x=tensor(a)[None]*128
            images=dict(source=a,**{k:rgb(net(x)/128) for k,net in engines.items()})
            for k,v in images.items():hashes[k].append(hashlib.sha256(v.tobytes()).hexdigest())
            row=dict(frame=index,pts=stamp)
            if previous is not None and index%5==0:
                mx,my,valid,static,_=ev.flow_mask(previous['source'],a)
                for k in images:row[k]=ev.temporal(previous[k],images[k],mx,my,valid,static)
            for k in ['large_pixel','local_blocks']:
                p=cache/f'{k}_{index:04d}.png';Image.fromarray(images[k]).save(p,compress_level=1);paths[k].append(p)
            board=Image.new('RGB',(2560,394),'#111722');draw=ImageDraw.Draw(board)
            for j,(k,label) in enumerate([('source','原图'),('v14','旧V14 / 纯B1整数'),('large_pixel','像素放大近似 / 纯B1整数'),('local_blocks','局部方块轮廓 / 纯B1整数')]):
                board.paste(Image.fromarray(images[k]).resize((640,360),Image.Resampling.NEAREST),(640*j,34))
                draw.text((640*j+7,6),label,font=font,fill='white')
            p=cache/f'compare_{index:04d}.png';board.save(p,compress_level=1);paths['comparison'].append(p)
            if index in [0,30,60]:board.save(dest/f'{scene}_{index:04d}.jpg',quality=93)
            ids.append(index);pts.append(stamp);metrics.append(row);previous=images
            if index%30==0:print('TWO_ROUTE_FRAME',scene,index,round(time.perf_counter()-begin,1),flush=True);disk_guard()
        assert len(ids)>85 and ids==list(range(len(ids)))
        video={k:ev.encode_vfr(cache,v,pts,dest/f'{scene}_{k}.mp4',imageio_ffmpeg.get_ffmpeg_exe()) for k,v in paths.items()}
        record=dict(id=scene,source=clip['source'],source_sha256=sha(clip['source']),frames=len(ids),frame_ids=ids,pts_seconds=pts,
            duration_seconds=pts[-1]-pts[0]+float(np.median(np.diff(pts))),
            coefficients={k:dict(path=str(p),sha256=sha(p)) for k,p in coeffs.items()},v14_sha256=sha(DEPLOY),
            video=video,raw_unique_frames={k:len(set(v)) for k,v in hashes.items()},
            consecutive_duplicates={k:sum(a==b for a,b in zip(v,v[1:])) for k,v in hashes.items()},
            temporal_summary={k:mean_metrics(metrics,k) for k in images},metrics=metrics,
            extra_inference_operators=False,inference_history_or_flow=False,software_playback_fps_not_board_fps=True,
            exact6_grid_achieved=False,semantic3D_MC_achieved=False)
        save_json(dest/f'{scene}_report.json',record)
        resolved=cache.resolve();assert resolved.parent==(OUT/'cache').resolve() and resolved.name in ['two-outdoor','two-indoor']
        shutil.rmtree(resolved)
        reports.append({k:v for k,v in record.items() if k not in ['metrics','pts_seconds','frame_ids']})
        save_json(dest/'summary.json',dict(clips=reports,all_two_complete=len(reports)==2,elapsed_seconds=time.perf_counter()-begin,disk=disk_guard()))
        print('TWO_ROUTE_CLIP_DONE',scene,len(ids),flush=True)
    print('TWO_ROUTES_COMPLETE',flush=True)

if __name__=='__main__':main()
