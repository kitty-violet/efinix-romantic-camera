"""Curve-A layer-shape and MC research review, same native frames and PTS."""
from core import *
import evaluation_utils as ev
from evaluate_pixel import mean_metrics
from evaluate_r3_r4 import content,controls
from r16_pixel_layer_geometry import cell_audit
from network_ab.training.media import decode_frames
from network_ab.training.fpga_video_resize import area_3to2
import imageio_ffmpeg

@torch.inference_mode()
def main():
    setup(271125);dest=OUT/'pixel-a-shapes-and-mc-review';dest.mkdir(exist_ok=True);disk_guard();begin=time.perf_counter()
    paths=dict(v14=DEPLOY,old_A=OUT/'exports/pixel-user-curve-A/coefficients-padded.bin',
        pixel_r21=OUT/'exports/pixel-A-square-r21/coefficients-padded.bin',
        pixel_r23=OUT/'exports/pixel-A-square-r23/coefficients-padded.bin',
        mc_r20=OUT/'exports/mc-structure-support-r20/coefficients-padded.bin')
    for key in ['pixel_r21','pixel_r23','mc_r20']:
        verification=json.loads((paths[key].parent/'verification.json').read_text('utf-8'))
        assert verification['sha256_padded']==sha(paths[key])
        assert all(v['max_error']==0 for v in verification['full720_qat_vs_decoded_integer_nodes'].values())
    engines={key:IntegerEngine(p.read_bytes()) for key,p in paths.items()}
    save_json(dest/'binding.json',dict(coefficients={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},
        chosen_tone_curve='A',saturation_relative=1.3,inference_filter=False,inference_flow=False,
        MC_style_accepted=False,pixel_geometry_visual_accepted=False,source_script_sha256=sha(__file__)))
    ctr=controls(engines);save_json(dest/'colour_controls.json',ctr)
    for key in ['pixel_r21','pixel_r23']:
        assert np.max(np.abs(np.array(ctr[key]['gray_curve_mean'])-ctr['old_A']['gray_curve_mean']))==0
    manifest=json.loads((ROOT/'output/b1-ink-wash-20261026/independent_test/manifest.json').read_text('utf-8'))
    clips=sorted(manifest['clips'],key=lambda c:['station','outdoor','dusk','corridor','park','foliage'].index(c['id']))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);reports=[]
    for clip in clips:
        scene=clip['id'];duration=3 if scene in ['outdoor','station'] else 1
        cache=OUT/'cache'/('a-layer-shapes-'+scene);cache.mkdir(parents=True,exist_ok=True)
        # Reuse our own interrupted render cache. Never recursively remove a
        # directory to restart. Reject any unexpected entry before overwriting.
        assert cache.resolve().parent==(OUT/'cache').resolve() and not cache.is_symlink()
        allowed=('pixel_r21_','pixel_r23_','mc_r20_','comparison_','crop_',scene+'_')
        assert all(p.is_file() and p.name.startswith(allowed) and p.suffix in ['.png','.ffconcat'] for p in cache.iterdir())
        video_keys=['pixel_r21','pixel_r23','mc_r20','comparison']+(['ceiling_crop'] if scene=='station' else [])
        pictures={key:[] for key in video_keys};ids=[];pts=[];records=[];prev=None;hashes={key:[] for key in ['source',*engines]}
        for index,stamp,full in decode_frames(clip['source'],seconds=duration,hlg_exposure_stops=clip['exposure_stops']):
            if stamp>=duration:break
            if time.perf_counter()-begin>1500:raise TimeoutError('25 minute review ceiling')
            if ids:assert index==ids[-1]+1 and stamp>pts[-1]
            source=area_3to2(full);x=tensor(source)[None]*128;images=dict(source=source,**{k:rgb(n(x)/128) for k,n in engines.items()})
            for key,a in images.items():hashes[key].append(hashlib.sha256(a.tobytes()).hexdigest())
            row=dict(frame=index,pts=stamp)
            if prev is not None and index%5==0:
                mx,my,valid,static,_=ev.flow_mask(prev['source'],source)
                for key,a in images.items():row[key]=ev.temporal(prev[key],a,mx,my,valid,static)
            if index%30==0:
                row['content']={key:content(source,images[key]) for key in ['old_A','pixel_r21','pixel_r23','mc_r20']}
                row['cell_geometry']={key:cell_audit(images[key]) for key in ['old_A','pixel_r21','pixel_r23']}
                assert all(v['max_within_cell_rgb_range']==0 for v in row['cell_geometry'].values())
            canvas=Image.new('RGB',(1920,788),'#111722');draw=ImageDraw.Draw(canvas)
            tiles=[('source','原图'),('v14','旧V14 / 纯B1整数'),('old_A','曲线A / 原色层形状'),
                ('pixel_r21','曲线A / 方块色层 R21'),('pixel_r23','曲线A / 中间色层加强 R23'),('mc_r20','MC结构支持研究候选 / 未达标')]
            for j,(key,label) in enumerate(tiles):
                xx=j%3*640;yy=j//3*394;canvas.paste(Image.fromarray(images[key]).resize((640,360),Image.Resampling.NEAREST),(xx,yy+34));draw.text((xx+7,yy+6),label,font=font,fill='white')
            for key in ['pixel_r21','pixel_r23','mc_r20']:
                p=cache/f'{key}_{index:04d}.png';Image.fromarray(images[key]).save(p,compress_level=1);pictures[key].append(p)
            p=cache/f'comparison_{index:04d}.png';canvas.save(p,compress_level=1);pictures['comparison'].append(p)
            if scene=='station':
                crop=(slice(72,232),slice(200,456));cc=Image.new('RGB',(1536,356),'#111722');dd=ImageDraw.Draw(cc)
                for j,(key,label) in enumerate([('source','原图局部'),('old_A','曲线A原版 / 同一局部'),('pixel_r23','曲线A新候选R23 / 同一局部')]):
                    cc.paste(Image.fromarray(images[key][crop]).resize((512,320),Image.Resampling.NEAREST),(j*512,35));dd.text((j*512+5,5),label,font=font,fill='white')
                assert cc.width%2==0 and cc.height%2==0
                p=cache/f'crop_{index:04d}.png';cc.save(p,compress_level=1);pictures['ceiling_crop'].append(p)
                if index in [0,30,60]:cc.save(dest/f'{scene}_crop_{index:04d}.png')
            if index in [0,30,60]:canvas.save(dest/f'{scene}_{index:04d}.jpg',quality=94)
            records.append(row);ids.append(index);pts.append(stamp);prev=images
            if index%30==0:print('A_SHAPE_REVIEW_FRAME',scene,index,round(time.perf_counter()-begin,1),flush=True);disk_guard()
        assert ids==list(range(len(ids))) and len(ids)>duration*28
        assert all(len(set(h))==len(ids) for h in hashes.values())
        videos={key:ev.encode_vfr(cache,p,pts,dest/f'{scene}_{key}.mp4',imageio_ffmpeg.get_ffmpeg_exe()) for key,p in pictures.items()}
        report=dict(id=scene,source=clip['source'],source_sha256=sha(clip['source']),frames=len(ids),
            seconds=pts[-1]-pts[0]+float(np.median(np.diff(pts))),raw_unique_frames={k:len(set(v)) for k,v in hashes.items()},
            temporal={key:mean_metrics(records,key) for key in images},video=videos,
            content=[r['content'] for r in records if 'content' in r],cell_geometry=[r['cell_geometry'] for r in records if 'cell_geometry' in r],
            post_filter=False,inference_flow=False,original_frame_rate_preserved=True,no_drop_or_repeated_frames=True,
            style_pass_claim=False,temporal_superiority_claim=False)
        save_json(dest/f'{scene}_report.json',dict(summary=report,frame_ids=ids,pts_seconds=pts,records=records));reports.append(report)
        resolved=cache.resolve();assert resolved.parent==(OUT/'cache').resolve() and resolved.name=='a-layer-shapes-'+scene
        # Remove only the enumerated, checked files belonging to this clip,
        # then remove the now-empty directory. No broad recursive deletion.
        owned=list(resolved.iterdir())
        assert all(p.is_file() and p.resolve().parent==resolved and p.name.startswith(allowed) and p.suffix in ['.png','.ffconcat'] for p in owned)
        for p in owned:p.unlink()
        resolved.rmdir();save_json(dest/'summary.json',dict(clips=reports,all_six_complete=len(reports)==6,elapsed=time.perf_counter()-begin,disk=disk_guard()))
        print('A_SHAPE_REVIEW_CLIP_DONE',scene,len(ids),flush=True)
    print('A_SHAPE_MC_REVIEW_COMPLETE',flush=True)

if __name__=='__main__':main()
