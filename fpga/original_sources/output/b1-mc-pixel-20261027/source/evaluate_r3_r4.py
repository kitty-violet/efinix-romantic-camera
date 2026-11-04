"""Frozen revisions on main3s + extra1s held-out scene prefixes, original PTS."""
from core import *
import evaluation_utils as ev
from evaluate_pixel import mean_metrics
from r3_palette_blocks import geometry
from network_ab.training.media import decode_frames
from network_ab.training.fpga_video_resize import area_3to2
import imageio_ffmpeg
import argparse

def content(source,output):
    a=cv2.resize(ev.luminance(source),(64,36),interpolation=cv2.INTER_AREA)
    b=cv2.resize(ev.luminance(output),(64,36),interpolation=cv2.INTER_AREA)
    corr=float(np.corrcoef(a.flatten(),b.flatten())[0,1]) if a.std()>1 and b.std()>1 else None
    x=a[:,1:]-a[:,:-1];y=b[:,1:]-b[:,:-1]
    return dict(coarse_luma_correlation=corr,coarse_contrast_ratio=float(b.std()/max(.001,a.std())),
        coarse_edge_energy_ratio=float(np.abs(y).mean()/max(.001,np.abs(x).mean())))

@torch.inference_mode()
def controls(engines):
    a=torch.arange(256,device='cuda',dtype=torch.float32)[:,None,None,None].expand(-1,3,16,16)-128
    near=torch.full((16,16,3),120,dtype=torch.uint8).numpy();near[...,0]=123;near[...,2]=117
    results={}
    for k,net in engines.items():
        z=net(a)[:,:,8,8]+128
        n=net(tensor(near)[None]*128)[0,:,8,8]+128
        gray_chroma=float((z.max(1).values-z.min(1).values).max())
        results[k]=dict(gray_sweep_max_channel_difference=gray_chroma,
            near_gray_source=[123,120,117],near_gray_output=[float(v) for v in n.cpu()],
            near_gray_chroma=float(n.max()-n.min()),
            gray_curve_mean=[float(v) for v in z.mean(1).cpu()])
    return results

@torch.inference_mode()
def main(args):
    setup();disk_guard();dest=OUT/args.dest;dest.mkdir(exist_ok=True);begin=time.perf_counter()
    paths=dict(flat_r3=OUT/'exports'/args.flat_name/'coefficients-padded.bin',
        mc_r4=OUT/'exports'/args.mc_name/'coefficients-padded.bin',
        flat_previous=OUT/'exports/flat-pixel-v2-trial/coefficients-padded.bin',
        mc_previous=OUT/'exports/mc-local-blocks-trial/coefficients-padded.bin',v14=DEPLOY)
    if args.compact:paths={k:v for k,v in paths.items() if k in ['flat_r3','mc_r4','v14']}
    engines={k:IntegerEngine(p.read_bytes()) for k,p in paths.items()}
    save_json(dest/'binding.json',dict(coefficients={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},
        scripts={p.name:sha(p) for p in [Path(__file__),OUT/'source/r3_palette_blocks.py',OUT/'source/r4_box_projector.py',OUT/'source/fit_palette_revision.py']},
        input_policy='same fixed HLG-1EV and integer3:2 resize as earlier reports',no_inference_history=True,no_post_filter=True))
    save_json(dest/'colour_controls.json',controls(engines))
    manifest=json.loads((ROOT/'output/b1-ink-wash-20261026/independent_test/manifest.json').read_text('utf-8'))
    clips=sorted(manifest['clips'],key=lambda c:['outdoor','station','dusk','corridor','park','foliage'].index(c['id']))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);reports=[]
    for clip in clips:
        scene=clip['id'];duration=3 if scene in ['outdoor','station'] else 1
        cache=OUT/'cache'/(args.dest+'-'+scene);cache.mkdir(parents=True,exist_ok=False)
        ids=[];pts=[];records=[];hashes={k:[] for k in ['source',*engines]};previous=None
        pictures={k:[] for k in ['flat_r3','mc_r4','comparison']}
        for index,timestamp,full in decode_frames(clip['source'],seconds=duration,hlg_exposure_stops=clip['exposure_stops']):
            if timestamp>=duration:break
            if time.perf_counter()-begin>1200:raise TimeoutError('20 minute total review limit')
            if ids:assert index==ids[-1]+1 and timestamp>pts[-1]
            source=area_3to2(full);x=tensor(source)[None]*128
            images=dict(source=source,**{k:rgb(n(x)/128) for k,n in engines.items()})
            for k,z in images.items():hashes[k].append(hashlib.sha256(z.tobytes()).hexdigest())
            record=dict(frame=index,pts=timestamp)
            if previous is not None and index%5==0:
                mx,my,valid,static,_=ev.flow_mask(previous['source'],source)
                for k,z in images.items():record[k]=ev.temporal(previous[k],z,mx,my,valid,static)
            if index%30==0:
                record['geometry']={k:geometry(z) for k,z in images.items() if k!='source'}
                record['content']={k:content(source,z) for k,z in images.items() if k!='source'}
            canvas=Image.new('RGB',(1280 if args.compact else 1920,788),'#111722');draw=ImageDraw.Draw(canvas)
            tiles=[('source','原图'),('v14','旧V14 / 纯B1整数'),('flat_previous','上一轮像素学生'),
                   ('flat_r3','新大块像素R3 / 纯B1整数'),('mc_previous','上一轮局部方块学生'),('mc_r4','新方块投影R4 / 纯B1整数')]
            if args.compact:tiles=[('source','原图'),('v14','旧V14 / 纯B1整数'),('flat_r3',args.flat_label+' / 纯B1整数'),('mc_r4',args.mc_label+' / 纯B1整数')]
            for j,(k,label) in enumerate(tiles):
                cols=2 if args.compact else 3;xx=(j%cols)*640;yy=(j//cols)*394
                canvas.paste(Image.fromarray(images[k]).resize((640,360),Image.Resampling.NEAREST),(xx,yy+34))
                draw.text((xx+8,yy+6),label,font=font,fill='white')
            for k in ['flat_r3','mc_r4']:
                p=cache/f'{k}_{index:04d}.png';Image.fromarray(images[k]).save(p,compress_level=1);pictures[k].append(p)
            p=cache/f'compare_{index:04d}.png';canvas.save(p,compress_level=1);pictures['comparison'].append(p)
            if index in [0,30,60]:canvas.save(dest/f'{scene}_{index:04d}.jpg',quality=93)
            ids.append(index);pts.append(timestamp);records.append(record);previous=images
            if index%30==0:print('R34_FRAME',scene,index,round(time.perf_counter()-begin,1),flush=True);disk_guard()
        assert ids==list(range(len(ids))) and len(ids)>duration*28
        videos={k:ev.encode_vfr(cache,v,pts,dest/f'{scene}_{k}.mp4',imageio_ffmpeg.get_ffmpeg_exe()) for k,v in pictures.items()}
        assert all(len(set(v))==len(ids) for v in hashes.values())
        summary=dict(id=scene,source=clip['source'],source_sha256=sha(clip['source']),frames=len(ids),
            duration_seconds=pts[-1]-pts[0]+float(np.median(np.diff(pts))),temporal={k:mean_metrics(records,k) for k in images},
            raw_unique_frames={k:len(set(v)) for k,v in hashes.items()},video=videos,
            geometry=[r['geometry'] for r in records if 'geometry' in r],content=[r['content'] for r in records if 'content' in r],
            no_frame_drop_or_duplication=True,post_filter=False,inference_flow=False,board_fps=False)
        save_json(dest/f'{scene}_report.json',dict(summary=summary,frame_ids=ids,pts_seconds=pts,records=records))
        resolved=cache.resolve();assert resolved.parent==(OUT/'cache').resolve() and resolved.name==args.dest+'-'+scene
        shutil.rmtree(resolved);reports.append(summary)
        save_json(dest/'summary.json',dict(clips=reports,all_six_complete=len(reports)==6,elapsed_seconds=time.perf_counter()-begin,disk=disk_guard()))
        print('R34_CLIP_DONE',scene,len(ids),flush=True)
    print('R34_REVIEW_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mc-name',default='mc-projector-r4');p.add_argument('--dest',default='r3-r4-review');p.add_argument('--compact',action='store_true');p.add_argument('--mc-label',default='新方块投影R4')
    p.add_argument('--flat-name',default='flat-shared-r3');p.add_argument('--flat-label',default='大块像素R3')
    main(p.parse_args())
