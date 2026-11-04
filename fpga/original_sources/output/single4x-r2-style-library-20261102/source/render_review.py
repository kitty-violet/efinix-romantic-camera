"""Four exact source clips, five seconds each, for every exported new style."""
from pathlib import Path
import sys,time,json,hashlib,shutil,argparse,copy
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import library as l
from library import ROOT,D,OLD,torch,tx,Teacher,PackedEngine,save_json,sha,guard
from group_engine import GroupEngine
sys.path.insert(0,str(OLD/'source'))
from build_four_scene_acceptance import StreamEncoder
from finalize_four_scene_acceptance import reel
from network_ab.training.media import decode_frames
from network_ab.training.fpga_video_resize import area_3to2
import evaluation_utils as ev

R=D/'review';FONT='C:/Windows/Fonts/msyh.ttc'
def rgb_batch(t):return np.ascontiguousarray((t.permute(0,2,3,1).cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8))
def hash_array(a):return hashlib.sha256(a.tobytes()).hexdigest()
def image_metrics(a,b):
    e=a.astype(np.float32)-b.astype(np.float32);mse=float((e*e).mean())
    return dict(mae=float(np.abs(e).mean()),psnr_db=float(10*np.log10(255**2/max(mse,1e-9))))

def tiny_islands(pic):
    # Evaluation only: connected colour regions on the shared four-pixel grid.
    # A component of <=2 cells covers <=32 native pixels; no output is edited.
    import cv2
    q=pic[2::4,2::4].astype(np.int32)//32;labels=q[...,0]*64+q[...,1]*8+q[...,2]
    small_area=components=0
    for colour in np.unique(labels):
        mask=np.ascontiguousarray(labels==colour,dtype=np.uint8)
        n,_,stats,_=cv2.connectedComponentsWithStats(mask,8)
        area=stats[1:,cv2.CC_STAT_AREA];small=area[area<=2]
        small_area+=int(small.sum());components+=len(small)
    return dict(quantization='RGB each 32 levels, cell4x4, connected8',tiny_cells_fraction=small_area/labels.size,tiny_components=components)

def static_check(entries,sources):
    rows=[]
    for entry in entries:
        key=entry['id'];folder=D/'models'/key;engine=PackedEngine(folder);teacher=Teacher(entry)
        ckpath=folder/('selected.pt' if (folder/'selected.pt').exists() else 'qat.pt')
        ck=torch.load(ckpath,map_location='cpu',weights_only=True);model=l.StyleModel().cuda();model.load_state_dict(ck['model']);model.residual=ck['residual'];model.qat=True;model.eval()
        contact=Image.new('RGB',(1440,1200),'#111924');font=ImageFont.truetype(FONT,18);records=[]
        for row,s in enumerate(sources):
            n=len(s['pts'])//2;path=OLD/'acceptance-four-scenes-5s-20261029/originals'/f"{s['scene']}_{n:03d}.png"
            a=np.asarray(Image.open(path).convert('RGB'));assert hash_array(a)==s['raw_input_frame_sha256'][n]
            x=tx(a[None]);out=engine(x)
            with torch.inference_mode():assert torch.equal(out,model(x)),(key,s['scene'],'checkpoint/bin')
            b=rgb_batch(out)[0];t=rgb_batch(teacher(x))[0];m=image_metrics(t,b)
            for prefix,pic in [('old',t),('new',b)]:
                lum=pic.astype(float).mean(2);m[prefix+'_black_fraction']=float((lum<32).mean());m[prefix+'_white_fraction']=float((lum>224).mean())
                m[prefix+'_mean_abs_edge']=float((np.abs(np.diff(lum,axis=0)).mean()+np.abs(np.diff(lum,axis=1)).mean())/2)
                m[prefix+'_tiny_islands']=tiny_islands(pic)
            if key=='PIXEL_R39':
                cells=b.reshape(180,4,320,4,3);assert np.all(cells.max((1,3))==cells.min((1,3)))
            if key in ['INK','BW_LINE']:assert np.array_equal(b[...,0],b[...,1]) and np.array_equal(b[...,1],b[...,2])
            for col,(label,pic) in enumerate([('原图',a),('旧风格 / 教师参考' if key=='BW_LINE' else '旧风格整数',t),('新结构整数',b)]):
                draw=ImageDraw.Draw(contact);draw.text((col*480+4,row*300+3),s['label']+' / '+label,font=font,fill='white')
                contact.paste(Image.fromarray(pic).resize((480,270),Image.Resampling.BOX),(col*480,row*300+29))
            records.append(dict(scene=s['scene'],metrics=m))
        outfolder=R/'static'/key;outfolder.mkdir(parents=True,exist_ok=True);contact.save(outfolder/'四场景对比.jpg',quality=96)
        result=dict(style_id=key,source_sha256=entry['source_sha256'],coefficient_sha256=sha(folder/'coefficients-padded.bin'),full720_checkpoint_vs_bin_exact=True,records=records,visual_acceptance=False)
        save_json(outfolder/'verification.json',result);rows.append(result);print('STATIC_PASS',key,[round(x['metrics']['mae'],2) for x in records],flush=True)
        del engine,teacher,model;torch.cuda.empty_cache()
    return rows

def load_source(source):
    frames=[];indices=source['frame_indices']
    for n,(index,stamp,a) in enumerate(decode_frames(source['source'],indices=indices,hlg_exposure_stops=source['exposure_stops'])):
        assert index==indices[n] and abs(stamp-source['absolute_pts'][n])<1e-8
        a=area_3to2(a);assert hash_array(a)==source['raw_input_frame_sha256'][n];frames.append(a)
    assert len(frames)==source['frames'] and len(frames)>=150
    return frames

def run_group(entries,source,frames):
    scene=source['scene'];pts=source['pts'];started=time.perf_counter()
    pending=[e for e in entries if not (R/'native'/e['id']/(scene+'.json')).exists()]
    if not pending:return
    new=GroupEngine(folders=[D/'models'/e['id'] for e in pending])
    old=GroupEngine(entries=pending) if pending[0]['id']!='BW_LINE' else Teacher(pending[0])
    cached=all((l.ARCHIVE/'styles'/e['id']/'videos'/(scene+'.json')).exists() for e in pending)
    captures=[];old_reports=[]
    if cached:
        import cv2
        for e in pending:
            p=l.ARCHIVE/'styles'/e['id']/'videos'/(scene+'.json');meta=json.loads(p.read_text('utf-8'));movie=p.with_suffix('.mp4')
            assert meta['coefficient_sha256']==e['source_sha256'] and meta['input_frame_sha256']==source['raw_input_frame_sha256']
            assert meta['pts']==pts and sha(movie)==meta['video']['sha256']
            cap=cv2.VideoCapture(str(movie),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]) if hasattr(cv2,'CAP_PROP_N_THREADS') else cv2.VideoCapture(str(movie))
            assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(frames)
            captures.append(cap);old_reports.append(meta)
    writers={};comparisons={};records={};metric_lists={};hashes={};oldhash={};temporal={};previous={};inputs=[];panels={}
    font=ImageFont.truetype(FONT,18)
    flows={i:ev.flow_mask(frames[i-1],frames[i])[:4] for i in range(1,len(frames)) if i%30==0}
    exact_indices={0,len(frames)//2,len(frames)-1}|set(flows)|{i-1 for i in flows}
    for e in pending:
        key=e['id'];folder=R/'native'/key;folder.mkdir(parents=True,exist_ok=True)
        # Incomplete media are our own unfinalized prefix, never old archives.
        for p in [folder/(scene+'.mp4'),R/'comparisons'/key/(scene+'.mp4')]:
            if p.exists():p.unlink()
        writers[key]=StreamEncoder(folder/(scene+'.mp4'),pts)
        comparisons[key]=StreamEncoder(R/'comparisons'/key/(scene+'.mp4'),pts,1440,300)
        metric_lists[key]=[];hashes[key]=[];oldhash[key]=[];temporal[key]=[]
        panel=Image.new('RGB',(1440,300),'#111924');pd=ImageDraw.Draw(panel)
        for col,label in enumerate(['原图','旧风格视频' if cached else '旧风格','新结构 / INT8']):
            pd.text((col*480+4,3),label+' / '+e.get('current_display_name',e['name']),font=font,fill='white')
        panels[key]=panel
    # Prove grouped full-resolution pixels equal the individual BIN engine.
    first=tx(frames[0][None]);grouped=new(first)
    for i,e in enumerate(pending):assert torch.equal(grouped[i:i+1],PackedEngine(D/'models'/e['id'])(first))
    for n,a in enumerate(frames):
        guard();assert time.perf_counter()-started<1200,'20 minute cap per 4-style source group'
        x=tx(a[None]);out=rgb_batch(new(x));inputs.append(hash_array(a))
        if cached:
            decoded=[]
            for cap in captures:
                ok,bgr=cap.read();assert ok;decoded.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            visual_target=np.stack(decoded)
            if n in exact_indices:
                target=rgb_batch(old(x))
                for i,meta in enumerate(old_reports):assert hash_array(target[i])==meta['raw_output_frame_sha256'][n]
            else:target=visual_target
        else:target=visual_target=rgb_batch(old(x))
        small_input=Image.fromarray(a).resize((480,270),Image.Resampling.BOX)
        for i,e in enumerate(pending):
            key=e['id'];b=out[i];t=target[i];writers[key].write(b)
            hashes[key].append(hash_array(b));oldhash[key].append(hash_array(t) if not cached or n in exact_indices else old_reports[i]['raw_output_frame_sha256'][n])
            if not cached or n in exact_indices:
                m=image_metrics(t,b);m['old_comparison_is_exact_raw']=True
                m['new_vs_input_mae']=float(np.abs(b.astype(np.float32)-a).mean());m['old_vs_input_mae']=float(np.abs(t.astype(np.float32)-a).mean());metric_lists[key].append(m)
            panel=panels[key].copy();panel.paste(small_input,(0,29))
            for col,pic in [(1,visual_target[i]),(2,b)]:panel.paste(Image.fromarray(pic).resize((480,270),Image.Resampling.BOX),(col*480,29))
            comparisons[key].write(np.asarray(panel))
            if n in [0,len(frames)//2,len(frames)-1]:
                folder=R/'stills'/key;folder.mkdir(parents=True,exist_ok=True)
                Image.fromarray(b).save(folder/f'{scene}_{n:03d}_new.png')
                Image.fromarray(t).save(folder/f'{scene}_{n:03d}_old.png')
                if n==len(frames)//2:panel.save(folder/(scene+'_comparison.jpg'),quality=96)
            if n in flows:
                temporal[key].append(dict(frame=n,pts=pts[n],new=ev.temporal(previous[key][0],b,*flows[n]),old=ev.temporal(previous[key][1],t,*flows[n])))
            previous[key]=(b,t)
            if key=='PIXEL_R39':
                cells=b.reshape(180,4,320,4,3);assert np.all(cells.max((1,3))==cells.min((1,3))),'new pixel blocks lost 4x4 flatness'
            if key in ['INK','BW_LINE']:assert np.array_equal(b[...,0],b[...,1]) and np.array_equal(b[...,1],b[...,2])
        if n%50==0:print('VIDEO_FRAMES',scene,[e['id'] for e in pending],n,'sec',round(time.perf_counter()-started,1),flush=True)
    assert inputs==source['raw_input_frame_sha256']
    for e in pending:
        key=e['id'];native=writers[key].close(unique=key!='BW_LINE');compare=comparisons[key].close()
        if key!='BW_LINE':assert len(set(hashes[key]))==len(frames)
        exact_metrics=[m for m in metric_lists[key] if m['old_comparison_is_exact_raw']]
        r=dict(style_id=key,scene=scene,label=source['label'],frames=len(frames),unique_new_raw_frames=len(set(hashes[key])),duration_seconds=source['duration_seconds'],pts=pts,source_indices=source['frame_indices'],source_sha256=source['source_sha256'],input_sha256=inputs,new_raw_sha256=hashes[key],old_raw_sha256=oldhash[key],coefficient_sha256=sha(D/'models'/key/'coefficients-padded.bin'),teacher_sha256=e['source_sha256'],native_video=native,comparison_video=compare,grouped_vs_individual_full720_exact=True,mean_metrics={k:float(np.mean([m[k] for m in exact_metrics])) for k in exact_metrics[0] if k!='old_comparison_is_exact_raw'},exact_raw_metric_frames=len(exact_metrics),old_visual_reference='existing frozen same-input integer MP4 decoded; exact BIN spot checks and temporal pairs' if cached else 'fresh frozen source model integer output',temporal_records=temporal[key],temporal_flow_evaluation_only=True,no_inference_flow=True,no_post_filter=True,no_frame_drop=True,no_repeat_inserted=True,new_pixels_from_delivered_bin=True,pixel4x4_flatness_pass=True if key=='PIXEL_R39' else None,monochrome_exact=True if key in ['INK','BW_LINE'] else None)
        save_json(R/'native'/key/(scene+'.json'),r);records[key]=r
        print('VIDEO_STYLE_SCENE_DONE',key,scene,round(r['mean_metrics']['mae'],3),flush=True)
    del new,old;torch.cuda.empty_cache()
    for cap in captures:
        ok,_=cap.read();assert not ok;cap.release()
    return records

def finish_style(entry,sources):
    key=entry['id'];reports=[json.loads((R/'native'/key/(s['scene']+'.json')).read_text('utf-8')) for s in sources]
    (R/'reels').mkdir(exist_ok=True)
    native=R/'reels'/f'{key}_新结构_四场景各5秒.mp4'
    comparison=R/'reels'/f'{key}_原图旧版新版_四场景各5秒.mp4'
    if not native.exists():a,_=reel([R/'native'/key/(s['scene']+'.mp4') for s in sources],sources,native)
    else:
        from build_four_scene_acceptance import inspect_video
        pts=[];offset=0
        for s in sources:pts.extend([offset+t for t in s['pts']]);offset+=s['duration_seconds']
        a=inspect_video(native,pts)
    # Existing concat helper checks 720p, so concatenate comparisons with the
    # same timestamp construction and inspect their real 1440x300 geometry.
    if not comparison.exists():
        import subprocess
        from build_four_scene_acceptance import FF,inspect_video
        listpath=R/'reels'/f'{key}_comparison_concat.txt'
        listpath.write_text(''.join("file '"+str((R/'comparisons'/key/(s['scene']+'.mp4')).resolve()).replace('\\','/')+"'\n" for s in sources),encoding='utf-8')
        cmd=[FF,'-hide_banner','-loglevel','error','-nostdin','-y','-f','concat','-safe','0','-i',str(listpath),'-c','copy','-video_track_timescale','10000','-movflags','+faststart',str(comparison)]
        proc=subprocess.run(cmd,capture_output=True,timeout=90);assert proc.returncode==0,proc.stderr;listpath.unlink()
        pts=[];offset=0
        for s in sources:pts.extend([offset+t for t in s['pts']]);offset+=s['duration_seconds']
        b=inspect_video(comparison,pts,1440,300)
    else:b=dict(path=str(comparison),sha256=sha(comparison))
    contact=Image.new('RGB',(1440,1200),'#111924')
    for row,s in enumerate(sources):contact.paste(Image.open(R/'stills'/key/(s['scene']+'_comparison.jpg')),(0,row*300))
    contact.save(R/'stills'/key/'四场景对照.jpg',quality=96)
    save_json(R/'native'/key/'summary.json',dict(style_id=key,scenes=reports,reel=a,comparison_reel=b))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ids',nargs='*');parser.add_argument('--static-only',action='store_true');args=parser.parse_args();l.setup(20261102);guard();R.mkdir(exist_ok=True)
    entries=l.read_inventory();entries=[e for e in entries if not args.ids or e['id'] in args.ids]
    for e in entries:assert (D/'models'/e['id']/'training.json').exists(),e['id']
    sources=json.loads((OLD/'acceptance-four-scenes-5s-20261029/source_manifest.json').read_text('utf-8'))['scenes'];save_json(R/'source_manifest.json',dict(scenes=sources))
    if args.static_only:
        static_check(entries,sources);return
    groups=[entries[i:i+4] for i in range(0,len(entries)) if i%4==0]
    # Archived FP32 teacher has a different graph and is evaluated separately.
    normal=[e for e in entries if e['id']!='BW_LINE'];groups=[normal[i:i+4] for i in range(0,len(normal),4)]+[[e] for e in entries if e['id']=='BW_LINE']
    for source in sources:
        assert sha(source['source'])==source['source_sha256'];frames=load_source(source)
        for group in groups:
            run_group(group,source,frames)
            complete=[(e['id'],s['scene']) for e in entries for s in sources if (R/'native'/e['id']/(s['scene']+'.json')).exists()]
            save_json(R/'progress.json',dict(complete=len(complete)==4*len(entries),completed=len(complete),expected=4*len(entries),batch_style_ids=[e['id'] for e in entries],scope='this invocation only; final catalog verifies the full library',records=complete))
        del frames
    for e in entries:finish_style(e,sources)
    assert all(sha(e['source'])==e['source_sha256'] for e in entries)
    print('ALL_STYLE_VIDEOS_COMPLETE',len(entries)*4,flush=True)
if __name__=='__main__':main()
