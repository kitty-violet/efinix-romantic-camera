"""One 20s native reel/style: all four original scenes, no duplicate clip cache."""
from pathlib import Path
import sys,json,time,copy,hashlib,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import oneblock as o
from oneblock import torch,D,BASE,save_json,sha
import group_engine as ge
import render_review as old_review
from media import Encoder,ReuseEquivalentVideo,digest
from train_all import background

R=D/'review';FONT='C:/Windows/Fonts/msyh.ttc'

def rgb(t):return np.ascontiguousarray((t.permute(0,2,3,1).cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8))

def group(folders,version):
    previous=ge.PackedEngine;ge.PackedEngine=lambda f:o.Engine(f,version)
    try:return ge.GroupEngine(folders=folders)
    finally:ge.PackedEngine=previous

def feature_check(entry,x,a,b):
    key=entry['id'];m=o.Engine(D/'models'/key);t=o.Engine(entry['source_folder'],2);z=m(x,True);q=t(x,True);result={}
    if key=='PIXEL_R39':
        for name,pic in [('two',b),('one',a)]:
            c=pic.reshape(180,4,320,4,3);result[name+'_unequal_4x4_cells']=int((c.max((1,3))!=c.min((1,3))).any(2).sum());result[name+'_unique_grid_colours']=len(np.unique(pic[::4,::4].reshape(-1,3),axis=0))
        assert result['one_unequal_4x4_cells']==0
    if key=='INK':assert np.array_equal(a[...,0],a[...,1]) and np.array_equal(a[...,1],a[...,2]);result['monochrome_exact']=True
    if key=='OLD_FILM':
        import cv2
        for name,model,nodes in [('two',t,q),('one',m,z)]:
            f=nodes['up1_high'][0];bars=((f[5]>16)|(f[6]>16)).cpu().numpy().astype(np.uint8);delta=((nodes['rgb']/128-background(model,nodes)).mean(1)[0]*128).cpu().numpy()
            _,_,stats,_=cv2.connectedComponentsWithStats(bars,8);thin=stats[1:];thin=thin[(thin[:,cv2.CC_STAT_HEIGHT]<=8)&(thin[:,cv2.CC_STAT_WIDTH]>=8)]
            result[name]=dict(bar_coverage=float(bars.mean()),visible_positive_noise=float((delta>10).mean()),visible_negative_noise=float((delta< -10).mean()),thin_bar_count=len(thin),thin_bar_median_width=float(np.median(thin[:,cv2.CC_STAT_WIDTH])) if len(thin) else 0)
    del m,t,z,q;return result

@torch.inference_mode()
def main():
    o.setup(20261102);o.guard();entries=json.loads((D/'inventory.json').read_text('utf-8'))['styles']
    assert len(entries)==16 and all((D/'models'/e['id']/'training.json').exists() for e in entries)
    if (R/'complete.json').exists():print('R1_VIDEO_ALREADY_COMPLETE');return
    R.mkdir(parents=True,exist_ok=True);sources=json.loads((o.r2.OLD/'acceptance-four-scenes-5s-20261029/source_manifest.json').read_text('utf-8'))['scenes'];save_json(R/'source_manifest.json',dict(scenes=sources))
    pts=[];offset=0
    for s in sources:pts.extend([offset+t for t in s['pts']]);offset+=s['duration_seconds']
    writers={};records={e['id']:[] for e in entries};contacts={e['id']:Image.new('RGB',(1440,1200),'#111924') for e in entries};font=ImageFont.truetype(FONT,18);start=time.perf_counter();proofs=[]
    try:
        reused=[]
        for e in entries:
            k=e['id'];path=R/'reels'/f"{k}_单残差_四场景各5秒.mp4"
            # An interrupted file has no finalized report. Preserve that fact;
            # only this experiment's own unfinalized movie may be overwritten.
            if path.exists():path.unlink()
            if k not in ['A1','INK','OLD_FILM','PIXEL_R39']:
                expected=[]
                for s in sources:expected+=json.loads((BASE/'review/native'/k/(s['scene']+'.json')).read_text('utf-8'))['new_raw_sha256']
                summary=json.loads((BASE/'review/native'/k/'summary.json').read_text('utf-8'))['reel']
                writers[k]=ReuseEquivalentVideo(summary['path'],pts,expected,summary['sha256']);reused.append(k)
            else:writers[k]=Encoder(path,pts,k=='PIXEL_R39')
        for scene_index,source in enumerate(sources):
            frames=old_review.load_source(source);scene=source['scene'];flow_ids=[30,60,90,120,150];flows={n:old_review.ev.flow_mask(frames[n-1],frames[n])[:4] for n in flow_ids};exact={0,76,151,*flow_ids,*[n-1 for n in flow_ids]}
            for i in range(0,16,4):
                subset=entries[i:i+4];new=group([D/'models'/e['id'] for e in subset],1);teacher=group([e['source_folder'] for e in subset],2);rows={e['id']:dict(style_id=e['id'],scene=scene,label=source['label'],teacher_sha256=e['source_sha256'],coefficient_sha256=sha(D/'models'/e['id']/'coefficients-padded.bin'),source_sha256=source['source_sha256'],frames=152,pts=source['pts'],metrics=[],temporal_records=[],input_sha256=[]) for e in subset};previous={}
                for n,a in enumerate(frames):
                    assert time.perf_counter()-start<3600,'one-hour render cap'
                    if n%50==0:
                        o.guard();size=sum(p.stat().st_size for p in D.rglob('*') if p.is_file());assert size<3*2**30,'R1 own output above 3GiB'
                    x=o.tx(a[None]);out=rgb(new(x));target=rgb(teacher(x)) if n in exact else None
                    if n==0:
                        for j,e in enumerate(subset):
                            individual=o.Engine(D/'models'/e['id'])(x);assert np.array_equal(out[j],rgb(individual)[0]);proofs.append(dict(id=e['id'],scene=scene,group_vs_individual_full720_exact=True))
                    for j,e in enumerate(subset):
                        key=e['id'];b=out[j];writers[key].write(b);rows[key]['input_sha256'].append(digest(a))
                        if target is not None:
                            t=target[j];delta=np.abs(b.astype(np.float32)-t.astype(np.float32));rows[key]['metrics'].append(dict(frame=n,mean_absolute_difference=float(delta.mean()),max_channel_difference=int(delta.max()),one_vs_input_mae=float(np.abs(b.astype(np.float32)-a).mean()),two_vs_input_mae=float(np.abs(t.astype(np.float32)-a).mean())))
                            if n in flows:
                                rows[key]['temporal_records'].append(dict(frame=n,one=old_review.ev.temporal(previous[key][0],b,*flows[n]),two=old_review.ev.temporal(previous[key][1],t,*flows[n])))
                            previous[key]=(b,t)
                            if n==76:
                                folder=R/'stills'/key;folder.mkdir(parents=True,exist_ok=True);Image.fromarray(b).save(folder/f'{scene}_one.png');Image.fromarray(t).save(folder/f'{scene}_two.png')
                                if key in ['OLD_FILM','PIXEL_R39','INK']:rows[key]['special_features']=feature_check(e,x,b,t)
                                ck=torch.load(D/'models'/key/'selected.pt',map_location='cpu',weights_only=True);m=o.Model().cuda();m.load_state_dict(ck['model']);m.residual=ck['residual'];m.leak=0;m.qat=True;assert torch.equal(m(x),o.Engine(D/'models'/key)(x));del m,ck
                                panel=Image.new('RGB',(1440,300),'#111924');draw=ImageDraw.Draw(panel)
                                for col,(label,pic) in enumerate([('原图',a),('两残差 / 整数',t),('单残差 / 整数',b)]):
                                    draw.text((col*480+4,3),source['label']+' / '+label,font=font,fill='white');panel.paste(Image.fromarray(pic).resize((480,270),Image.Resampling.BOX),(col*480,29))
                                contacts[key].paste(panel,(0,scene_index*300))
                        if key=='PIXEL_R39':
                            cells=b.reshape(180,4,320,4,3);assert np.all(cells.max((1,3))==cells.min((1,3)))
                        if key=='INK':assert np.array_equal(b[...,0],b[...,1]) and np.array_equal(b[...,0],b[...,2])
                    if n%50==0:print('R1_VIDEO',scene,[e['id'] for e in subset],n,round(time.perf_counter()-start,1),flush=True)
                for e in subset:
                    row=rows[e['id']];assert row['input_sha256']==source['raw_input_frame_sha256'];row.update(exact_two_block_metric_frames=len(exact),no_inference_flow=True,no_post_filter=True,no_drop_or_repeat_inserted=True,full720_checkpoint_bin_equal=True);records[e['id']].append(row)
                    save_json(R/'scene-reports'/e['id']/(scene+'.json'),row)
                save_json(R/'progress.json',dict(complete=False,scene=scene,groups_done_in_scene=i//4+1,styles_with_4_scenes=[k for k,v in records.items() if len(v)==4],elapsed_seconds=time.perf_counter()-start))
                del new,teacher;torch.cuda.empty_cache()
            del frames,flows
        summary=[]
        for e in entries:
            key=e['id'];video=writers[key].close();contacts[key].save(R/'stills'/key/'四场景_原图双残差单残差.jpg',quality=96)
            result=dict(style_id=key,name=e.get('current_display_name',e['name']),coefficient_sha256=sha(D/'models'/key/'coefficients-padded.bin'),teacher_sha256=e['source_sha256'],video=video,scenes=records[key],no_inference_flow=True,no_post_filter=True,visual_acceptance=False);save_json(R/'results'/f'{key}.json',result);summary.append(result)
            print('R1_VIDEO_FINALIZED',key,video['frames'],video['maximum_all_frame_probe_rgb_difference'],video['pixel_unequal_cells'],flush=True)
        save_json(R/'complete.json',dict(status='PASS_16_NATIVE_REELS_64_FIVE_SECOND_SCENES',styles=16,frames=9728,scenes=64,newly_encoded_reels=4,reused_only_after_all_608_fresh_R1_raw_sha_matched=reused,group_proofs=proofs,seconds=time.perf_counter()-start));write_index(entries);print('ALL_16_R1_VIDEOS_COMPLETE',flush=True)
    except BaseException:
        for w in writers.values():
            if not hasattr(w,'proc') or w.proc.poll() is None:w.abort()
        raise

def write_index(entries):
    lines=['# 单残差16款：同源四景视频对照','','每款一段约20.245秒原生720p，依次室内、室外、人群、水面各152帧/约5.061秒，不重复保存同一视频的四个分段。所有帧由单残差导出BIN独立执行，无推理光流/滤镜。两残差原包与视频只读保留。','','|编号|名称|单残差20秒|两残差20秒|四景静帧对照|','|---|---|---|---|---|']
    for e in entries:
        k=e['id'];report=json.loads((R/'results'/f'{k}.json').read_text('utf-8'));new=Path(report['video']['path']);oldroot=BASE/'revisions/film-pixel-r2/review/display-reels' if k in ['OLD_FILM','PIXEL_R39'] else BASE/'review/reels';old=oldroot/f'{k}_新结构_四场景各5秒.mp4';assert old.exists()
        still=R/'stills'/k/'四场景_原图双残差单残差.jpg';lines.append(f"|{k}|{e.get('current_display_name',e['name'])}|[播放]({new.as_posix()})|[播放]({old.as_posix()})|[图片]({still.as_posix()})|")
    lines+=['','12款完全等价的异色/漫画/B03：本轮单残差BIN的608帧全部重新执行，逐帧原始RGB SHA与两残差旧记录相同后，复用同一旧视频，不重复编码占盘。旧文件保留原有H264/YUV420呈现，不能称新YUV444文件。4款有变化者重新编码为已确认兼容的BT601 YUV444 CRF0；像素原生视频解码后逐格检查。静帧三列原图/两残差/单残差，缩图不用作4×4格内RGB检验。MP4不是RGB字节精确样本。','']
    (D/'16款单残差视频索引.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
