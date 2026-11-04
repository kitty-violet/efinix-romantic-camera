"""Actual exported RGB integers, original 1073 VFR frames, no output filter."""
import time,subprocess,copy
import imageio_ffmpeg
from PIL import ImageDraw,ImageFont
from common import *
import group_engine as ge
import render_review as rr
from media import pts_expr

NEW=D/'models/PIXEL_COHERENT'
LABELS=['原图','单残差：修前','单残差：修后','双残差：原版保留']
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)

def canvas(images):
    a=np.zeros((1504,2560,3),np.uint8)
    for i,p in enumerate(images):
        x=i%2*1280;y=i//2*752;a[y+32:y+752,x:x+1280]=p
    im=Image.fromarray(a);dr=ImageDraw.Draw(im)
    for i,label in enumerate(LABELS):dr.text((i%2*1280+8,i//2*752+2),label,font=FONT,fill='white')
    return np.array(im)

class Writer:
    def __init__(self,path,ticks,width,height,pixel):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);assert not self.path.exists()
        self.ticks=ticks;self.width=width;self.height=height;self.pixel=pixel;self.count=0;self.hashes=[];self.probes=[];self.keys={}
        self.filter=self.path.with_suffix('.pts.txt');self.log=self.path.with_suffix('.encoder.log')
        self.filter.write_text("scale=out_color_matrix=bt601:out_range=tv,format=yuv444p,settb=1/90000,setpts='"+pts_expr(ticks)+"'\n",encoding='ascii')
        self.err=self.log.open('wb');duration=int(np.median(np.diff(ticks)))
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-nostdin','-y','-threads','2','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}','-framerate','60','-i','pipe:0','-an','-filter_threads','2','-filter_script:v',str(self.filter),'-fps_mode','passthrough','-enc_time_base','1:90000','-video_track_timescale','90000','-c:v','libx264','-threads','2','-crf','0','-preset','fast','-bf','0','-pix_fmt','yuv444p','-color_range','tv','-colorspace','smpte170m','-color_primaries','bt709','-color_trc','iec61966-2-1','-bsf:v',f'setts=duration={duration}','-movflags','+faststart',str(self.path)]
        self.proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.err)
    def write(self,a):
        assert a.shape==(self.height,self.width,3) and self.count<len(self.ticks)
        self.hashes.append(digest(a));self.probes.append(a[::4,::4].copy() if self.pixel else a[::23,::23].copy())
        if self.count in [0,536,1072]:self.keys[self.count]=a.copy()
        self.proc.stdin.write(a.tobytes());self.count+=1
    def close(self):
        self.proc.stdin.close();code=self.proc.wait(timeout=180);self.err.close();assert code==0,self.log.read_text('utf-8',errors='replace')
        cap=cv2.VideoCapture(str(self.path));i=0;maximum=0;pts_error=0.;hashes=[];keymax=0
        assert cap.isOpened() and int(cap.get(3))==self.width and int(cap.get(4))==self.height
        while True:
            ok,b=cap.read()
            if not ok:break
            assert i<self.count;a=cv2.cvtColor(b,cv2.COLOR_BGR2RGB)
            q=a[::4,::4] if self.pixel else a[::23,::23]
            maximum=max(maximum,int(np.abs(q.astype(np.int16)-self.probes[i].astype(np.int16)).max()))
            if self.pixel:assert np.array_equal(a,q.repeat(4,0).repeat(4,1)),('decoded pixel cell',self.path,i)
            if i in self.keys:keymax=max(keymax,int(np.abs(a.astype(np.int16)-self.keys[i].astype(np.int16)).max()))
            pts_error=max(pts_error,abs(cap.get(cv2.CAP_PROP_POS_MSEC)/1000-self.ticks[i]/90000));hashes.append(digest(a));i+=1
        cap.release();assert i==self.count==1073 and maximum<=4 and keymax<=4 and pts_error<1e-5,(i,maximum,keymax,pts_error)
        report=dict(path=str(self.path),sha256=sha(self.path),frames=i,width=self.width,height=self.height,pts_error_seconds=pts_error,maximum_rgb_probe_difference=maximum,maximum_full_keyframe_difference=keymax,pixel_unequal_cells=0 if self.pixel else None,raw_sha256=self.hashes,decoded_sha256=hashes,unique_raw_frames=len(set(self.hashes)),source_duplicate_frames_not_removed=True,no_frames_inserted=True,format='H264 CRF0 YUV444 BT601 limited/sRGB',rgb_bit_exact=False)
        save(self.path.with_suffix('.verification.json'),report);self.filter.unlink();self.log.unlink();self.probes.clear();self.keys.clear();return report
    def abort(self):
        try:self.proc.stdin.close()
        except Exception:pass
        if self.proc.poll() is None:self.proc.terminate()
        self.proc.wait(timeout=15);self.err.close()

def group_r1():
    previous=ge.PackedEngine;ge.PackedEngine=lambda p:o.Engine(p)
    try:return ge.GroupEngine(folders=[R1,NEW])
    finally:ge.PackedEngine=previous

@torch.inference_mode()
def main():
    o.setup(20261102);guard();start=time.perf_counter();source=json.loads((D/'source-manifest.json').read_text('utf-8'))
    assert sha(VIDEO)==source['video']['sha256'] and sha(PHOTO)==source['photo']['sha256']
    individual=[o.Engine(R1),o.Engine(NEW),o.Engine(R2,2)];together=group_r1();checkpoint=model();ck=torch.load(NEW/'selected.pt',weights_only=True,map_location='cpu');checkpoint.load_state_dict(ck['model']);checkpoint.residual=ck['residual']
    (D/'review').mkdir(exist_ok=True);stills=[]
    for p in sorted((D/'inputs').glob('*.png')):
        a=np.array(Image.open(p));x=o.tx(a[None]);ys=[rgb(e(x)) for e in individual];assert torch.equal(checkpoint(x),individual[1](x))
        for key,b in zip(['one_before','one_after','two'],ys):Image.fromarray(b).save(D/'review'/f'{p.stem}_{key}.png');stills.append(dict(case=p.stem,variant=key,**block_metrics(b,a)))
        Image.fromarray(canvas([a,*ys])).save(D/'review'/f'{p.stem}_四版对照.png')
    save(D/'review/still-metrics.json',stills)
    # Retest original independent four scenes with the same actual frame pairs.
    regress=[];originals=o.r2.OLD/'acceptance-four-scenes-5s-20261029/originals'
    scenes=json.loads((o.r2.OLD/'acceptance-four-scenes-5s-20261029/source_manifest.json').read_text('utf-8'))['scenes']
    for s in scenes:
        frames=rr.load_source(s);pairs=[]
        for n in [30,60,90,120,150]:
            aa,bb=frames[n-1],frames[n];flow=rr.ev.flow_mask(aa,bb)[:4];row=dict(frame=n)
            for key,e in zip(['one_before','one_after','two'],individual):
                pa=rgb(e(o.tx(aa[None])));pb=rgb(e(o.tx(bb[None])));row[key]=rr.ev.temporal(pa,pb,*flow)
            pairs.append(row)
        regress.append(dict(scene=s['scene'],source_sha256=s['source_sha256'],pairs=pairs));del frames
        print('PIXEL_REGRESSION',s['scene'],flush=True)
    save(D/'review/old-four-scene-temporal.json',regress)
    records=source['video']['records'];ticks=[r['pts_90000']-records[0]['pts_90000'] for r in records]
    writers=[Writer(D/'review/雪山视频_原图单残差修前修后双残差.mp4',ticks,2560,1504,False),Writer(D/'review/雪山视频_单残差修后_原生720p.mp4',ticks,1280,720,True),Writer(D/'review/雪山视频_双残差_原生720p.mp4',ticks,1280,720,True)]
    temporal=[];samples=[];previous=None;cap=cv2.VideoCapture(str(VIDEO));i=0
    try:
        while True:
            ok,b=cap.read()
            if not ok:break
            assert i<len(records);raw=cv2.cvtColor(b,cv2.COLOR_BGR2RGB);a,_=video_input(raw)
            assert digest(raw)==records[i]['original_rgb_sha256'] and digest(a)==records[i]['input720_sha256']
            assert round(cap.get(cv2.CAP_PROP_POS_MSEC)*90)==records[i]['pts_90000']
            x=o.tx(a[None]);two=individual[2](x);outs=together(x)
            tensors=[outs[0:1],outs[1:2],two]
            if i in [0,536,1072]:
                assert torch.equal(tensors[0],individual[0](x)) and torch.equal(tensors[1],individual[1](x))
                assert torch.equal(checkpoint(x),tensors[1])
            ys=[]
            for t in tensors:
                assert torch.equal(t,t[:,:,::4,::4].repeat_interleave(4,-2).repeat_interleave(4,-1));ys.append(rgb(t))
            writers[0].write(canvas([a,*ys]));writers[1].write(ys[1]);writers[2].write(ys[2])
            if i%60==0 or i in [536,1072]:
                samples.append(dict(frame=i,metrics={k:block_metrics(y,a) for k,y in zip(['one_before','one_after','two'],ys)}))
                if previous is not None:
                    flow=rr.ev.flow_mask(previous[0],a)[:4]
                    temporal.append(dict(frame=i,**{key:rr.ev.temporal(pa,pb,*flow) for key,pa,pb in zip(['one_before','one_after','two'],previous[1],ys)}))
            previous=(a,ys);i+=1
            if i%100==0:guard();assert time.perf_counter()-start<3600;print('PIXEL_FULL_VIDEO',i,round(time.perf_counter()-start,1),flush=True)
        cap.release();assert i==1073
        videos=[]
        for w in writers:videos.append(w.close());print('PIXEL_VIDEO_VERIFIED',w.path.name,flush=True)
        save(D/'review/complete.json',dict(status='PASS_NUMERICAL_MEDIA_NOT_VISUAL_ACCEPTANCE',source_frames=1073,all_source_frames_read_by_pixel_hash=True,all_original_pts_preserved=True,source_video_sha256=sha(VIDEO),variants=['original','one_before','one_after','two'],coefficient_sha256=[sha(p/'coefficients-padded.bin') for p in [R1,NEW,R2]],all_output_4x4_cells_equal=True,checkpoint_bin_full720_equal=True,group_vs_individual_full720_equal=True,no_inference_post_filter=True,no_inference_flow=True,videos=videos,feature_samples=samples,temporal_records=temporal,seconds=time.perf_counter()-start))
        print('PIXEL_COMPLETE',flush=True)
    except BaseException:
        cap.release()
        for w in writers:
            if w.proc.poll() is None:w.abort()
        raise

if __name__=='__main__':main()
