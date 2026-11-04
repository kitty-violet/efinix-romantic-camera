"""Compatible YUV444 display encoding with measured RGB/PTS/cell checks."""
from pathlib import Path
import subprocess,json,hashlib
import cv2,numpy as np,imageio_ffmpeg
FF=imageio_ffmpeg.get_ffmpeg_exe()

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def pts_expr(ticks,lo=0):
    if len(ticks)==1:return str(ticks[0])
    mid=len(ticks)//2;return f'if(lt(N,{lo+mid}),{pts_expr(ticks[:mid],lo)},{pts_expr(ticks[mid:],lo+mid)})'

class Encoder:
    def __init__(self,path,pts,pixel=False):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);assert not self.path.exists()
        self.pts=list(pts);self.pixel=pixel;self.count=0;self.hashes=[];self.probes=[];self.keyframes={}
        self.script=self.path.with_suffix('.pts-filter.txt');self.log=self.path.with_suffix('.encoder.log')
        ticks=[int(round((v-pts[0])*10000)) for v in pts];assert all(b>a for a,b in zip(ticks,ticks[1:]))
        self.script.write_text("scale=out_color_matrix=bt601:out_range=tv,format=yuv444p,settb=1/10000,setpts='"+pts_expr(ticks)+"'\n",encoding='ascii')
        self.err=self.log.open('wb')
        cmd=[FF,'-hide_banner','-loglevel','error','-nostdin','-y','-threads','2','-f','rawvideo','-pixel_format','rgb24','-video_size','1280x720','-framerate','30','-i','pipe:0','-an','-filter_threads','2','-filter_script:v',str(self.script),'-fps_mode','passthrough','-enc_time_base','1:10000','-video_track_timescale','10000','-c:v','libx264','-threads','2','-crf','0','-preset','fast','-bf','0','-pix_fmt','yuv444p','-color_range','tv','-colorspace','smpte170m','-color_primaries','bt709','-color_trc','iec61966-2-1','-bsf:v','setts=duration=333','-movflags','+faststart',str(self.path)]
        self.proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.err)
    def write(self,a):
        assert a.shape==(720,1280,3) and a.dtype==np.uint8 and self.count<len(self.pts)
        self.hashes.append(digest(a))
        if self.pixel:
            cells=a.reshape(180,4,320,4,3);assert np.all(cells.max((1,3))==cells.min((1,3)));self.probes.append(a[::4,::4].copy())
        else:self.probes.append(a[::17,::17].copy())
        if self.count%152 in [0,76,151]:self.keyframes[self.count]=a.copy()
        self.proc.stdin.write(a.tobytes());self.count+=1
    def close(self):
        assert self.count==len(self.pts);self.proc.stdin.close();code=self.proc.wait(timeout=240);self.err.close();assert code==0,self.log.read_text('utf-8',errors='replace')
        cap=cv2.VideoCapture(str(self.path));hashes=[];pts=[];maximum=0;keymax=0;bad=0;i=0
        assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))==(1280,720)
        while True:
            ok,b=cap.read()
            if not ok:break
            assert i<len(self.probes);a=cv2.cvtColor(b,cv2.COLOR_BGR2RGB);pts.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000);hashes.append(digest(a))
            probe=a[::4,::4] if self.pixel else a[::17,::17]
            maximum=max(maximum,int(np.abs(probe.astype(np.int16)-self.probes[i].astype(np.int16)).max()))
            if i in self.keyframes:keymax=max(keymax,int(np.abs(a.astype(np.int16)-self.keyframes[i].astype(np.int16)).max()))
            if self.pixel:
                cells=a.reshape(180,4,320,4,3);bad+=int((cells.max((1,3))!=cells.min((1,3))).any(2).sum())
            i+=1
        cap.release();assert i==len(self.pts)==len(set(hashes))==len(set(self.hashes))
        error=max(abs(a-(b-self.pts[0])) for a,b in zip(pts,self.pts));assert error<.00001 and maximum<=4 and keymax<=4 and bad==0,(error,maximum,keymax,bad)
        result=dict(path=str(self.path),frames=i,unique_raw_frames=len(set(self.hashes)),unique_encoded_frames=len(set(hashes)),pts_error_seconds=error,codec='H264 CRF0 YUV444 BT601/SMPTE170M limited sRGB transfer',rgb_bit_exact=False,maximum_all_frame_probe_rgb_difference=maximum,maximum_rgb_difference_12_full_keyframes=keymax,maximum_all_pixel_rgb_difference=maximum if self.pixel else None,pixel_unequal_cells=bad if self.pixel else None,encoded_frame_sha256=hashes,raw_frame_sha256=self.hashes,width=1280,height=720,sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.script.unlink();self.log.unlink();self.probes.clear();self.keyframes.clear();return result
    def abort(self):
        try:self.proc.stdin.close()
        except Exception:pass
        if self.proc.poll() is None:self.proc.terminate()
        try:self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:self.proc.kill()
        if not self.err.closed:self.err.close()

class ReuseEquivalentVideo:
    """Reuse a frozen R2 movie only after every fresh R1 raw SHA matches it.

    The old lossy display encoding is explicitly retained. This saves disk,
    not model execution: all 608 R1 BIN frames still execute and are checked.
    """
    def __init__(self,path,pts,expected_raw_sha,expected_video_sha):
        self.path=Path(path);self.pts=list(pts);self.expected=list(expected_raw_sha);self.expected_video_sha=expected_video_sha
        assert hashlib.sha256(self.path.read_bytes()).hexdigest()==expected_video_sha
        self.cap=cv2.VideoCapture(str(self.path));assert self.cap.isOpened();self.count=0;self.hashes=[];self.encoded=[];self.maximum=0;self.pts_error=0
    def write(self,a):
        assert self.count<len(self.pts);h=digest(a);assert h==self.expected[self.count],('R1/R2 equivalence failed',self.path,self.count)
        ok,b=self.cap.read();assert ok;b=cv2.cvtColor(b,cv2.COLOR_BGR2RGB)
        self.maximum=max(self.maximum,int(np.abs(a[::17,::17].astype(np.int16)-b[::17,::17].astype(np.int16)).max()))
        self.pts_error=max(self.pts_error,abs(self.cap.get(cv2.CAP_PROP_POS_MSEC)/1000-(self.pts[self.count]-self.pts[0])))
        self.hashes.append(h);self.encoded.append(digest(b));self.count+=1
    def close(self):
        ok,_=self.cap.read();self.cap.release();assert not ok and self.count==608==len(self.pts)==len(set(self.hashes))==len(set(self.encoded));assert self.pts_error<.00001
        assert hashlib.sha256(self.path.read_bytes()).hexdigest()==self.expected_video_sha
        return dict(path=str(self.path),frames=608,unique_raw_frames=608,unique_encoded_frames=608,pts_error_seconds=self.pts_error,codec='original frozen R2 H264/YUV420 display reused unchanged after all 608 fresh R1 raw frame SHA match',rgb_bit_exact=False,maximum_all_frame_probe_rgb_difference=self.maximum,maximum_rgb_difference_12_full_keyframes=None,maximum_all_pixel_rgb_difference=None,pixel_unequal_cells=None,encoded_frame_sha256=self.encoded,raw_frame_sha256=self.hashes,width=1280,height=720,sha256=self.expected_video_sha,reused_exact_equivalent_r2_video=True,all_new_R1_raw_frames_actually_executed=True,all_608_raw_sha_match_R2=True)
    def abort(self):self.cap.release()
