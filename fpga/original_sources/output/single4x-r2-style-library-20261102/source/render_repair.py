"""Render repaired BINs; native and comparison videos use lossless RGB H264."""
from pathlib import Path
import sys,json,time,hashlib,subprocess
import numpy as np
import cv2
import library as l
import render_review as rr
from build_four_scene_acceptance import StreamEncoder as OriginalEncoder,balanced_pts,FF

BASE=l.D;D=BASE/'revisions/film-pixel-r2'

class LosslessRGBEncoder(OriginalEncoder):
    def __init__(self,path,pts,width=1280,height=720):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);assert not self.path.exists()
        self.pts=list(pts);self.width=width;self.height=height;self.count=0;self.raw_hashes=[]
        self.script=self.path.with_suffix('.pts-filter.txt');ticks=[int(round((t-pts[0])*10000)) for t in pts]
        assert all(b>a for a,b in zip(ticks,ticks[1:]))
        self.script.write_text("settb=1/10000,setpts='"+balanced_pts(ticks)+"'\n",encoding='ascii')
        self.log=self.path.with_suffix('.encoder.log');self.error=self.log.open('wb')
        cmd=[FF,'-hide_banner','-loglevel','error','-nostdin','-y','-threads','2','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}','-framerate','30','-i','pipe:0','-an','-filter_threads','2','-filter_script:v',str(self.script),'-fps_mode','passthrough','-enc_time_base','1:10000','-video_track_timescale','10000','-c:v','libx264rgb','-threads','2','-crf','0','-preset','fast','-bf','0','-pix_fmt','rgb24','-bsf:v',f'setts=duration={int(round(np.median(np.diff(pts))*10000))}','-movflags','+faststart',str(self.path)]
        self.proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.error)
    def write(self,a):
        self.raw_hashes.append(hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest());super().write(a)
    def close(self,unique=True):
        result=super().close(unique=unique);cap=cv2.VideoCapture(str(self.path));observed=[];flat=True
        while True:
            ok,b=cap.read()
            if not ok:break
            a=cv2.cvtColor(b,cv2.COLOR_BGR2RGB);observed.append(hashlib.sha256(a.tobytes()).hexdigest())
            if self.width==1280 and self.height==720 and 'PIXEL_R39' in str(self.path):
                cells=a.reshape(180,4,320,4,3);flat=flat and bool(np.all(cells.max((1,3))==cells.min((1,3))))
        cap.release();assert observed==self.raw_hashes,(self.path,'RGB roundtrip mismatch')
        assert flat,(self.path,'decoded pixel cells not exactly flat')
        result.update(codec='libx264rgb CRF0 RGB lossless',decoded_rgb_all_frame_hashes_exact=True,decoded_4x4_flatness=flat if 'PIXEL_R39' in str(self.path) and self.width==1280 else None)
        return result

def main():
    l.setup(20261102);rr.D=D;rr.R=D/'review';l.D=D;rr.StreamEncoder=LosslessRGBEncoder
    if '--static-only' in sys.argv:
        sys.argv=[sys.argv[0],'--ids','OLD_FILM','PIXEL_R39','--static-only']
    else:
        # Prove the exact codec path on irregular timestamps and arbitrary RGB.
        p=D/'codec-selftest.mp4';e=LosslessRGBEncoder(p,[0.,.0333,.07,.1011,.15],32,24)
        rng=np.random.default_rng(3412)
        for _ in range(5):e.write(rng.integers(0,256,(24,32,3),dtype=np.uint8))
        result=e.close();p.unlink();l.save_json(D/'codec-selftest.json',result)
        sys.argv=[sys.argv[0],'--ids','OLD_FILM','PIXEL_R39']
    rr.main()

if __name__=='__main__':main()
