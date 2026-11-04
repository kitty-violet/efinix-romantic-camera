# Pure evaluation helpers copied from the existing video evaluator.
# No old style coefficients, model initialization, or directory globals imported.
import hashlib, subprocess, re
from pathlib import Path
import numpy as np
import cv2
from network_ab.training.media import probe_video
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def luminance(a):return a.astype(np.float32)@np.array([.2126,.7152,.0722],np.float32)

def highpass(a):return a.astype(np.float32)-cv2.GaussianBlur(a.astype(np.float32),(0,0),2,borderType=cv2.BORDER_REPLICATE)

def flow_mask(a,b):
    a,b=[cv2.resize(cv2.cvtColor(x,cv2.COLOR_RGB2GRAY),(640,360),interpolation=cv2.INTER_AREA) for x in (a,b)]
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    back=dis.calc(b,a,None);fw=dis.calc(a,b,None)
    yy,xx=np.mgrid[:360,:640].astype(np.float32);mx,my=xx+back[...,0],yy+back[...,1]
    opposite=cv2.remap(fw,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    aw=cv2.remap(a,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    valid=(np.sum((back+opposite)**2,2)<.01*np.sum(back**2+opposite**2,2)+.5)&(mx>2)&(my>2)&(mx<637)&(my<357)&(np.abs(b.astype(np.float32)-aw)<18)
    f=cv2.resize(back,(1280,720),interpolation=cv2.INTER_LINEAR)*2
    v=cv2.resize(valid.astype(np.uint8),(1280,720),interpolation=cv2.INTER_NEAREST).astype(bool)
    yy,xx=np.mgrid[:720,:1280].astype(np.float32)
    mx,my=xx+f[...,0],yy+f[...,1]
    v[:8]=False;v[-8:]=False;v[:,:8]=False;v[:,-8:]=False
    static=v&(np.linalg.norm(f,axis=2)<.5)
    return mx,my,v,static,np.linalg.norm(f,axis=2)

def temporal(previous,current,mx,my,valid,static):
    old=cv2.remap(previous,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    delta=np.abs(current.astype(np.float32)-old)
    hp_old=cv2.remap(highpass(previous),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    hp_delta=np.abs(highpass(current)-hp_old)
    amplitude=float(np.sqrt(np.mean(highpass(current)[valid]**2))) if valid.any() else 0.
    return dict(warp_mae=float(delta[valid].mean()) if valid.any() else None,hp_warp_mae=float(hp_delta[valid].mean()) if valid.any() else None,hp_error_over_signal=float(hp_delta[valid].mean()/amplitude) if amplitude>1e-3 else None,unaligned_static_mae=float(np.abs(current.astype(np.float32)-previous)[static].mean()) if static.sum()>100 else None,static_pixels=int(static.sum()),mean_luma_jump=float(abs(luminance(current).mean()-luminance(previous).mean())))

def spatial(source,out):
    y=luminance(out);sy=luminance(source)
    gx=cv2.Sobel(y,cv2.CV_32F,1,0,ksize=3)/8;gy=cv2.Sobel(y,cv2.CV_32F,0,1,ksize=3)/8
    sx=cv2.Sobel(sy,cv2.CV_32F,1,0,ksize=3)/8;ssy=cv2.Sobel(sy,cv2.CV_32F,0,1,ksize=3)/8
    strength=np.hypot(sx,ssy);mask=strength>10
    alignment=np.maximum(gx*sx+gy*ssy,0)/(strength+1e-4)
    xx=cv2.GaussianBlur(gx*gx,(0,0),2);yy=cv2.GaussianBlur(gy*gy,(0,0),2);xy=cv2.GaussianBlur(gx*gy,(0,0),2)
    coherence=np.sqrt((xx-yy)**2+4*xy**2)/(xx+yy+1e-3)
    return dict(hp_luma_rms=float(np.sqrt(np.mean(highpass(y)**2))),directional_coherence=float(coherence.mean()),aligned_strong_edge_ratio=float(alignment[mask].mean()/(strength[mask].mean()+1e-4)) if mask.any() else None,dark_fraction=float((y<10).mean()),bright_fraction=float((y>245).mean()),blue_fraction=float(((out[...,2]>out[...,0]+20.)&(out[...,2]>out[...,1])).mean()),yellow_fraction=float(((out[...,0]>out[...,2]+25.)&(out[...,1]>out[...,2]+15.)).mean()))

def encode_vfr(folder,paths,pts,final,ffmpeg):
    listing=folder/(final.stem+'.ffconcat')
    lines=['ffconcat version 1.0']
    durations=np.diff(pts).tolist();durations.append(float(np.median(np.diff(pts))))
    for path,duration in zip(paths,durations):
        lines.extend(["file '"+path.as_posix()+"'",'option framerate 10000',f'duration {duration:.8f}'])
    listing.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    cmd=[ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-y','-f','concat','-safe','0','-i',str(listing),'-map','0:v:0','-fps_mode','passthrough','-enc_time_base','1:10000','-video_track_timescale','10000','-c:v','libx264','-threads','2','-crf','16','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(final)]
    subprocess.run(cmd,check=True,timeout=180)
    check=subprocess.run([ffmpeg,'-hide_banner','-i',str(final),'-vf','showinfo','-f','null','-'],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
    observed=[float(v) for v in re.findall(r'\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:([-+\deE.]+)',check.stderr)]
    assert check.returncode==0 and len(observed)==len(pts),(final,len(observed),len(pts))
    error=max(abs((a-pts[0])-b) for a,b in zip(pts,observed));assert error<.001,(final,error)
    return dict(path=str(final),sha256=sha(final),frames=len(observed),max_pts_error_seconds=error,encoded_resolution=probe_video(final)['width'],no_frame_drop_or_duplication=True)
