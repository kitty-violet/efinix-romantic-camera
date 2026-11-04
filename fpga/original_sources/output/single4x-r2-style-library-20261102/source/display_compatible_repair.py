"""Pointwise RGB->BT601 YUV444 lossless display encoding, without chroma blur.

The source RGB H264 is a numerical intermediate, not a user preview: the app
misdisplays its GBR colour matrix. This is presentation conversion only.
"""
from pathlib import Path
import json,subprocess,argparse,hashlib
import cv2,numpy as np
import imageio_ffmpeg

BASE=Path(__file__).resolve().parents[1];D=BASE/'revisions/film-pixel-r2';R=D/'review'
FF=imageio_ffmpeg.get_ffmpeg_exe()

def transcode(source,dest,pixel=False,cleanup=False):
    source=Path(source);dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    cmd=[FF,'-hide_banner','-loglevel','error','-nostdin','-y','-threads','2','-i',str(source),'-an','-filter_threads','2','-vf','scale=out_color_matrix=bt601:out_range=tv,format=yuv444p','-c:v','libx264','-threads','2','-crf','0','-preset','fast','-bf','0','-pix_fmt','yuv444p','-color_range','tv','-colorspace','smpte170m','-color_primaries','bt709','-color_trc','iec61966-2-1','-fps_mode','passthrough','-enc_time_base','1:10000','-video_track_timescale','10000','-movflags','+faststart',str(dest)]
    p=subprocess.run(cmd,capture_output=True,timeout=240);assert p.returncode==0,p.stderr
    c1=cv2.VideoCapture(str(source));c2=cv2.VideoCapture(str(dest));frames=0;maximum=0;total=0;elements=0;bad_cells=0;hashes=[];src_pts=[];dst_pts=[]
    while True:
        ok,a=c1.read();ok2,b=c2.read();assert ok==ok2
        if not ok:break
        src_pts.append(c1.get(cv2.CAP_PROP_POS_MSEC)/1000);dst_pts.append(c2.get(cv2.CAP_PROP_POS_MSEC)/1000)
        a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB);b=cv2.cvtColor(b,cv2.COLOR_BGR2RGB);err=np.abs(a.astype(np.int16)-b.astype(np.int16));maximum=max(maximum,int(err.max()));total+=int(err.sum());elements+=a.size;frames+=1
        hashes.append(hashlib.sha256(b.tobytes()).hexdigest())
        if pixel:
            z=b.reshape(180,4,320,4,3);bad_cells+=int((z.max((1,3))!=z.min((1,3))).any(2).sum())
    c1.release();c2.release();assert maximum<=4 and bad_cells==0,(maximum,bad_cells)
    assert all(b>a for a,b in zip(src_pts,src_pts[1:]));pts_error=max(abs(a-b) for a,b in zip(src_pts,dst_pts));assert pts_error<.00001,pts_error
    result=dict(source_rgb_reference=str(source),display_path=str(dest),frames=frames,maximum_rgb_codec_difference=maximum,mean_rgb_codec_difference=total/elements,unequal_4x4_cells=bad_cells if pixel else None,maximum_pts_difference_seconds=pts_error,decoded_frame_sha256=hashes,pixel_format='yuv444p',colour_matrix='SMPTE170M/BT601 limited; sRGB transfer',no_chroma_subsampling=True,no_spatial_filter=True,rgb_bit_exact=False)
    if cleanup:
        assert source.resolve().is_relative_to(R.resolve()) and source.suffix=='.mp4'
        assert source.parts[len(R.parts)] in ['native','reels']
        result.update(source_rgb_reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_reference_cache_removed=True,source_reference_bytes=source.stat().st_size)
        proof=dest.with_suffix('.verification.json');proof.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        source.unlink()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--one',action='store_true');a=p.parse_args();records=[]
    if a.one:
        records.append(transcode(R/'native/PIXEL_R39/indoor.mp4',R/'display/PIXEL_R39/indoor.mp4',True))
    else:
        for key in ['OLD_FILM','PIXEL_R39']:
            for scene in ['indoor','outdoor','crowd','water']:
                rec=transcode(R/'native'/key/(scene+'.mp4'),R/'display'/key/(scene+'.mp4'),key=='PIXEL_R39',cleanup=True);rec.update(style_id=key,scene=scene);records.append(rec)
                print('COMPATIBLE_NATIVE',key,scene,rec['maximum_rgb_codec_difference'],rec['unequal_4x4_cells'],flush=True)
            for suffix in ['新结构_四场景各5秒','原图旧版新版_四场景各5秒']:
                rec=transcode(R/'reels'/f'{key}_{suffix}.mp4',R/'display-reels'/f'{key}_{suffix}.mp4',key=='PIXEL_R39' and suffix.startswith('新结构'),cleanup=True)
                rec.update(style_id=key,reel_kind=suffix);records.append(rec)
    path=D/('display-codec-one.json' if a.one else 'display-codec-verification.json');path.write_text(json.dumps(dict(status='PASS_MEASURED_CODEC_DIFFERENCE_NOT_RGB_EXACT',records=records),ensure_ascii=False,indent=2),encoding='utf-8')
    print('DISPLAY_COMPATIBLE_COMPLETE',len(records),flush=True)

if __name__=='__main__':main()
