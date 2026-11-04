"""Decode-delivery QA: continuity, same-frame codec error, and sampled crops."""
import json,math
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from .fpga_video_resize import bilinear_2to3
from .runtime import write_json,file_hash

OUT=Path(r'E:\fpga\model-training-workspace\runs\region-detail-deploy-20261004\block-separation-video-20261004-v1')


def main():
    cv2.setNumThreads(2)
    path=OUT/'plaza_balanced_15s_1080p.mp4';cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
    rows=[];n=0;duplicates=0;last=None;sampled=[]
    while True:
        ok,bgr=cap.read()
        if not ok:break
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        if last is not None:duplicates+=int(np.array_equal(rgb,last))
        if n in (0,90,210,330,449):
            expected=bilinear_2to3(np.array(Image.open(OUT/f'frame_{n:03d}_native720.png')))
            delta=rgb.astype(np.float32)-expected.astype(np.float32);mse=float((delta*delta).mean())
            rows.append(dict(frame=n,MAE=float(np.abs(delta).mean()),PSNR=10*math.log10(255**2/mse)))
            Image.fromarray(rgb).save(OUT/f'decoded_{n:03d}_1080p.png')
            sampled.append((n,rgb))
        n+=1;last=rgb
    shape=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))];fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    assert n==450 and shape==[1920,1080] and fps==30 and duplicates==0
    # These thumbnails are navigation aids only; native and decoded frames are retained.
    f=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    board=Image.new('RGB',(960,3*304),'#17202a');draw=ImageDraw.Draw(board)
    for i,(frame,rgb) in enumerate(sampled):
        x=(i%2)*480;y=(i//2)*304;draw.text((x+5,y+4),f'{frame/30:.2f}s / 第{frame}帧',font=f,fill='white')
        board.paste(Image.fromarray(rgb).resize((480,270),Image.Resampling.LANCZOS),(x,y+34))
    board.save(OUT/'video_contact.png')
    write_json(OUT/'decode_verification.json',dict(decoded_frames=n,size=shape,fps=fps,seconds=n/fps,
        identical_adjacent_decoded_frames=duplicates,video_sha256=file_hash(path),codec_error_from_exact_preencode_frames=rows,
        note='Codec error measures only compression against the same preencode frame, not style quality or temporal stability.'))
    print('DECODE_QA_DONE',n,rows,flush=True)


if __name__=='__main__':main()
