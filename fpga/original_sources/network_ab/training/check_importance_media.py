"""Check served videos, encoding error, and same-frame nearby faces."""
import argparse
import json
import urllib.request
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .importance_pruned_teacher import RUN
from .fpga_video_resize import area_3to2
from .runtime import configure_workspace,file_hash,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',required=True);args=ap.parse_args()
    root=configure_workspace();run=root/'runs'/RUN;report=json.loads((run/('videos_'+args.candidate)/'report.json').read_text());assert report['status']=='COMPLETE'
    cv2.setNumThreads(1);records=[];near=None
    for row in report['records']:
        path=Path(row['path']);assert file_hash(path)==row['validation']['sha256']
        url='http://127.0.0.1:8883/'+RUN+'/videos_'+args.candidate+'/'+path.name
        with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-1023'}),timeout=20) as response:
            data=response.read();status=response.status;content_range=response.headers.get('Content-Range')
        assert status==206 and len(data)==1024
        with path.open('rb') as f:assert f.read(1024)==data
        cap=cv2.VideoCapture(str(path));frames=[]
        for i in range(row['validation']['frames']):
            ok,bgr=cap.read();assert ok
            if i==90:encoded=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            if row['id']=='plaza' and i==359:near=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            if i%30==0:frames.append((i,Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((320,180))))
        assert not cap.read()[0];cap.release()
        before=np.asarray(Image.open(path.parent/f"{row['id']}_090_preencode1080.png"));d=np.abs(before.astype(float)-encoded.astype(float))
        canvas=Image.new('RGB',(960,208*((len(frames)+2)//3)),'#151b24');draw=ImageDraw.Draw(canvas)
        for j,(i,frame) in enumerate(frames):
            x=j%3*320;y=j//3*208;draw.text((x+7,y+7),f'{row["id"]} {i/30:.0f}s',fill='white');canvas.paste(frame,(x,y+28))
        canvas.save(run/(row['id']+'_filmstrip.png'))
        records.append(dict(id=row['id'],http_range_status=status,content_range=content_range,range_bytes_match=True,
                            frame90_encoding_mae=float(d.mean()),frame90_encoding_psnr=float(10*np.log10(255**2/max((d*d).mean(),1e-12))),decoded_frames=len(range(row['validation']['frames']))))
    paths=[root/'runs/clear-style-control-20261003/video_sources/plaza_source_1080p.mp4',root/'runs/b1-distillation-20261003/videos-2000/plaza_teacher_1080p.mp4',
           root/'runs/teacher-pruning-calibrated-20261003/videos_keep50/plaza_pruned_1080p.mp4']
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);panel=Image.new('RGB',(1520,430),'#151b24');draw=ImageDraw.Draw(panel)
    for j,(p,label) in enumerate(zip(paths+[None],['原片','原教师','上一轮半通道','本轮重要性剪枝'])):
        if p:
            cap=cv2.VideoCapture(str(p))
            for i in range(360):ok,bgr=cap.read();assert ok
            cap.release();rgb=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        else:rgb=near
        assert rgb is not None;panel.paste(Image.fromarray(rgb).crop((900,340,1280,720)),(j*380,30));draw.text((j*380+8,5),label,font=font,fill='white')
    draw.text((7,406),'广场11.97秒同帧视频解码还原到720；未锐化、未贴回原脸。',font=font,fill='#c3d5e9');panel.save(run/'near_faces.png')
    write_json(run/'media_checks.json',dict(status='PASS',records=records,near_face_frame=359,
        scope='All frames sequentially decoded; HTTP byte ranges verified; encoding loss at frame90 and sampled filmstrips are not a no-flicker certificate.',code_sha256=file_hash(__file__)))
    print('MEDIA_CHECKS_PASS',flush=True)


if __name__=='__main__':main()
