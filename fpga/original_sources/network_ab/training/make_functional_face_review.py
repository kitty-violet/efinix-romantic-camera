"""Near-person comparison at the existing fixed11.97s frame, no face paste."""
import cv2
from PIL import Image,ImageDraw,ImageFont
from .audit_style_branches import RUN,load
from .prepare_b1_distillation import rgb_output
from .fpga_video_resize import area_3to2
from .runtime import configure_workspace
from .engine import set_reproducible


def main():
    root=configure_workspace();set_reproducible(2026100404);run=root/'runs'/RUN
    cap=cv2.VideoCapture(str(root/'runs/clear-style-control-20261003/video_sources/plaza_source_1080p.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES,359);assert cap.get(cv2.CAP_PROP_POS_FRAMES)==359;ok,bgr=cap.read();assert ok;cap.release()
    source=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));teacher=rgb_output(load(root,'teacher'),source)
    values=[Image.fromarray(source),Image.fromarray(teacher),Image.open(run/'reference/videos/plaza_359_native720.png'),Image.open(run/'videos/plaza_359_native720.png')]
    labels=['原片 / 11.97秒','教师目标','上轮修正版','本轮候选'];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    p=Image.new('RGB',(1520,438),'#151b24');d=ImageDraw.Draw(p)
    for i,(label,v) in enumerate(zip(labels,values)):
        d.text((i*380+5,6),label,font=font,fill='white');p.paste(v.crop((900,340,1280,720)),(i*380,30))
    d.text((6,412),'原生720像素裁切；统一固定位置；没有局部修脸或贴原片。',font=font,fill='#b9d5ef');p.save(run/'near_faces.png')
    print('NEAR_FACE_REVIEW_READY',flush=True)


if __name__=='__main__':main()
