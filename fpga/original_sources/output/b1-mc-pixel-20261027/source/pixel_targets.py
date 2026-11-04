"""Deterministic flat pixel teacher; training labels only, never inference."""
from pathlib import Path
import json, hashlib
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(r'E:\fpga')
CONFIG=dict(cell=4, levels=6, transition_rgb=14., material_matching=False,
            preserve_existing_shading=True, invent_geometry=False, inference_filter=False)

def target(a,cell=4):
    h,w=a.shape[:2];assert h%cell==0 and w%cell==0
    low=a.astype(np.float32).reshape(h//cell,cell,w//cell,cell,3).mean((1,3))
    q=np.zeros_like(low)
    for t in (np.arange(5)+.5)*51:
        q+=51*np.clip((low-t)/CONFIG['transition_rgb']+.5,0,1)
    q=np.clip(q+.5,0,255).astype(np.uint8)
    return q.repeat(cell,0).repeat(cell,1)

def panel(arrays,labels,path,width=640,nearest=True):
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    out=Image.new('RGB',(width*len(arrays),width*arrays[0].shape[0]//arrays[0].shape[1]+34),'#151b22')
    d=ImageDraw.Draw(out)
    for j,(a,label) in enumerate(zip(arrays,labels)):
        height=round(width*a.shape[0]/a.shape[1])
        out.paste(Image.fromarray(a).resize((width,height),Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS),(width*j,34))
        d.text((width*j+8,6),label,font=font,fill='white')
    out.save(path)

def main():
    dest=ROOT/'targets';dest.mkdir(exist_ok=True)
    records=[]
    for scene,source in [('outdoor','outdoor'),('indoor','station')]:
        p=PROJECT/f'output/b1-ink-wash-20261026/independent_test/{source}_0000_source.png'
        a=np.asarray(Image.open(p).convert('RGB'));q=target(a)
        Image.fromarray(a).save(dest/f'{scene}_source.png');Image.fromarray(q).save(dest/f'{scene}_target.png')
        panel([a,q],['原图','MC平面像素目标 / 非B1'],dest/f'{scene}_comparison.jpg')
        crop=(slice(192,384),slice(448,768))
        panel([a[crop],q[crop]],['原图局部 ×2','目标局部 ×2 / 最近邻显示'],dest/f'{scene}_detail.png')
        records.append(dict(id=scene,source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                            target=str(dest/f'{scene}_target.png'),cell=4,shape=list(q.shape)))
    (ROOT/'target_manifest.json').write_text(json.dumps(dict(config=CONFIG,records=records,
        reference='https://www.minecraft.net/en-us/vibrant-visuals-update',
        scope='Borrow stepped silhouette and flat pixel colour; no material matching or 3D voxel reconstruction.'),ensure_ascii=False,indent=2),encoding='utf-8')
    print('PIXEL_TARGETS_COMPLETE',flush=True)

if __name__=='__main__':main()
