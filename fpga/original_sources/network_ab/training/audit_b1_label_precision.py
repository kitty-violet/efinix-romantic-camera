"""Reproduce the coarse contour-label precision defect at matched settings."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .runtime import write_json,file_hash
from .train_b1_structural_gate import CROPS,BASE


def edge(rgb,floating):
    gray=cv2.cvtColor(rgb.astype(np.float32) if floating else rgb,cv2.COLOR_RGB2GRAY)
    gray=cv2.GaussianBlur(gray,(0,0),4.2)
    if floating:
        dx=np.rint(cv2.Sobel(gray,cv2.CV_32F,1,0)*16).astype(np.int16)
        dy=np.rint(cv2.Sobel(gray,cv2.CV_32F,0,1)*16).astype(np.int16)
        raw=cv2.Canny(dx,dy,6*16,12*16,L2gradient=True)
    else:raw=cv2.Canny(gray,6,12,L2gradient=True)
    n,labels,stats,_=cv2.connectedComponentsWithStats(raw,8);keep=np.zeros(n,np.uint8)
    for k in range(1,n):
        if stats[k,4]>=40 and max(stats[k,2],stats[k,3])>=20:keep[k]=255
    return keep[labels],dict(components=n-1,rejected_components=int((keep[1:]==0).sum()),
                            raw_pixels=int((raw>0).sum()),retained_pixels=int((keep[labels]>0).sum()))


def main():
    out=BASE.parent/'b1-label-precision-audit-20261005';out.mkdir(exist_ok=False)
    cases=json.loads((BASE/'data.json').read_text(encoding='utf-8'))['validation'];records=[];images={}
    for r in cases:
        rgb=np.array(Image.open(r['target']).convert('RGB'));pair={}
        for name,floating in [('uint8',False),('float',True)]:
            im,stats=edge(rgb,floating);Image.fromarray(im).save(out/(r['id']+'_'+name+'.png'));images[(r['id'],name)]=im;pair[name]=stats
        records.append(dict(id=r['id'],target_sha256=file_hash(r['target']),results=pair))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);board=Image.new('RGB',(840,1256),'#17202a');draw=ImageDraw.Draw(board)
    for i,(key,box,label) in enumerate(CROPS):
        for j,(name,title) in enumerate([('uint8','旧：先取整'),('float','修正：保留梯度精度')]):
            board.paste(Image.fromarray(images[(key,name)]).convert('RGB').crop(box),(420*j,314*i+34));draw.text((420*j+5,314*i+4),label+' '+title,font=font,fill='white')
    board.save(out/'native_matched_comparison.png')
    write_json(out/'audit.json',dict(matched_parameters=dict(sigma=4.2,low=6,high=12,minimum_pixels=40,minimum_extent=20),
        changed_variable='Precision before gradient and nonmaximum suppression; float derivative encoded at x16',
        training_only=True,not_network_output=True,source_sha256=file_hash(__file__),records=records))
    print('LABEL_PRECISION_AUDIT',out)


if __name__=='__main__':main()
