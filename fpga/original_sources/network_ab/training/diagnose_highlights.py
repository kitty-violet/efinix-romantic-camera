"""Trace highlight artifacts against the exact source frames before changing style."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .runtime import configure_workspace,write_json,file_hash


def luminance(a):return a[:,:,0]*.2126+a[:,:,1]*.7152+a[:,:,2]*.0722


def main():
    root=configure_workspace();old=root/'runs/texture-recovery-20261003';out=root/'runs/highlight-diagnosis-20261003';out.mkdir(exist_ok=True)
    report=json.loads((old/'refined_videos/report.json').read_text(encoding='utf-8'))
    row=next(r for r in report['records'] if r['id']=='new_hallway')
    assert file_hash(row['source'])==row['source_sha256'] and file_hash(row['output'])==row['output_sha256']
    src=cv2.VideoCapture(row['source']);dst=cv2.VideoCapture(row['output']);records=[];samples=[]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    for ix in range(360):
        aok,a=src.read();bok,b=dst.read();assert aok and bok
        a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB);b=cv2.cvtColor(b,cv2.COLOR_BGR2RGB)
        a=np.array(Image.fromarray(a).resize((1280,720),Image.Resampling.BICUBIC));b=np.array(Image.fromarray(b).resize((1280,720),Image.Resampling.BICUBIC))
        la,lb=luminance(a.astype(float)),luminance(b.astype(float))
        ma=(la>=240);mb=(lb>=240);created=mb&(la<230)
        count,_,stats,_=cv2.connectedComponentsWithStats(created.astype(np.uint8),8)
        bands=[]
        for lo,hi in [(0,128),(128,180),(180,210),(210,230),(230,240),(240,256)]:
            mask=(la>=lo)&(la<hi);bands.append({'source_range':[lo,hi],'fraction':float(mask.mean()),'mean_output':float(lb[mask].mean()) if mask.any() else None,'mean_boost':float((lb-la)[mask].mean()) if mask.any() else None})
        records.append({'frame':ix,'source_Y240_fraction':float(ma.mean()),'output_Y240_fraction':float(mb.mean()),'new_Y240_from_source_below230':float(created.mean()),'max_new_component_pixels':int(stats[1:,4].max()) if count>1 else 0,'bands':bands})
        if ix%30==0 or ix==359:
            Image.fromarray(a).save(out/f'source_{ix:03d}.png');Image.fromarray(b).save(out/f'output_{ix:03d}.png');samples.append((ix,a,b))
    src.release();dst.release()
    for start in range(0,len(samples),4):
        page=samples[start:start+4];board=Image.new('RGB',(1280,400*len(page)),'#161b22');d=ImageDraw.Draw(board)
        for j,(ix,a,b) in enumerate(page):
            d.text((8,j*400+4),f'{ix/30:.1f}s 原片（既有HDR→SDR）',font=font,fill='white');d.text((648,j*400+4),'用户认可版本：网络结果',font=font,fill='white')
            board.paste(Image.fromarray(a).resize((640,360)),(0,j*400+40));board.paste(Image.fromarray(b).resize((640,360)),(640,j*400+40))
        board.save(out/f'comparison_{start//4}.jpg',quality=94)
    write_json(out/'audit.json',{'status':'SOURCE_OUTPUT_AUDIT_COMPLETE','source':row,'records':records,'samples':[r[0] for r in samples],'note':'Encoded source/output aligned; thresholds characterize brightness, not scene radiance or clipping proof.'})
    worst=max(records,key=lambda r:r['new_Y240_from_source_below230'])
    print('AUDIT_COMPLETE',json.dumps({'worst':worst,'mean_source_white':np.mean([r['source_Y240_fraction'] for r in records]),'mean_output_white':np.mean([r['output_Y240_fraction'] for r in records]),'mean_new_white':np.mean([r['new_Y240_from_source_below230'] for r in records])}),flush=True)


if __name__=='__main__':main()
