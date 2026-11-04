"""Describe changes without treating darkness, blur or mean errors as quality passes."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from .runtime import file_hash,write_json


def main():
    cv2.setNumThreads(4)
    p=argparse.ArgumentParser();p.add_argument('--render',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True);a=p.parse_args()
    now=json.loads(a.render.read_text());old=json.loads(a.previous.read_text())
    current_audit=json.loads((a.render.parent/'audit.json').read_text())
    old_audit=json.loads((a.previous.parent/'audit.json').read_text())
    if current_audit['render_sha256']!=file_hash(a.render) or old_audit['render_sha256']!=file_hash(a.previous):
        raise ValueError('Audit is not bound to renders')
    old_clips={r['id']:r for r in old['clips']};old_metrics={r['id']:r for r in old_audit['clips']}
    new_metrics={r['id']:r for r in current_audit['clips']};records=[]
    for row in now['clips']:
        if row['id'] not in old_clips:continue
        prev=old_clips[row['id']]
        if row['clip_sha256']!=prev['clip_sha256']:raise ValueError('Different source timeline')
        ca,cb=cv2.VideoCapture(prev['output']),cv2.VideoCapture(row['output'])
        vals={k:[] for k in ('new_vs_old_mae','old_luma','new_luma','old_saturation','new_saturation',
                            'old_contrast_normalized_detail','new_contrast_normalized_detail')}
        for index in range(150):
            oa,fa=ca.read();ob,fb=cb.read()
            if not oa or not ob:raise ValueError('Truncated timeline')
            fa=cv2.resize(fa,(1280,720),interpolation=cv2.INTER_AREA)
            fb=cv2.resize(fb,(1280,720),interpolation=cv2.INTER_AREA)
            vals['new_vs_old_mae'].append(float(np.abs(fa.astype(np.float32)-fb).mean()))
            for label,im in (('old',fa),('new',fb)):
                y=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
                vals[label+'_luma'].append(float(y.mean()))
                vals[label+'_saturation'].append(float(cv2.cvtColor(im,cv2.COLOR_BGR2HSV)[:,:,1].mean()))
                hp=y-cv2.GaussianBlur(y,(0,0),1.2)
                vals[label+'_contrast_normalized_detail'].append(float(hp.std()/max(y.std(),1.)))
        ca.release();cb.release()
        metrics={k:{'mean':float(np.mean(v)),'min':float(min(v)),'max':float(max(v))} for k,v in vals.items()}
        worst=int(np.argmax(new_metrics[row['id']]['per_frame_metrics']['appearance_change_motion_error']))+1
        board=Image.new('RGB',(1920,1100),'#111');draw=ImageDraw.Draw(board)
        for j,frame in enumerate((max(0,worst-1),worst)):
            for k,(label,path) in enumerate((('SOURCE',row['clip']),('PREVIOUS',prev['output']),('NEW',row['output']))):
                cap=cv2.VideoCapture(path);cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,pic=cap.read();cap.release()
                if not ok:raise ValueError('Cannot inspect worst frame')
                im=Image.fromarray(cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)).resize((640,360),Image.Resampling.LANCZOS)
                board.paste(im,(k*640,j*550+20));draw.text((k*640+5,j*550+3),f'{label} frame={frame} ({frame/30:.3f}s)',fill='white')
        # Keep output compact while preserving 16:9 thumbnails.
        compact=Image.new('RGB',(1920,760),'#111')
        compact.paste(board.crop((0,0,1920,380)),(0,0));compact.paste(board.crop((0,550,1920,930)),(0,380))
        compact.save(a.render.parent/f"{row['id']}_worst_motion_pair.jpg",quality=95)
        records.append({'id':row['id'],'appearance':metrics,'per_frame':vals,
                        'old_temporal':old_metrics[row['id']]['metrics'],
                        'new_temporal':new_metrics[row['id']]['metrics'],'worst_new_appearance_motion_frame':worst})
    write_json(a.render.parent/'version_comparison.json',{'schema':'shared-version-comparison-v1',
        'current_render_sha256':file_hash(a.render),'previous_render_sha256':file_hash(a.previous),
        'clips':records,'interpretation':'Descriptive changes only. Lower temporal error can come from lost detail; stronger color change is not proof of stronger game style. No automatic style pass.'})
    print('COMPARED',len(records),'exact source timelines',flush=True)


if __name__=='__main__':main()
