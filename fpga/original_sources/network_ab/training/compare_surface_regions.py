"""Lossless 720p region comparison for exact shared source frame 75.

Coordinates are diagnostic only. No spatial rule is used by inference.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .coherent_shared_model import SharedAppearanceNet
from .coherent_surface_model import SurfaceAppearanceNet
from .diagnose_coherent_surface import texture_stats,luma
from .runtime import file_hash,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--render',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    now=json.loads(a.render.read_text());old=json.loads(a.previous.read_text());models=[]
    for data in (old,now):
        if file_hash(data['weights'])!=data['weights_sha256']:raise ValueError('Weight drift')
        state=torch.load(data['weights'],map_location='cpu',weights_only=True)
        cls=SurfaceAppearanceNet if state['schema']=='surface-appearance-fp32-weights-v1' else SharedAppearanceNet
        net=cls(**state['config']).cuda().eval();net.load_state_dict(state['generator']);models.append(net)
    rois={'test_street':{'远树及天空':(340,180,790,390),'路面细纹':(300,490,680,690)},
          'holdout_corridor':{'石阶与接缝':(30,420,360,700)},
          'train_escalator':{'墙砖':(1010,280,1230,510)}}
    plots=[];records=[]
    with torch.no_grad():
        for row in now['clips']:
            if row['id'] not in rois:continue
            cap=cv2.VideoCapture(row['clip']);cap.set(cv2.CAP_PROP_POS_FRAMES,75);ok,bgr=cap.read();cap.release()
            if not ok:raise ValueError('Source decode failure')
            im=np.array(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((1280,720),Image.Resampling.BICUBIC))
            x=torch.from_numpy(im.copy()).permute(2,0,1).unsqueeze(0).cuda().float()/128-1
            results=[m(x)[0].permute(1,2,0).cpu().numpy()*128+128 for m in models]
            for name,(x1,y1,x2,y2) in rois[row['id']].items():
                source=im[y1:y2,x1:x2].astype(np.float32);before,after=[r[y1:y2,x1:x2] for r in results]
                plots.append((name,source,before,after))
                records.append({'scene':row['id'],'region':name,'xyxy_720':[x1,y1,x2,y2],
                  'source_luma':float(luma(source).mean()),'old_luma':float(luma(before).mean()),'new_luma':float(luma(after).mean()),
                  'old_texture':texture_stats(source,before),'new_texture':texture_stats(source,after)})
    font=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',20)
    board=Image.new('RGB',(1500,330*len(plots)),'#111820');draw=ImageDraw.Draw(board)
    for i,(name,*images) in enumerate(plots):
        for j,im in enumerate(images):
            tile=Image.fromarray(im.round().clip(0,255).astype(np.uint8))
            scale=min(480/tile.width,285/tile.height);tile=tile.resize((round(tile.width*scale),round(tile.height*scale)),Image.Resampling.NEAREST if scale>1 else Image.Resampling.LANCZOS)
            board.paste(tile,(j*500+10,i*330+35))
            draw.text((j*500+10,i*330+3),name+' · '+('原片','对照模型','本轮模型')[j],font=font,fill='white')
    board.save(a.render.parent/'surface_regions_comparison.jpg',quality=97)
    write_json(a.render.parent/'surface_regions.json',{'status':'DIAGNOSTIC_NOT_STYLE_PASS',
       'current_weights_sha256':now['weights_sha256'],'previous_weights_sha256':old['weights_sha256'],
       'frame':75,'records':records,'note':'Lower fine-detail amplitude may mean intended simplification or undesirable blur; inspect contours and full video.'})
    print(json.dumps(records,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
