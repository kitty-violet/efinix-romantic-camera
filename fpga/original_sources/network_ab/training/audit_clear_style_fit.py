"""Check learned target fit on reviewed training and held-out regression images."""
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from .coherent_surface_model import SurfaceAppearanceNet
from .runtime import configure_workspace, write_json


def main():
    root=configure_workspace();run=root/'runs/clear-style-control-20261003';out=run/'fit_audit';out.mkdir(exist_ok=True)
    torch.set_num_threads(3);cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state=torch.load(run/'student_train/checkpoint_002000.pt',map_location='cpu',weights_only=True)
    net=SurfaceAppearanceNet(**state['config']['model']).cuda().eval();net.load_state_dict(state['ema'])
    manifest=json.loads((run/'paired_data/manifest.json').read_text(encoding='utf-8'))
    inputs=[]
    for ix in [3,24,34,48,62,74,98]:
        row=next(x for x in manifest['pairs'] if x['parent_pair_index']==ix)
        inputs.append((f'train_{ix:03d}','train',row['current'],row['current_target']))
    for case in ['quiet_stairs','plaza_stone','park_people']:
        inputs.append((case,'reviewed_validation',run/'anime'/case/'source.png',run/'guard_review'/(case+'.png')))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);rows=[]
    for name,split,source,target in inputs:
        a=np.array(Image.open(source).convert('RGB'));b=np.array(Image.open(target).convert('RGB'))
        x=torch.from_numpy(a.copy()).permute(2,0,1)[None].cuda().float()/128-1
        with torch.no_grad():y=net(x)
        c=(y[0].permute(1,2,0).cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)
        Image.fromarray(c).save(out/(name+'_student.png'))
        aa,bb,cc=[v.astype(np.float32) for v in (a,b,c)]
        high=lambda v:v-cv2.GaussianBlur(v,(0,0),1.5)
        rows.append({'case':name,'split':split,'identity_target_mae':float(np.abs(aa-bb).mean()),
                     'student_target_mae':float(np.abs(cc-bb).mean()),'student_source_mae':float(np.abs(cc-aa).mean()),
                     'identity_target_highpass_mae':float(np.abs(high(aa)-high(bb)).mean()),
                     'student_target_highpass_mae':float(np.abs(high(cc)-high(bb)).mean())})
        board=Image.new('RGB',(1200,265),(22,26,32));d=ImageDraw.Draw(board)
        for i,(label,im) in enumerate(zip(['原图','动画教师','2000步CNN'],[a,b,c])):
            d.text((i*400+8,4),name+' / '+label,font=font,fill='white')
            board.paste(Image.fromarray(im).resize((400,225),Image.Resampling.LANCZOS),(i*400,40))
        board.save(out/(name+'_overview.jpg'),quality=94)
    write_json(out/'report.json',{'step':state['step'],'records':rows,'note':'Pixel/highpass target fit only; training examples cannot establish generalization, and these metrics are not aesthetic quality.'})
    print(json.dumps(rows,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
