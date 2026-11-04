"""Compare only fixed validation sources before choosing a video candidate."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch.nn import functional as F
from . import coherent_shared_train as base
from .coherent_surface_model import SurfaceAppearanceNet
from .coherent_material_loss import texture_view
from .coherent_material_match import reference_targets,fixed_material_loss
from .losses import PerceptualLoss
from .runtime import configure_workspace,file_hash,write_json


def main():
    root=configure_workspace();run=root/'runs/coherent-material-20261003';out=run/'validation_review';out.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    paths=[root/'runs/coherent-surface-20261002/final-4850/shared_style_fp32.pt',run/'train/checkpoint_000300.pt',run/'stronger_train/checkpoint_000300.pt']
    names=['稳定版4850','材质权重3 · 300步','材质权重30 · 300步'];models=[]
    for p in paths:
        state=torch.load(p,map_location='cpu',weights_only=True)
        cfg=state['config']['model'] if 'model' in state['config'] else state['config']
        model=SurfaceAppearanceNet(**cfg).cuda().eval();model.load_state_dict(state.get('ema',state['generator']));models.append(model)
    samples=base.Samples(root/'runs/coherent-shared-20261002/data/manifest.json',20261003)
    vgg=PerceptualLoss(root).cuda();bankstate=torch.load(run/'bank/material_bank.pt',map_location='cpu',weights_only=True)
    bank={k:v.cuda() for k,v in bankstate['bank'].items()};groups={};records=[]
    for row in samples.data['content']:
        if row['split']=='val':groups.setdefault(row['source'],[]).append(row)
    font=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',20)
    with torch.no_grad():
        for source,rows in sorted(groups.items()):
            row=rows[len(rows)//2];im=samples.image(row['path']);x=base.tensor(im)
            results=[m(x) for m in models];ref=vgg.extract(texture_view(results[0]));targets=reference_targets(ref,bank)
            board=Image.new('RGB',(1280,405*3),'#111820');draw=ImageDraw.Draw(board);metrics=[]
            crop=(720,460,1280,720) if '101046' in source else (80,150,640,410)
            for i,y in enumerate(results):
                rgb=(y[0].permute(1,2,0).cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)
                pic=Image.fromarray(rgb);board.paste(pic.resize((640,360),Image.Resampling.LANCZOS),(0,i*405+35))
                tile=pic.crop(crop);board.paste(tile,(670,i*405+70))
                draw.text((5,i*405+5),names[i]+' · 左全景 / 右720p原尺度局部',font=font,fill='white')
                metrics.append({'name':names[i],'material_loss':float(fixed_material_loss(vgg.extract(texture_view(y)),targets)),
                  'rgb_mae_vs_anchor':float((y-results[0]).abs().mean()*128),
                  'lowfreq_mae_vs_anchor':float(F.l1_loss(F.avg_pool2d(y,16),F.avg_pool2d(results[0],16))*128)})
                pic.save(out/f'{Path(source).stem}_{i}.png')
            path=out/f'{Path(source).stem}_candidates.jpg';board.save(path,quality=97)
            records.append({'source':row['path'],'source_sha256':row['sha256'],'crop_xyxy':crop,'board':str(path),'candidates':metrics})
    a=[json.loads(x)['sample'] for x in (run/'train/metrics_from_000000.jsonl').read_text().splitlines()]
    b=[json.loads(x)['sample'] for x in (run/'stronger_train/metrics_from_000000.jsonl').read_text().splitlines()]
    if len(a)!=300 or a!=b:raise ValueError('Ablation sample orders differ')
    write_json(out/'review.json',{'status':'VISUAL_SELECTION_REQUIRED','same_300_samples':True,'models':[{ 'path':str(p),'sha256':file_hash(p)} for p in paths],'records':records})
    print(json.dumps(records,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
