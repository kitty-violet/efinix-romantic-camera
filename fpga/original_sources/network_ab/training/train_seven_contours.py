"""Seven contour scales with explicit penalties for stray network responses.

Endpoints are the user's light/strong teacher-label candidates, unchanged.
No RGB loss, thresholding, morphology or cleanup is used at model inference.
"""
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
import torch
from torch import nn
from torch.nn import functional as F
from . import train_moderate_contours as runner
from .coarse_contour_targets import target_parameters
from .train_teacher_strokes import criterion as amplitude_loss
from .runtime import file_hash,write_json

LEVELS={f'g{i+1}':dict(sigma=.65+.85*i/6,low=round(36-8*i/6),high=round(72-16*i/6),
    min_component_pixels=round(12+12*i/6),min_extent=round(6+4*i/6)) for i in range(7)}


def target(rgb):
    return np.stack([target_parameters(rgb,v) for v in LEVELS.values()],-1).astype(np.uint8)


class SevenContourNet(runner.GrainContourNet):
    def __init__(self,difference=True):
        super().__init__(difference);self.edge=nn.Conv2d(8,7,1)


def criterion(logit,t,tracing=.35,margin=16):
    p=logit[...,margin:-margin,margin:-margin].sigmoid();q=t[...,margin:-margin,margin:-margin]
    loss,terms=amplitude_loss(logit,t,margin=margin)
    # Average background loss previously diluted the sparse but visible false
    # edges. Penalize the strongest wrong responses, as many as true pixels.
    hard=[]
    for b in range(p.shape[0]):
        for c in range(p.shape[1]):
            wrong=(p[b,c]*(1-q[b,c])).flatten()
            k=max(64,int(q[b,c].sum().detach()))
            hard.append(wrong.topk(min(k,wrong.numel())).values.mean())
    stray=torch.stack(hard).mean()
    mass=(p.mean((-1,-2))-q.mean((-1,-2))).abs().mean()
    total=loss+1.25*stray+2*mass
    return total,dict(**terms,stray_peak=float(stray.detach()),response_mass=float(mass.detach()))


@torch.no_grad()
def evaluate(t,out,baseline=True):
    out.mkdir(exist_ok=False);records=[];base=runner.base
    for row in t.samples.data['validation']:
        rgb=base.load_rgb(row['target']);labels=target(rgb)
        probs=t.net(base.tensor(base.load_rgb(row['source']))[None].cuda()).sigmoid()[0].cpu().numpy()
        for k,level in enumerate(LEVELS):
            p=probs[k];q=labels[:,:,k]
            Image.fromarray(np.uint8((1-p)*255)).save(out/(row['id']+'_'+level+'.png'))
            Image.fromarray(np.uint8((1-q)*255)).save(out/(row['id']+'_'+level+'_target.png'))
            near=cv2.dilate(q,np.ones((3,3),np.uint8))>0
            records.append(dict(id=row['id'],level=level,**base.scores(p,q),
                background_response_mean=float(p[~near].mean()),target_line_response_mean=float(p[q>0].mean()),
                total_probability=float(p.mean()),target_density=float(q.mean())))
    cases=[('val_stairs_090',(140,380,460,660)),('val_stairs_090',(420,180,740,460)),('val_plaza_090',(80,400,400,680))]
    board=Image.new('RGB',(2560,948),'#1b2430');draw=ImageDraw.Draw(board)
    pairs=Image.new('RGB',(2240,1896),'#1b2430');pd=ImageDraw.Draw(pairs)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    for j,(key,box) in enumerate(cases):
        paths=[base.ROOT/'native-teacher-strokes-v1/eval-001600'/(key+'_probability.png')]+[out/(key+'_'+l+'.png') for l in LEVELS]
        for i,p in enumerate(paths):
            draw.text((320*i+4,316*j+4),'旧版实际预测' if i==0 else f'{i}档实际预测',font=font,fill='white')
            board.paste(Image.open(p).convert('RGB').crop(box),(320*i,316*j+34))
        for i,l in enumerate(LEVELS):
            for z,suffix in enumerate(['_target','']):
                yy=632*j+316*z;pd.text((320*i+4,yy+4),f'{i+1}档'+('目标' if z==0 else '实际预测'),font=font,fill='white')
                pairs.paste(Image.open(out/(key+'_'+l+suffix+'.png')).convert('RGB').crop(box),(320*i,yy+34))
    board.save(out/'comparison.png');pairs.save(out/'targets_vs_predictions.png')
    write_json(out/'report.json',dict(step=t.step,records=records,actual_predictions=True,levels=LEVELS,
        prediction_postprocessing=False,no_color_loss=True,reviewed_validation_not_blind=True))
    print('SEVEN_EVAL',t.step,[(l,round(np.mean([r['background_response_mean'] for r in records if r['level']==l]),4)) for l in LEVELS],flush=True)


def main():
    base=runner.base;base.NativeContourNet=SevenContourNet;base.Samples=runner.Samples;base.criterion=criterion
    runner.LEVELS=LEVELS;runner.target=target;base.evaluate=evaluate
    base.LABEL=dict(kind='seven_teacher_structure_levels',parameters=LEVELS,label_only=True,
        endpoints='moderate-contour-target-screen-v1 light/strong unchanged',no_RGB_loss=True)
    if '--name' not in sys.argv:sys.argv.extend(['--name','seven-contour-clean-v1'])
    if '--scope' not in sys.argv:sys.argv.extend(['--scope','all'])
    if sys.argv[1]=='prepare':
        if '--initial' not in sys.argv:sys.argv.extend(['--initial',str(base.ROOT/'native-teacher-strokes-v1/checkpoint_001600.pt')])
        base.main();out=base.ROOT/sys.argv[sys.argv.index('--name')+1]
        cfg=json.loads((out/'config.json').read_text(encoding='utf-8'));state=torch.load(cfg['initial'],map_location='cpu',weights_only=True)
        model=state['model'];model['edge.weight']=model['edge.weight'].repeat(7,1,1,1);model['edge.bias']=model['edge.bias'].repeat(7)
        path=out/'initial_seven_heads.pt';torch.save(dict(model=model,source=cfg['initial'],source_sha256=cfg['initial_sha256']),path)
        cfg.update(initial=str(path),initial_sha256=file_hash(path),graph='context90_native_contour_seven_grains',lr=.0003,
            objective='teacher contours with explicit strongest false-response penalty; no original RGB reconstruction',levels=list(LEVELS))
        write_json(out/'config.json',cfg);binding=json.loads((out/'binding.json').read_text())
        for name in ['train_seven_contours.py','train_moderate_contours.py','train_context_contours.py','moderate_contour_targets.py','coarse_contour_targets.py','train_teacher_strokes.py']:
            p=Path(__file__).with_name(name);binding['code'][str(p)]=file_hash(p);shutil.copy2(p,out/'source'/p.name)
        binding['config']=file_hash(out/'config.json');write_json(out/'binding.json',binding);return
    base.main()


if __name__=='__main__':main()
