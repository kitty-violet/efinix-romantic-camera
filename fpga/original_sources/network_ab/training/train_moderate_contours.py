"""Three small detail-reduction levels; no color or source-image reconstruction."""
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch import nn
from torch.nn import functional as F
from . import train_native_contours as base
from .train_context_contours import ContextContourNet
from .moderate_contour_targets import LEVELS,target
from .runtime import file_hash,write_json

ORIGINAL_CRITERION=base.criterion


class GrainContourNet(ContextContourNet):
    def __init__(self,difference=True):
        super().__init__(difference);self.edge=nn.Conv2d(8,3,1)


class Samples(base.Samples):
    def image(self,i,nxt=False):
        key=(i,nxt)
        if key not in self.cache:
            r=self.data['pairs'][i];a=base.load_rgb(r['next' if nxt else 'current']);b=base.load_rgb(r['next_target' if nxt else 'current_target'])
            self.cache[key]=(a,b,target(b))
        return self.cache[key]

    def batch(self,native=False):
        specs=[]
        for _ in range(1 if native else 2):
            i=self.rng.choice(self.groups[self.rng.choice(sorted(self.groups))]);nxt=bool(self.rng.randrange(2))
            size=0 if native else 256;x=0 if native else self.rng.randrange(1280-size+1)//8*8;y=0 if native else self.rng.randrange(720-size+1)//8*8
            specs.append((i,nxt,x,y,size))
        xs=[];ys=[]
        for i,nxt,x,y,size in specs:
            src,_,lab=self.image(i,nxt)
            if size:src=src[y:y+size,x:x+size];lab=lab[y:y+size,x:x+size]
            xs.append(base.tensor(src));ys.append(torch.from_numpy(lab.copy()).permute(2,0,1).float())
        return torch.stack(xs).cuda(),torch.stack(ys).cuda(),specs


def criterion(z,t,tracing=.35,margin=16):
    losses=[];terms=[]
    for c in range(z.shape[1]):
        a,b=ORIGINAL_CRITERION(z[:,c:c+1],t[:,c:c+1],tracing,margin);losses.append(a);terms.append(b)
    return torch.stack(losses).mean(),{k:sum(v[k] for v in terms)/len(terms) for k in terms[0]}


@torch.no_grad()
def evaluate(t,out,baseline=True):
    out.mkdir(exist_ok=False);records=[]
    for r in t.samples.data['validation']:
        a,b=base.load_rgb(r['source']),base.load_rgb(r['target']);x=base.tensor(a)[None].cuda()
        probs=t.net(x).sigmoid()[0].cpu().numpy();labels=target(b)
        for k,level in enumerate(LEVELS):
            p=probs[k];Image.fromarray(np.uint8(np.clip(1-p,0,1)*255)).save(out/(r['id']+'_'+level+'.png'))
            records.append(dict(id=r['id'],level=level,**base.scores(p,labels[:,:,k])))
    cases=[('val_stairs_090',(140,380,460,660)),('val_stairs_090',(420,180,740,460)),('val_plaza_090',(80,400,400,680))]
    board=Image.new('RGB',(1600,len(cases)*316),'#1b2430');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    for j,(key,box) in enumerate(cases):
        r=next(r for r in t.samples.data['validation'] if r['id']==key)
        paths=[r['target'],base.ROOT/'native-teacher-strokes-v1/eval-001600'/(key+'_probability.png')]+[out/(key+'_'+l+'.png') for l in LEVELS]
        for i,(name,p) in enumerate(zip(['教师图','封存旧版：实际预测','轻减细节：实际预测','中间档：实际预测','较强减细节：实际预测'],paths)):
            d.text((320*i+4,316*j+4),name,font=font,fill='white');board.paste(Image.open(p).convert('RGB').crop(box),(320*i,316*j+34))
    board.save(out/'comparison.png');write_json(out/'report.json',dict(step=t.step,records=records,actual_predictions=True,
        label_levels=LEVELS,no_prediction_filter=True,no_color_loss=True,no_source_reconstruction=True,approved_for_coloring=False))
    print('MODERATE_EVAL',t.step,[(l,round(np.mean([r['predicted_fraction'] for r in records if r['level']==l]),4)) for l in LEVELS],flush=True)


def main():
    base.NativeContourNet=GrainContourNet;base.Samples=Samples;base.criterion=criterion;base.evaluate=evaluate
    base.LABEL=dict(kind='teacher_coarse_structure_three_moderate_levels',parameters=LEVELS,
        only_training_targets=True,line_width_native720=1,no_color_reconstruction=True,user_acceptance_required_before_coloring=True)
    if '--name' not in sys.argv:sys.argv.extend(['--name','moderate-contour-grains-v1'])
    if '--scope' not in sys.argv:sys.argv.extend(['--scope','all'])
    if sys.argv[1]=='prepare':
        if '--initial' not in sys.argv:sys.argv.extend(['--initial',str(base.ROOT/'native-teacher-strokes-v1/checkpoint_001600.pt')])
        base.main();out=base.ROOT/sys.argv[sys.argv.index('--name')+1]
        cfg=json.loads((out/'config.json').read_text(encoding='utf-8'));state=torch.load(cfg['initial'],map_location='cpu',weights_only=True)
        model=state['model'];model['edge.weight']=model['edge.weight'].repeat(3,1,1,1);model['edge.bias']=model['edge.bias'].repeat(3)
        path=out/'initial_three_heads.pt';torch.save(dict(model=model,source=cfg['initial'],source_sha256=cfg['initial_sha256']),path)
        cfg.update(initial=str(path),initial_sha256=file_hash(path),graph='context90_native_contour_three_grains',lr=.0008,
            objective='coarse teacher structure only, no RGB reconstruction; three close levels between old and former A',levels=list(LEVELS))
        write_json(out/'config.json',cfg);binding=json.loads((out/'binding.json').read_text())
        for name in ['train_moderate_contours.py','train_context_contours.py','moderate_contour_targets.py','coarse_contour_targets.py']:
            p=Path(__file__).with_name(name);binding['code'][str(p)]=file_hash(p);shutil.copy2(p,out/'source'/p.name)
        binding['config']=file_hash(out/'config.json');write_json(out/'binding.json',binding);return
    base.main()


if __name__=='__main__':main()
