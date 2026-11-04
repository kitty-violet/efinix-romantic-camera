"""Learn a painting-domain B1 codec, then map source images into its codes.

Stage A reconstructs the existing TRAIN paintings: verifies colour levels and
subpixel contours through the actual sampling graph before source translation.
Stage B freezes that decoder and supervises source encodings with the same
codec's target encodings. No fixed colour lattice, binary contour fields, new
runtime operators or postfilter. Target-input images are diagnostic only.
"""
from pathlib import Path
import copy
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont
from . import train_b1_minimum_area as m
from . import train_b1_late_region_supervision as driver
from .train_b1_target_latent import TargetLatentTraining
from .train_b1_native_coverage_counts import planned_native
from .train_b1_joint_region_colour import joint_region_loss
from .train_b1_clean_regions import CleanRegionTraining


def codec_loss(p,t):
    colour=F.l1_loss(p,t)
    mse=F.mse_loss(p,t)
    edges=[]
    for dim in (-1,-2):
        gp,gt=torch.diff(p,dim=dim),torch.diff(t,dim=dim)
        # Native signed transition, including target edge coverage. Both
        # staircase noise and over-wide transitions differ from this target.
        w=(gt.abs().amax(1,keepdim=True)>3/128).float()
        edges.append(((gp-gt).abs()*(1+4*w)).mean())
    edge=torch.stack(edges).mean()
    return colour+mse+edge,dict(codec_colour=colour,codec_mse=mse,codec_transition=edge)


class PaintingCodecTraining(m.AreaTraining):
    preflight_native_resume=True

    def __init__(self,args,data):
        assert not args.freeze_head and not args.class_manifest
        assert args.temporal==0,'Target reconstruction has no source-flow objective'
        super().__init__(args,data)
        self.opt.param_groups[1]['lr']=.002
        self.base_lrs=[g['lr'] for g in self.opt.param_groups]

    def batch_style(self,native):
        _,target,specs,flow,valid=super().batch_style(native)
        assert flow is None
        return target,target,specs,None,None

    @staticmethod
    def appearance_loss(p,t,mode):return codec_loss(p,t)

    def additional_loss(self,p,t,native):return p.sum()*0,{}

    def update(self,native=None):
        return super().update(planned_native(self.step) if native is None else native)

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(painting_codec=True,source_to_style=False,visual_acceptance=False,
            independent_colour_fields=False,inference_added_operators=0)
        torch.save(state,path)


class PaintingCodeTransferTraining(TargetLatentTraining):
    preflight_native_resume=True
    input_stability_loss=CleanRegionTraining.input_stability_loss

    def __init__(self,args,data):
        state=torch.load(args.teacher_checkpoint or args.warm,map_location='cpu',weights_only=True)
        assert state.get('painting_codec'),'Do not silently use a photographic reconstruction teacher'
        super().__init__(args,data)
        self.opt.param_groups[1]['lr']=.002
        self.base_lrs=[g['lr'] for g in self.opt.param_groups]

    @staticmethod
    def appearance_loss(p,t,mode):return p.sum()*0,{}

    def additional_loss(self,p,t,native):
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        region,values=joint_region_loss(p,t,labels)
        with torch.no_grad():wanted=self.teacher(self.full_target,all_nodes=True)['res3_add']
        actual=self.nodes['res3_add']
        if not native:actual=actual[...,10:-10,10:-10];wanted=wanted[...,10:-10,10:-10]
        scale=wanted.std((0,2,3),keepdim=True).clamp_min(4).detach()
        code=F.smooth_l1_loss(actual/scale,wanted/scale,beta=.5)
        edges=sum(F.l1_loss(torch.diff(actual/scale,dim=d),torch.diff(wanted/scale,dim=d)) for d in (-1,-2))/2
        values.update(painting_code=code,painting_code_transition=edges)
        return region+self.args.classification*(code+.25*edges),values

    def update(self,native=None):
        return super().update(planned_native(self.step) if native is None else native)

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(painting_code_transfer=True,source_to_style=True,visual_acceptance=False,
            decoder_frozen=True,inference_added_operators=0,minimum_area_inference_guarantee=False)
        torch.save(state,path)


@torch.inference_mode()
def evaluate_codec(net,rows):
    records=[]
    for row in rows:
        t=m.t.r.tensor(m.t.r.read(row['target']))[None];p=net(t)[0]
        loss,terms=codec_loss(p,t)
        records.append(dict(id=row['id'],**m.t.r.metrics(p,t),**{k:float(v) for k,v in terms.items()},score=float(loss)))
    return dict(records=records,mean={k:sum(r[k] for r in records)/len(records) for k in records[0] if k!='id'},
                source_to_style=False,diagnostic_target_input=True)


@torch.inference_mode()
def preview_codec(net,rows,dest):
    dest.mkdir(exist_ok=True);images={}
    for row in rows:
        t=m.t.r.tensor(m.t.r.read(row['target']))[None];im=m.t.r.c.rgb_image(net(t)[0][0])
        im.save(dest/(row['id']+'_target_reconstruction.png'))
        images[row['id']]=[Image.open(row['target']),im]
    board=Image.new('RGB',(840,314*len(m.t.r.c.CROPS)),'#17202a');d=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(im,title) in enumerate(zip(images[key],('绘画目标输入·内部诊断','目标重建·不是原图风格推理'))):
            board.paste(im.crop(box),(420*j,314*i+34));d.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png')


if __name__=='__main__':
    import sys
    if '--transfer' in sys.argv:
        sys.argv.remove('--transfer')
        from .train_b1_joint_region_colour import evaluate,preview
        driver.main(training_factory=PaintingCodeTransferTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
    else:
        driver.main(training_factory=PaintingCodecTraining,extra_sources=[Path(__file__)],evaluation=evaluate_codec,preview=preview_codec)
