"""Diagnostic only: can the material objective change pixels without a CNN?

These optimized images are NOT network predictions or deployable results.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch.nn import functional as F
from . import coherent_shared_train as base
from .coherent_surface_model import SurfaceAppearanceNet
from .coherent_surface_train import smooth
from .coherent_surface_prior import luminance
from .coherent_material_loss import texture_view
from .coherent_material_match import reference_targets,fixed_material_loss
from .losses import PerceptualLoss
from .runtime import configure_workspace,write_json


def main():
    root=configure_workspace();run=root/'runs/coherent-material-20261003';out=run/'pixel_diagnostic';out.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    samples=base.Samples(root/'runs/coherent-shared-20261002/data/manifest.json',20261003)
    state=torch.load(root/'runs/coherent-surface-20261002/final-4850/shared_style_fp32.pt',map_location='cpu',weights_only=True)
    net=SurfaceAppearanceNet(**state['config']).cuda().eval().requires_grad_(False);net.load_state_dict(state['generator'])
    vgg=PerceptualLoss(root).cuda();data=torch.load(run/'bank/material_bank.pt',map_location='cpu',weights_only=True)
    bank={k:v.cuda() for k,v in data['bank'].items()};records=[];font=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',18)
    for index in range(3):
        x,_,_,_,_,spec=samples.next()
        with torch.no_grad():
            anchor=net(x);ref=vgg.extract(texture_view(anchor));targets=reference_targets(ref,bank)
            initial=float(fixed_material_loss(ref,targets))
        value=torch.nn.Parameter(anchor.clone());opt=torch.optim.Adam([value],lr=.012)
        for step in range(80):
            features=vgg.extract(texture_view(value));yl,al=luminance(value),luminance(anchor)
            material=fixed_material_loss(features,targets)
            tone=F.l1_loss(F.avg_pool2d(value,16),F.avg_pool2d(anchor,16))
            chroma=F.l1_loss(value-yl,anchor-al);edge=[]
            for d,e in zip(base.differences(smooth(yl,3)),base.differences(smooth(al,3))):
                mask=e.abs()>.06;edge.append(((d-e).abs()*mask).sum()/(mask.sum()+1))
            loss=30*material+8*tone+4*chroma+.15*(value-anchor).abs().mean()+4*sum(edge)+.3*base.relative_mse(features['relu4_3'],ref['relu4_3'])
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            with torch.no_grad():value.clamp_(-1,127/128)
        with torch.no_grad():
            final=float(fixed_material_loss(vgg.extract(texture_view(value)),targets))
            mae=float((value-anchor).abs().mean()*128)
        images=[]
        for tensor in (x,anchor,value):
            array=(tensor.detach()[0].permute(1,2,0).cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8);images.append(Image.fromarray(array))
        w,h=images[0].size;board=Image.new('RGB',(w*3,h+40),'#111820');draw=ImageDraw.Draw(board)
        for i,pic in enumerate(images):
            board.paste(pic,(w*i,40));draw.text((w*i+5,5),('训练源','4850稳定CNN','逐图优化诊断·不是CNN产物')[i],font=font,fill='white')
        board.save(out/f'probe_{index}.jpg',quality=97)
        records.append({'sample':spec,'initial_material':initial,'optimized_material':final,'rgb_mae':mae,'steps':80,'artifact':str(out/f'probe_{index}.jpg')})
        print(json.dumps(records[-1],ensure_ascii=False),flush=True)
    write_json(out/'diagnostic.json',{'status':'IMAGE_OPTIMIZATION_DIAGNOSTIC_NOT_DEPLOYABLE','records':records,
      'limitation':'No temporal/generalization claim; per-image pixel optimization only tests the objective, not a trained model.'})


if __name__=='__main__':main()
