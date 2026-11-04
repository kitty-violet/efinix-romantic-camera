"""Freeze actual game feature targets and test sensitivity before training."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F
from . import coherent_shared_train as base
from .coherent_surface_model import SurfaceAppearanceNet
from .coherent_material_loss import LEVELS,texture_view,feature_rows,material_loss
from .losses import PerceptualLoss
from .runtime import configure_workspace,file_hash,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=configure_workspace();torch.set_num_threads(4);torch.manual_seed(20261003)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if (a.output/'material_bank.pt').exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True,exist_ok=True)
    manifest=root/'runs/coherent-shared-20261002/data/manifest.json'
    weights=root/'runs/coherent-surface-20261002/final-4850/shared_style_fp32.pt'
    samples=base.Samples(manifest,20261003);vgg=PerceptualLoss(root).cuda()
    bank={k:[] for k in LEVELS};records=[]
    with torch.no_grad():
        for i,row in enumerate(samples.styles):
            # Native reviewed 720p target crop: do not shrink a panorama to 512x288.
            im=samples.style(row);features=vgg.extract(texture_view(base.tensor(im)))
            for k in LEVELS:
                points=feature_rows(features[k],stride=3)
                ids=torch.linspace(0,len(points)-1,min(256,len(points)),device='cuda').long()
                bank[k].append(points[ids].cpu())
            records.append({'path':row['path'],'sha256':row['sha256'],'safe_roi_xyxy':row['safe_roi_xyxy'],'shape':list(im.shape)})
            if i%10==0:print('BANK',i+1,len(samples.styles),flush=True)
    bank={k:torch.cat(v) for k,v in bank.items()}
    binding={'manifest_sha256':file_hash(manifest),'vgg_sha256':vgg.weights_sha256,
      'loss_code_sha256':file_hash(Path(__file__).with_name('coherent_material_loss.py'))}
    torch.save({'schema':'reviewed-material-bank-v1','binding':binding,'bank':bank,'targets':records},a.output/'material_bank.pt')
    bank={k:v.cuda() for k,v in bank.items()}
    state=torch.load(weights,map_location='cpu',weights_only=True)
    net=SurfaceAppearanceNet(**state['config']).cuda().eval();net.load_state_dict(state['generator'])
    probes=[]
    with torch.no_grad():
        for i in range(8):
            x,_,_,_,_,spec=samples.next();y=net(x)
            blur=F.avg_pool2d(F.pad(y,(2,)*4,mode='reflect'),5,stride=1)
            variants={'current':y,'brightness_plus12':y+12/128,'contrast_08':y*.8,'blur5':blur}
            scores={k:float(material_loss(vgg.extract(texture_view(v)),bank)) for k,v in variants.items()}
            probes.append({'source':spec,'scores':scores})
    # Differentiability check on a real output, not synthetic success alone.
    test=y.detach().clone().requires_grad_(True)
    value=material_loss(vgg.extract(texture_view(test)),bank);gradient,=torch.autograd.grad(value,test)
    result={'status':'PROBE_COMPLETE_NOT_STYLE_PASS','binding':binding,'targets':len(records),
      'features':{k:list(v.shape) for k,v in bank.items()},'probes':probes,
      'gradient_l2':float(gradient.norm()),'gradient_finite':bool(torch.isfinite(gradient).all()),
      'mean_scores':{k:float(np.mean([r['scores'][k] for r in probes])) for k in variants},
      'note':'Brightness/contrast invariance prevents a pure grading shortcut; no claim that lower loss proves material quality.'}
    write_json(a.output/'probe.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('probes','binding')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
