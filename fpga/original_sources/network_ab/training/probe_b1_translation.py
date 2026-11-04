"""Controlled source translations isolate stride-phase sensitivity of trained B1."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from network_ab.contract_torch import build_fp32_model
from .runtime import configure_workspace,write_json,file_hash


def main():
    root=configure_workspace();run=root/'runs/b1-distillation-20261003';torch.set_num_threads(2)
    path=run/'final-2000/b1_style_fp32.pt';net=build_fp32_model().cpu().eval()
    net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model']);records=[]
    with torch.inference_mode():
        for name in ['val_stairs_090','val_plaza_210']:
            rgb=np.array(Image.open(run/'data'/(name+'_source.png')))
            x=torch.from_numpy(rgb).permute(2,0,1)[None].float()/128-1
            reference=net(x)
            for shift in [1,2,3,4]:
                y=net(F.pad(x,(shift,0,0,0),mode='replicate')[:,:,:,:1280])
                diff=(y[:,:,:,shift:]-reference[:,:,:,:-shift]).abs()[:,:,80:-80,80:-80]*128
                v=diff.flatten().numpy()
                row={'case':name,'horizontal_shift_px':shift,'aligned_mae_rgb255':float(v.mean()),
                     'p95_rgb255':float(np.percentile(v,95)),'max_rgb255':float(v.max())}
                if shift==4:assert row['max_rgb255']<.01
                records.append(row)
    result={'status':'CONTROLLED_TRANSLATION_PROBE_COMPLETE','records':records,'weights_sha256':file_hash(path),
            'method':'Shift the same source right by 1/2/3/4 pixels, run single-frame B1, align outputs, exclude 80px borders. CPU FP32.',
            'scope':'Measures encoder stride phase sensitivity, not actual video flicker acceptance or a proof no training can improve it.'}
    write_json(run/'translation_probe.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
