"""Probe phase-dependent survival of thin structures in actual B1 bytes."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .audit_b1_sampling_v2 import FALLBACK
from .train_b1_learned_colour import IntegerB1,runtime_from_image
from .runtime import configure_workspace,file_hash,write_json


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    configure_workspace();a.out.mkdir(exist_ok=True);assert not (a.out/'thin_structures.json').exists()
    report=[]
    for name,path in [('v1',FALLBACK/'step_001800/coefficients-padded.bin'),('candidate',a.package)]:
        net=IntegerB1(runtime_from_image(path.read_bytes()))
        uniform=torch.stack([torch.full((3,160,160),v,device='cuda',dtype=torch.float32) for v in (-96,64)])
        q=net(uniform);y=(q[:,0]+2*q[:,1]+q[:,2])/4
        black,white=[float(v) for v in y[:,60:100,60:100].mean((1,2))];denom=max(1,white-black)
        records=[]
        for axis in ('x','y'):
            for width in (1,2,3,4,6,8,12,16):
                batch=torch.full((4,3,160,160),64.,device='cuda')
                for phase in range(4):
                    if axis=='x':batch[phase,:,:,72+phase:72+phase+width]=-96
                    else:batch[phase,:,72+phase:72+phase+width,:]=-96
                q=net(batch);y=(q[:,0]+2*q[:,1]+q[:,2])/4
                if axis=='y':y=y.transpose(-1,-2)
                profiles=y[:,60:100].mean(1).cpu().numpy()
                retention=[max(0,float((white-v[60:115].min())/denom)) for v in profiles]
                records.append(dict(axis=axis,width_720=width,contrast_retention_by_phase=retention,
                    minimum=min(retention),maximum=max(retention),phase_spread=max(retention)-min(retention)))
        report.append(dict(name=name,package=str(path),sha256=file_hash(path),records=records))
        print(name,[(w,round(min(r['minimum'] for r in records if r['width_720']==w),3),
                         round(max(r['maximum'] for r in records if r['width_720']==w),3)) for w in (1,2,3,4,6,8,12,16)],flush=True)
    write_json(a.out/'thin_structures.json',dict(results=report,
        caveat='Synthetic high-contrast line survival is weight/colour dependent. Not a universal resolution or natural-image quality claim.'))


if __name__=='__main__':main()
