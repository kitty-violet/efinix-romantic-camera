"""Actual-byte response to tiny input changes without any scene motion."""
import argparse
import hashlib
import torch
from . import train_b1_minimum_area as m
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',nargs='+',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    configure_workspace();set_reproducible(20261006219);rows=m.load_data()['validation'];results=[]
    for path in a.checkpoints:
        state=torch.load(path,map_location='cpu',weights_only=True);q=m.t.r.initialize();q.load_state_dict(state['model']);q.quantized=True
        q.package['residual_encoding']=state['residual_encoding'];raw=q.export();net=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw));del q
        records=[]
        for row in (rows[2],rows[4],rows[5]):
            x=m.t.r.tensor(m.t.r.read(row['path']))[None]*128;y=net(x)
            generator=torch.Generator(device='cuda').manual_seed(314)
            variations={'exposure+1':(x+1).clamp(-128,127),'exposure-1':(x-1).clamp(-128,127),
                        'noise0.5':(x+torch.randn(x.shape,generator=generator,device='cuda')*.5).round().clamp(-128,127)}
            for name,z in variations.items():
                error=(net(z)-y).abs().mean(1)[...,80:-80,80:-80]
                records.append(dict(id=row['id'],perturbation=name,mean=float(error.mean()),
                    p95=float(torch.quantile(error.flatten(),.95)),over16=float((error>16).float().mean())))
        result=dict(checkpoint=path,checkpoint_sha256=file_hash(path),package_sha256=hashlib.sha256(raw).hexdigest(),records=records)
        result['mean']={k:sum(v[k] for v in records)/len(records) for k in ('mean','p95','over16')};results.append(result)
        print('INPUT_NOISE_AUDIT',path,result['mean'],flush=True)
        del net;torch.cuda.empty_cache()
    write_json(a.output,dict(results=results,actual_bytes=True,scope='No motion, exact integer source perturbations; not full video acceptance',visual_acceptance=False))


if __name__=='__main__':main()
