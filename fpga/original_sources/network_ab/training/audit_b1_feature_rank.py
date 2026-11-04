"""Training-source activation diversity audit, not a capacity proof.

Covariance rank and channel activity only identify a possible bottleneck;
neither a low rank nor many live channels proves representational sufficiency.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .runtime import file_hash,write_json


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
    samples={};ids=[];rng=np.random.default_rng(20261006309)
    rows=m.load_data()['training'][::12]
    keys=['stem','down1.pw','down2.pw','res1_conv1.pw','res1_add',
          'res2_conv1.pw','res2_add','res3_conv1.pw','res3_add','up1_low_dw','up1_low_pw']
    for row in rows:
        assert row['split']=='train';ids.append(row['id'])
        x=int(rng.integers(0,961));y=int(rng.integers(0,401))
        image=m.t.r.read(row['path'])[y:y+320,x:x+320]
        image=torch.tensor(image.transpose(2,0,1)[None].copy(),dtype=torch.float32)/128-1
        nodes=net(image,all_nodes=True)
        for k in keys:
            v=nodes[k][0,:,4:-4:3,4:-4:3].flatten(1).T.double()
            samples.setdefault(k,[]).append(v)
    report=[]
    for k,items in samples.items():
        z=torch.cat(items);mean=z.mean(0);std=z.std(0)
        v=z-mean;cov=v.T@v/(len(v)-1)
        eig=torch.linalg.eigvalsh(cov).clamp_min(0).flip(0)
        mass=eig/eig.sum().clamp_min(1e-20)
        record=dict(node=k,channels=z.shape[1],samples=len(z),
            std=std.tolist(),always_zero=(z.abs().amax(0)==0).nonzero().flatten().tolist(),
            channels_std_below1=int((std<1).sum()),
            pcs_95=int((mass.cumsum(0)<.95).sum())+1,
            pcs_99=int((mass.cumsum(0)<.99).sum())+1,
            participation_rank=float(eig.sum().square()/eig.square().sum().clamp_min(1e-20)),
            zero_fraction=(z==0).double().mean(0).tolist())
        report.append(record)
        print('FEATURE_DIVERSITY',{q:v for q,v in record.items() if q not in ('std','zero_fraction')},flush=True)
    write_json(a.out,dict(checkpoint=a.checkpoint,sha256=file_hash(a.checkpoint),
        source_sha256=file_hash(Path(__file__)),training_ids=ids,records=report,
        validation_used=False,capacity_proof=False))


if __name__=='__main__':main()
