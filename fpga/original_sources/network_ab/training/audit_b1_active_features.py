"""CPU-only audit of nonlinear activity in a saved integer checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .train_b1_learned_colour import ColourB1
from .probe_b1_cartoon_int8 import IntegerB1
from .b1_trained_image import runtime_from_image
from .runtime import write_json,file_hash


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[2]/'model-training-workspace/runs'
    net=ColourB1(revive=False,source=root/'b1-reconstruction-research-20261005/delivery/coefficients-padded.bin')
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);net.load_state_dict(state['model']);net.quantized=True
    if 'residual_encoding' in state:net.package['residual_encoding']=state['residual_encoding']
    actual=IntegerB1(runtime_from_image(net.export()),device='cpu')
    data=json.loads((root/'b1-minimum-area-20261006/data.json').read_text(encoding='utf-8'))
    rows=data['training'];selected=[];seen=set()
    for r in rows:
        if r['video'] not in seen:selected.append(r);seen.add(r['video'])
    keys=['stem','down1.pw','down2.pw','res1_conv1.pw','res2_conv1.pw','res3_conv1.pw','res3_add']
    records=[]
    for row in selected:
        rgb=np.array(Image.open(row['path']).convert('RGB'))[200:456,512:768]
        x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)-128
        nodes,_=actual(x,True)
        for key in keys:
            z=nodes[key][0,:,4:-4,4:-4].flatten(1)
            centered=z-z.mean(1,keepdim=True);sv=torch.linalg.svdvals(centered)
            energy=sv.square().cumsum(0)/sv.square().sum().clamp_min(1e-12)
            gate=((z>0)&(z<127)).float().mean(1)
            records.append(dict(id=row['id'],node=key,rank99=int(torch.searchsorted(energy,.99))+1,channels=len(z),
                always_linear_channels=int((gate>.995).sum()),zero_fraction=float((z==0).float().mean()),
                saturation_fraction=float((z.abs()>=127).float().mean()),channel_std=z.std(1).tolist()))
    summary={k:{metric:float(np.mean([r[metric] for r in records if r['node']==k])) for metric in
        ('rank99','channels','always_linear_channels','zero_fraction','saturation_fraction')} for k in keys}
    write_json(a.out,dict(checkpoint=a.checkpoint,checkpoint_sha256=file_hash(a.checkpoint),cpu_only=True,
        training_crop=(512,200,768,456),source_ids=[r['id'] for r in selected],records=records,summary=summary,
        note='Rank/activity are diagnostics, not a proof of learnability or visual success.'))
    print('ACTIVITY_CPU_AUDIT',summary,flush=True)


if __name__=='__main__':main()
