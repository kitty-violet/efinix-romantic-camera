"""Read-only CPU inspection of active channels in a completed checkpoint."""
import argparse,json
from collections import defaultdict
import torch
from .runtime import configure_workspace,contract_guard,write_json,file_hash
from .train_b1_cartoon_budget import RUN
from .train_b1_distillation import read_image,as_tensor
from network_ab.contract_torch import build_fp32_model


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--step',type=int,default=250);a=ap.parse_args()
    torch.set_num_threads(2);root=configure_workspace();run=root/'runs'/RUN;_,plan=contract_guard()
    path=run/f'train/checkpoint_{a.step:06d}.pt';s=torch.load(path,map_location='cpu',weights_only=True)
    net=build_fp32_model().eval();net.load_state_dict(s['model'])
    manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'));groups=defaultdict(list)
    for r in manifest['pairs']:groups[r['source']].append(r)
    features=defaultdict(list)
    with torch.inference_mode():
        for rows in groups.values():
            row=rows[len(rows)//2];im=read_image(row['current'])
            for x,y in [(200,160),(680,320)]:
                values=net(as_tensor(im[y:y+256,x:x+256])[None],return_tensors=True)
                for key,v in values.items():features[key].append(v[0].reshape(v.shape[1],-1)[:,::16].clone())
    records=[]
    for row in plan['nodes']:
        key=row['id'];v=torch.cat(features[key],1);mean=v.mean(1);std=v.std(1);mx=v.abs().max(1).values
        sv=torch.linalg.svdvals(v-mean[:,None]);energy=sv.square();total=energy.sum();rank=int((energy.cumsum(0)<total*.99).sum()+1) if total>1e-12 else 0
        records.append(dict(id=key,channels=len(mean),maximum=mx.tolist(),std=std.tolist(),zero_channels=(mx<1e-7).nonzero().flatten().tolist(),
            nearly_constant_channels=(std<1e-5).nonzero().flatten().tolist(),centered_rank99=rank))
    out=run/f'feature_health_{a.step:06d}.json';write_json(out,dict(checkpoint_sha256=file_hash(path),source_count=len(groups),train_crops=14,records=records,
        scope='Read-only activity and covariance inspection on14 training crops. No causal or global capacity claim.'))
    print(json.dumps([dict(id=r['id'],channels=r['channels'],zero=r['zero_channels'],rank99=r['centered_rank99']) for r in records]),flush=True)


if __name__=='__main__':main()
