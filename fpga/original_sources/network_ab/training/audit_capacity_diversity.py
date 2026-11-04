"""Check whether function-preserving widening actually used distinct channels."""
import json
from pathlib import Path
import numpy as np,torch
from .b1_capacity_component_net import CapacityComponentNet
from .train_b1_distillation import read_image,as_tensor
from .train_luma_regions import ROOT,RUN
from .runtime import file_hash,write_json


@torch.no_grad()
def main():
    torch.set_num_threads(2);base=ROOT/'runs'/RUN/'b1-capacity-control-v2';out=base/'capacity-diversity.json';assert not out.exists()
    data=json.loads((base/'capacity/data.json').read_text(encoding='utf-8'));nets={};records=[]
    for tag,path in [('initial',base/'capacity/initial.pt'),('trained',base/'capacity/fp32_checkpoint_002000.pt')]:
        net=CapacityComponentNet().eval();net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model']);nets[tag]=net
    names=['down.pw','context.down.pw','wide.down.pw','res1_conv1.pw','res1_conv2.pw','decode.pw'];sources={}
    for i,r in enumerate(data['pairs']):sources.setdefault(r['source'],i)
    for i in list(sources.values()):
        r=data['pairs'][i];arr=read_image(r['current']);x=as_tensor(arr[200:520,480:800])[None];values={k:n(x,True) for k,n in nets.items()}
        for name in names:
            for tag in nets:
                a=values[tag][name][0].flatten(1).double();half=len(a)//2;gap=(a[:half]-a[half:]).square().mean().sqrt();rms=a.square().mean().sqrt().clamp_min(1e-9)
                centered=a-a.mean(1,keepdim=True);eigen=torch.linalg.eigvalsh(centered@centered.T/a.shape[1]).clamp_min(0);prob=eigen/eigen.sum().clamp_min(1e-20)
                effective=float(torch.exp(-(prob*prob.clamp_min(1e-20).log()).sum()))
                records.append(dict(pair_index=i,node=name,stage=tag,duplicate_relative_rms=float(gap/rms),effective_rank=effective,channels=len(a)))
        print('CAPACITY_DIVERSITY',i,flush=True)
    write_json(out,dict(records=records,checkpoint_sha256=file_hash(base/'capacity/fp32_checkpoint_002000.pt'),script_sha256=file_hash(__file__),training_only_crops=True,diagnostic_not_proof_of_capacity_adequacy=True))


if __name__=='__main__':main()
