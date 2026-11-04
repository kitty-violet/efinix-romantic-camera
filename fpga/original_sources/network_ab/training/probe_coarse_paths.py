"""Measure whether regional paths contribute; no optimizer or validation fit."""
import json
from pathlib import Path
import torch,numpy as np
from PIL import Image
from .coarse_skip_net import CoarseSkipNet
from .train_luma_regions import ROOT,RUN,display
from .train_b1_distillation import as_tensor,read_image,to_rgb
from .runtime import file_hash,write_json


@torch.no_grad()
def main():
    torch.set_num_threads(2);root=ROOT/'runs'/RUN;out=root/'coarse-path-ablation-v1';out.mkdir(exist_ok=False)
    source=root/'ordinal-regions-fp32/folded_002000.pt';state=torch.load(source,map_location='cpu',weights_only=True)
    net=CoarseSkipNet().eval();net.load_state_dict(state['model']);data=json.loads((source.parent/'data.json').read_text(encoding='utf-8'))
    samples=[('train',data['pairs'][40]['current']),('val_stairs_090',next(r['source'] for r in data['validation'] if r['id']=='val_stairs_090'))];records=[]
    for name,path in samples:
        x=as_tensor(read_image(path))[None];nodes=net(x,True);reference=display(nodes['rgb']);variants=[]
        activations={k:dict(zero_fraction=float((v==0).float().mean()),rms=float(v.square().mean().sqrt()),max=float(v.max()),min=float(v.min())) for k,v in nodes.items()}
        for branch in ['native.pw','context.up.pw','wide.up.pw']:
            h=net.layers[branch.replace('.','__')].register_forward_hook(lambda m,a,v:torch.zeros_like(v))
            y=display(net(x));h.remove();Image.fromarray(to_rgb(y[0])).save(out/f'{name}_without_{branch}.png')
            variants.append(dict(branch=branch,mae_rgb255=float((y-reference).abs().mean()*128),pixel_change_fraction=float((y!=reference).any(1).float().mean())))
        records.append(dict(id=name,source=path,activations=activations,ablation=variants));print(name,variants,flush=True)
    write_json(out/'report.json',dict(records=records,checkpoint=str(source),checkpoint_sha256=file_hash(source),script_sha256=file_hash(__file__)))


if __name__=='__main__':main()
