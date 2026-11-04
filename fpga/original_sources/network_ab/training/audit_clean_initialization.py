"""Training-source feature diversity before independent initialization training."""
import json
import torch
from .b1_capacity_component_net import CapacityComponentNet
from .clean_skip_capacity_net import CleanSkipNet
from .train_luma_regions import ROOT,RUN
from .train_b1_distillation import read_image,as_tensor
from .runtime import file_hash,write_json


@torch.no_grad()
def main():
    torch.set_num_threads(2);base=ROOT/'runs'/RUN;out=base/'teacher-clean-skip-fresh-v1/initial_diversity.json';assert not out.exists()
    paths={'copied_widening':base/'b1-capacity-control-v2/capacity/initial.pt','independent':base/'teacher-clean-skip-fresh-v1/fresh_initial.pt'}
    nets={};data=json.loads((base/'teacher-clean-skip-fresh-v1/data.json').read_text(encoding='utf-8'));rows=[];sources={}
    for i,r in enumerate(data['pairs']):sources.setdefault(r['source'],i)
    for tag,p in paths.items():
        n=CapacityComponentNet() if tag=='copied_widening' else CleanSkipNet();n.load_state_dict(torch.load(p,map_location='cpu',weights_only=True)['model']);nets[tag]=n.eval()
    for i in sources.values():
        x=as_tensor(read_image(data['pairs'][i]['current'])[200:520,480:800])[None]
        for tag,n in nets.items():
            values=n(x,True)
            for key in ['down.pw','context.down.pw','wide.down.pw','res1_conv1.pw']:
                v=values[key][0].flatten(1).double();z=v-v.mean(1,keepdim=True);eig=torch.linalg.eigvalsh(z@z.T/z.shape[1]).clamp_min(0);p=eig/eig.sum().clamp_min(1e-20)
                rank=float(torch.exp(-(p*p.clamp_min(1e-20).log()).sum()));gap=float((v[:16]-v[16:]).square().mean().sqrt()/v.square().mean().sqrt().clamp_min(1e-9))
                rows.append(dict(source_index=i,initialization=tag,node=key,effective_rank=rank,copy_gap_relative_rms=gap,zero_fraction=float((v==0).double().mean())))
    summary={tag:{key:sum(r['effective_rank'] for r in rows if r['initialization']==tag and r['node']==key)/len(sources) for key in ['down.pw','context.down.pw','wide.down.pw','res1_conv1.pw']} for tag in nets}
    write_json(out,dict(records=rows,summary=summary,weights={k:file_hash(v) for k,v in paths.items()},not_quality_evidence=True,script_sha256=file_hash(__file__)));print('INITIAL_DIVERSITY',summary,flush=True)


if __name__=='__main__':main()
