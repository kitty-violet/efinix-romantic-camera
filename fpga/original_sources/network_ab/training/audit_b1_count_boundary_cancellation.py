"""Check counter-moving fields hidden by a correct total colour count.

Same-colour field agreement does not test adjacent different colours. The
minimum L1 field change between totals a,b is abs(a-b). Any excess indicates
opposing field changes, which may interact with spatial threshold filtering.
This diagnostic does not itself prove those changes cause visible artifacts.
"""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .build_b1_ordered_palette import GROUPS
from .runtime import write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();torch.set_num_threads(2);s=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(s['model']);net.quantized=True;net.package['residual_encoding']=copy.deepcopy(s['residual_encoding'])
    rows=[];seen=set()
    for row in m.load_data(args.data)['training']:
        if row['video'] not in seen:seen.add(row['video']);rows.append(row)
    records=[]
    for row in rows:
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
        fields=net(x,all_nodes=True)['up2_low_dw'][0,:15];values=[]
        for group,(a,b) in enumerate(GROUPS):
            f=fields[a:b];total=f.sum(0);canonical=(total[None]-2*torch.arange(b-a)[:,None,None]).clamp(0,2)
            for offset in (1,2,4):
                for dim in (-1,-2):
                    delta=f.narrow(dim,offset,f.shape[dim]-offset)-f.narrow(dim,0,f.shape[dim]-offset)
                    minimal=delta.sum(0).abs();path=delta.abs().sum(0);active=minimal>0
                    excess=path-minimal;assert excess.min()>=0
                    values.append(dict(group=group,offset=offset,dim=dim,transitions=int(active.sum()),
                        counter_moving_fraction=float((excess[active]>0).float().mean()) if active.any() else 0.,
                        excess_field_change=float(excess[active].mean()) if active.any() else 0.,
                        same_total_hidden_change_fraction=float(((minimal==0)&(path>0)).float().mean()),
                        canonical_equal_fraction=float((canonical==f).all(0).float().mean())))
        records.append(dict(id=row['id'],records=values))
    ss=[v for r in records for v in r['records']];den=sum(v['transitions'] for v in ss)
    mean={k:sum(v[k]*v['transitions'] for v in ss)/max(1,den) for k in ('counter_moving_fraction','excess_field_change','same_total_hidden_change_fraction','canonical_equal_fraction')}
    write_json(args.out,dict(checkpoint_sha256=file_hash(args.checkpoint),records=records,mean=mean,scope='Seven TRAIN sources; diagnostic, no updates or causal claim.'))
    print('COUNT_BOUNDARY_CANCELLATION',mean,flush=True)


if __name__=='__main__':main()
