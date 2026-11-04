"""Check target colour coverage and within-region field consistency on TRAIN.

Only locations already predicting the correct total colour are used for the
field-assignment ambiguity statistic. This separates colour errors from code
inconsistency. No image correction is applied.
"""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .train_b1_ternary_regions import region_codes
from .build_b1_ordered_palette import GROUPS
from .probe_b1_decoder_coverage import fragment
from .runtime import write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();torch.set_num_threads(2)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model']);net.quantized=True;net.package['residual_encoding']=copy.deepcopy(state['residual_encoding'])
    levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9));z=torch.zeros(len(levels),16,1,1)
    for c,(a,b) in enumerate(GROUPS):z[:,a:b,0,0]=(levels[:,c,None]-2*torch.arange(b-a)[None]).clamp(0,2)
    pal=fragment(net,z,'up2_low_dw')['rgb_head'][:,:,0,0].numpy()+128;pk=tuple(map(tuple,pal.tolist()))
    rows=[];seen=set()
    for row in m.load_data(args.data)['training']:
        if row['video'] not in seen:seen.add(row['video']);rows.append(row)
    records=[]
    for row in rows:
        rgb=m.t.r.read(row['path']);x=torch.from_numpy(rgb.transpose(2,0,1).copy())[None].float()/128-1
        nodes=net(x,all_nodes=True);f=nodes['up2_low_dw'][0,:15].numpy().astype(np.int64)
        counts=[f[a:b].sum(0) for a,b in GROUPS];code=counts[0]*81+counts[1]*9+counts[2]
        truth=region_codes(row['target'],row['region'],pk).astype(np.int64)
        lab=m.load_region(row['region']).astype(np.int64);ll=lab[::2,::2]
        pure=(ll==lab[1::2,::2])&(ll==lab[::2,1::2])&(ll==lab[1::2,1::2])
        correct=(code==truth[::2,::2])&pure;rid=ll[correct];n=int(lab.max())+1
        count=np.bincount(rid,minlength=n);valid=count>=16;ambiguity=[]
        for plane in f:
            hist=np.bincount(rid*3+plane[correct],minlength=n*3).reshape(n,3)
            pp=hist/count.clip(1)[:,None]
            ambiguity.append(float(((1-(pp*pp).sum(1))*count*valid).sum()/max(1,(count*valid).sum())))
        target=m.t.r.read(row['target']).astype(np.float32)
        q=pal[truth];covered=((q-target)**2).mean()**.5
        records.append(dict(id=row['id'],correct_count_fraction=float(correct.sum()/max(1,pure.sum())),
            eligible_correct_regions=int(valid.sum()),mean_field_disagreement=float(np.mean(ambiguity)),target_palette_rmse=float(covered),
            label_dtype='uint16',max_target_index=int(truth.max())))
    write_json(args.out,dict(checkpoint=args.checkpoint,sha256=file_hash(args.checkpoint),records=records,
        scope='Seven TRAIN sources; same-correct-colour field ambiguity is diagnostic, not proof of visual seams'))
    print(json.dumps(records,indent=2),flush=True)


if __name__=='__main__':main()
