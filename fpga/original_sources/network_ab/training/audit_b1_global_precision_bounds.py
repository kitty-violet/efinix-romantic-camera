"""Conservative interval bounds for safe bias absorption; no fitting/updates.

Per-channel intervals over all legal RGB inputs ignore correlations and can be
loose. Positive narrow preactivation bounds can justify recentering a channel;
sampled means alone cannot. This audit does not modify the checkpoint.
"""
import argparse,copy
import torch
from . import train_b1_minimum_area as m
from .runtime import write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();torch.set_num_threads(2);s=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(s['model']);net.package['residual_encoding']=copy.deepcopy(s['residual_encoding'])
    bounds={'input':(torch.full((3,),-128.,dtype=torch.float64),torch.full((3,),127.,dtype=torch.float64))};records=[]
    for r in net.plan['nodes']:
        name=r['id'];key=name.replace('.','__');low,high=bounds[r['inputs'][0]]
        if key in net.weight:
            w=net.weight[key].double();b=net.bias[key].double();positive=w.clamp_min(0);negative=w.clamp_max(0)
            if r['kind']=='dw3x3':
                lo=b+(positive.flatten(1).sum(1)*low+negative.flatten(1).sum(1)*high)
                hi=b+(positive.flatten(1).sum(1)*high+negative.flatten(1).sum(1)*low)
            else:
                pos=positive.sum((-1,-2));neg=negative.sum((-1,-2));lo=b+pos@low+neg@high;hi=b+pos@high+neg@low
            eligible=(lo>2)&(hi<125)&((hi-lo)<64)
            records.append(dict(node=name,channels=len(lo),globally_positive_narrow=int(eligible.sum()),
                candidate_channels=torch.where(eligible)[0].tolist(),pre_lower=lo.tolist(),pre_upper=hi.tolist()))
            lower=0 if r['activation']=='relu' else -128
            bounds[name]=(lo.clamp(lower,127),hi.clamp(lower,127))
        elif r['kind']=='nearest_x2':bounds[name]=(low,high)
        else:
            lo2,hi2=bounds[r['inputs'][1]];q=net.package['residual_encoding'][name]
            if not q['main_scaled']:low,lo2=lo2,low;high,hi2=hi2,high
            factor=q['multiplier']/131072;post=q['post_multiplier']/131072
            bounds[name]=(((low*factor).clamp(-128,127)+lo2).mul(post).clamp(-128,127),
                          ((high*factor).clamp(-128,127)+hi2).mul(post).clamp(-128,127))
    eligible=[r for r in records if r['globally_positive_narrow']]
    write_json(a.out,dict(checkpoint_sha256=file_hash(a.checkpoint),scope='Conservative FP32 clipped-graph interval bounds over all legal RGB; not an integer-error proof.',
        parameter_updates=0,records=records,eligible_channels=sum(r['globally_positive_narrow'] for r in records)))
    print('GLOBAL_PRECISION_BOUNDS',[(r['node'],r['globally_positive_narrow']) for r in eligible],flush=True)


if __name__=='__main__':main()
