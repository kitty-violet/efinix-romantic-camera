"""Keep scaffold roles distinct: coarse color + localized teacher boundaries.

Same inference graph, but colors cannot be dense teacher pixels in the oracle
diagnostic. Boundary confidence is trained with separately normalized positive
and negative classification, overlap, and a penalty outside the target band.
"""
import copy,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from . import train_contour_regions as base
from .contour_region_net import scaffold_target as continuous_target,plan,budget
from .runtime import file_hash,write_json
from .train_luma_regions import ROOT,RUN

original_guide_loss=base.guide_losses


def scaffold_target(rgb):
    target=continuous_target(rgb)
    # 8 native pixels -> one color sample; this conditioning grid is produced
    # by the encoder at inference, not by filtering the final RGB output.
    color=F.avg_pool2d((rgb+1)*.5,8).repeat_interleave(4,-2).repeat_interleave(4,-1)
    edge=(target[:,7:]>.12).float()
    return torch.cat([color,target[:,3:7],edge],1)


def guide_losses(p,t,margin):
    values=original_guide_loss(p,t,margin);m=margin//2
    prediction=p[:,7:,m:-m,m:-m];label=t[:,7:,m:-m,m:-m]
    logits=6*(prediction-.5);error=F.binary_cross_entropy_with_logits(logits,label,reduction='none')
    values['contour_class']=(error*label).sum()/label.sum().clamp_min(1)+(error*(1-label)).sum()/(1-label).sum().clamp_min(1)
    probability=logits.sigmoid();values['contour_overlap']=1-(2*(probability*label).sum()+1)/(probability.sum()+label.sum()+1)
    outside=1-F.max_pool2d(label,3,1,1);values['contour_outside']=(F.relu(prediction-.05)*outside).sum()/outside.sum().clamp_min(1)
    return values


def prepare(out,steps):
    source=ROOT/'runs'/RUN/'contour-region-v1/fp32_checkpoint_002000.pt';s=torch.load(source,map_location='cpu',weights_only=True)
    assert all(file_hash(k)==v for k,v in s['binding']['code'].items());out.mkdir(exist_ok=False)
    initial=out/'initial.pt';torch.save(dict(model=s['model'],source=str(source),source_sha256=file_hash(source)),initial)
    (out/'data.json').write_bytes((source.parent/'data.json').read_bytes());cfg=copy.deepcopy(s['config'])
    cfg.update(seed=2026100472,fp32_steps=steps,fp32_lr=.0007,initial=str(initial),initial_sha256=file_hash(initial),
        scaffold_schema='coarse-color8-and-thin-boundary-v2',scaffold_pretrain_steps=300,
        guide_weights=dict(guide_color=8,guide_boundary=2,guide_interior=2,contour_class=.6,contour_overlap=.6,contour_outside=2),
        method='Coarse region color bottleneck with separately supervised boundary confidence, orientation, overlap and outside-band errors')
    write_json(out/'config.json',cfg);write_json(out/'graph.json',plan());binding=copy.deepcopy(s['binding'])
    binding['code'][str(Path(__file__))]=file_hash(__file__);binding.update(config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json'));write_json(out/'binding.json',binding)
    # Sharp labels must score better than spread-out boundaries, not just have
    # higher contrast. A shifted or extra edge also receives explicit loss.
    t=torch.zeros(1,8,48,48);t[:,:3]=.5;t[:,7,:,24]=1
    broad=t.clone();broad[:,7:]=F.max_pool2d(t[:,7:],5,1,2)
    shifted=t.roll(4,-1);exact=guide_losses(t,t,8);wider=guide_losses(broad,t,8);offset=guide_losses(shifted,t,8)
    assert wider['contour_class']>exact['contour_class'] and offset['contour_class']>exact['contour_class']
    assert wider['contour_outside']>exact['contour_outside']
    write_json(out/'preparation.json',dict(budget=budget(),source_sha256=file_hash(source),inference_graph_unchanged=True,
        label_change='region colors at 90p broadcast to360p; direction contrast at360p; binary boundary confidence at360p',
        no_inference_pooling_added=True,classification_checks={name:{k:float(v) for k,v in loss.items()} for name,loss in [('exact',exact),('wide',wider),('shifted',offset)]}))
    print('SHARP_CONTOUR_PREPARED',flush=True)


def main():
    base.scaffold_target=scaffold_target;base.guide_losses=guide_losses;base.prepare=prepare
    if '--name' not in sys.argv:sys.argv.extend(['--name','sharp-contour-region-v2'])
    base.main()


if __name__=='__main__':main()
