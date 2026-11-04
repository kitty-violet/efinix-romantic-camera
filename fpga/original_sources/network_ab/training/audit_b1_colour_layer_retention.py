"""Compare distinguishable target-region layers through actual B1 packages.

All models use the same revised region labels. Lab76 thresholds describe
colour separation only, not semantic boundaries or a visual acceptance score.
"""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .audit_b1_target_chain import region_means,adjacency,lab
from .probe_b1_decoder_coverage import fragment
from .build_b1_ordered_palette import GROUPS
from .runtime import file_hash,write_json


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',nargs='+',required=True)
    ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();torch.set_num_threads(2);rows=m.load_data(args.data)['validation'];results=[]
    for checkpoint in args.checkpoints:
        s=torch.load(checkpoint,map_location='cpu',weights_only=True)
        net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(s['model']);net.quantized=True
        net.package['residual_encoding']=copy.deepcopy(s['residual_encoding']);ternary=bool(s.get('ternary_region_decoder'))
        levels=torch.cartesian_prod(*[torch.arange(n) for n in ((15,9,9) if ternary else (8,5,5))])
        z=torch.zeros(len(levels),16,1,1)
        for c,(a,b) in enumerate(GROUPS):
            if ternary:z[:,a:b,0,0]=(levels[:,c,None]-2*torch.arange(b-a)[None]).clamp(0,2)
            else:z[:,a:b,0,0]=(levels[:,c,None]>torch.arange(b-a)[None]).float()*127
        palette=fragment(net,z,'up2_low_dw')['rgb_head'][:,:,0,0].numpy()+128
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu');records=[]
        for row in rows:
            target=m.t.r.read(row['target']);labels=m.load_region(row['region']).astype(np.int64)
            means,count=region_means(target,labels);a,b,length=adjacency(labels)
            tt=lab(means);strength=np.linalg.norm(tt[a]-tt[b],axis=1);strong=strength>=5
            idx=((means[:,None]-palette[None])**2).sum(2).argmin(1)
            lost=(idx[a]==idx[b])&strong
            source=m.t.r.read(row['path']);x=torch.from_numpy(source.transpose(2,0,1).copy())[None].float()-128
            output=(actual(x)[0].permute(1,2,0).numpy()+128)
            predicted,_=region_means(output,labels);pp=lab(predicted)
            wanted=tt[a]-tt[b];got=pp[a]-pp[b]
            gap=np.linalg.norm(got,axis=1);collapsed=(gap<2)&strong
            reversed=((wanted*got).sum(1)<0)&strong
            den=max(1,length[strong].sum())
            records.append(dict(id=row['id'],target_regions=len(count),strong_pairs=int(strong.sum()),
                palette_lost_boundary_fraction=float(length[lost].sum()/den),
                output_collapsed_boundary_fraction=float(length[collapsed].sum()/den),
                output_reversed_boundary_fraction=float(length[reversed].sum()/den),
                palette_rgb_mae=float((np.abs(palette[idx]-means).mean(1)*count).sum()/count.sum())))
        fields=['palette_lost_boundary_fraction','output_collapsed_boundary_fraction','output_reversed_boundary_fraction','palette_rgb_mae']
        mean={k:float(np.mean([r[k] for r in records])) for k in fields}
        results.append(dict(checkpoint=checkpoint,sha256=file_hash(checkpoint),constant_colours=len(np.unique(palette,axis=0)),records=records,mean=mean))
        print('COLOUR_LAYER_RETENTION',Path(checkpoint).parent.name,mean,flush=True)
    write_json(args.out,dict(results=results,actual_integer_bytes=True,
        thresholds='Target adjacent Lab76>=5; output regional mean contrast<2 defines collapsed diagnostic',
        scope='Same six previously seen validation images and labels. Not semantics or visual acceptance.'))


if __name__=='__main__':main()
