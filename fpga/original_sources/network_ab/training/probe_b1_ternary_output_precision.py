"""Use the existing final activation range for RGB edge coverage.

The former ternary decoder reduced the final coordinates to28/16/16 before
RGB mixing, though127 is available. Recode constant counts to112/96/96 and
divide the existing RGB coefficients accordingly. Constant colours are tested
exhaustively; no new levels are claimed inside flat regions and no new ops.
"""
import argparse,copy,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .probe_b1_decoder_coverage import fragment
from .probe_b1_ternary_field_targets import canonical_states
from .build_b1_ordered_palette import GROUPS
from .train_b1_joint_region_colour import joint_region_loss
from .runtime import write_json,file_hash


@torch.no_grad()
def install(net):
    net.weight['up2_low_pw'].zero_();net.bias['up2_low_pw'].zero_();net.up2_low_pw_m.fill_(65536)
    for c,((a,b),step) in enumerate(zip(GROUPS,(8,12,12))):net.weight['up2_low_pw'][c,a:b,0,0]=step
    kernel=net.weight['up2_high_dw'].new_tensor([[1,2,1],[2,8,2],[1,2,1]])
    mm=round(131072/float(kernel.sum()))
    net.weight['up2_high_dw'].zero_();net.bias['up2_high_dw'].zero_()
    net.weight['up2_high_dw'][:3,0]=kernel*mm/131072;net.up2_high_dw_m[:,:3]=mm
    net.weight['rgb_head'][:,:3]/=net.weight['rgb_head'].new_tensor([4,6,6])[None,:,None,None]
    # Original head entries2/.75/.5 are exact at the inherited1/4 quantum.
    s=(net.rgb_head_m/131072).float().flatten()
    assert torch.allclose(net.weight['rgb_head']/s[:,None,None,None],(net.weight['rgb_head']/s[:,None,None,None]).round())


@torch.inference_mode()
def geometry_diagnostic(net):
    pairs=[(0,1214),(81*4+9*3+3,81*10+9*6+6),(81*6+9*4+4,81*7+9*4+4),
        (81*5+9*2+7,81*9+9*7+2),(81*8+9*4+3,81*12+9*5+5)]
    records=[]
    yy,xx=torch.meshgrid(torch.arange(48),torch.arange(64),indexing='ij');mask=xx-0.6*yy>16.3
    for a,b in pairs:
        ids=torch.where(mask,b,a)[None];states=F.pad(canonical_states(ids),(0,0,0,0,0,1))
        rgb=fragment(net,states,'up2_low_dw')['rgb_head'][0].permute(1,2,0)
        endpoints=fragment(net,F.pad(canonical_states(torch.tensor([a,b])[:,None,None]),(0,0,0,0,0,1)),'up2_low_dw')['rgb_head'][:,:,0,0]
        low,high=endpoints;delta=high-low
        alpha=((rgb-low)*delta).sum(-1)/delta.square().sum().clamp_min(1)
        line=low+alpha.clamp(0,1)[...,None]*delta
        edge=(alpha>0)&(alpha<1);off=(rgb-line).square().sum(-1).sqrt()
        records.append(dict(a=a,b=b,off_segment_mean=float(off[edge].mean()),off_segment_max=float(off.max())))
    return records


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args();torch.set_num_threads(2)
    dest=Path(args.out);dest.mkdir(exist_ok=False);state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=copy.deepcopy(state['residual_encoding']);initial=copy.deepcopy(net.state_dict())
    z=F.pad(canonical_states(torch.arange(1215)[:,None,None]),(0,0,0,0,0,1))
    baseline=fragment(net,z,'up2_low_dw')['rgb_head'].clone()
    rows=m.load_data(args.data)['validation'];images={};records=[]
    for name in ('current','range127'):
        net.load_state_dict(initial)
        if name=='range127':install(net)
        maximum=float((fragment(net,z,'up2_low_dw')['rgb_head']-baseline).abs().max());assert maximum==0
        geometry=geometry_diagnostic(net);sub=dest/name;sub.mkdir();raw=net.export();assert len(raw)==20480
        (sub/'coefficients-padded.bin').write_bytes(raw);actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');tests=[]
        for row in rows:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            ours=net(x,all_nodes=True);ref,_=actual(x*128,True)
            err=max(float((v-ref[k]).abs().max()) for k,v in ours.items());assert err==0
            t=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()/128-1
            labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
            p=ref['rgb_head']/128;loss,terms=joint_region_loss(p,t,labels)
            im=m.t.r.c.rgb_image(p[0]);images[name,row['id']]=im;im.save(sub/(row['id']+'.png'))
            tests.append(dict(id=row['id'],all_node_error=err,score=float(loss),**{k:float(v) for k,v in terms.items()}))
        saved={k:v for k,v in state.items() if k not in ('optimizer','rng','torch_rng','cuda_rng','bounded_training_model')}
        saved.update(model=copy.deepcopy(net.state_dict()),final_coordinate_range=[112,96,96] if name=='range127' else [28,16,16],
            parent_checkpoint=args.checkpoint,parent_sha256=file_hash(args.checkpoint),added_inference_operators=0,
            manually_calibrated_final_precision=name=='range127',visual_acceptance=False)
        torch.save(saved,sub/'initial.pt')
        mean={k:sum(r[k] for r in tests)/len(tests) for k in tests[0] if k!='id'}
        records.append(dict(name=name,constant_error=maximum,geometry=geometry,records=tests,mean=mean,sha256=hashlib.sha256(raw).hexdigest()))
        print('TERNARY_OUTPUT_PRECISION',name,geometry,mean,flush=True)
    by_id={r['id']:r for r in rows};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');d=ImageDraw.Draw(board)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(name,im) in enumerate([('target',Image.open(by_id[key]['target'])),('current',images['current',key]),('range127',images['range127',key])]):
            board.paste(im.crop(box),(420*j,i*314+34));d.text((420*j+4,i*314+5),label+' | '+name,font=font,fill='white')
    board.save(dest/'comparison.png');write_json(dest/'review.json',dict(records=records,all_constant_colours_unchanged=True,
        reference='Use existing activation range; controlled reuse of coverage precision in the ternary decoder.',visual_acceptance=False))


if __name__=='__main__':main()
