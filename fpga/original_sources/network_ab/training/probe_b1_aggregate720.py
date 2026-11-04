"""Sum continuous colour fields before the final720 count quantization.

Controlled comparison with the paired-count decoder. No new inference layer:
existing PW aggregates Y/Co/Cg, existing720 DW rounds each total once. A
hand-constructed decoder candidate, not a claim of visual acceptance.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .probe_b1_ternary_field_targets import canonical_states
from .audit_b1_step_ringing import test_inputs,metrics
from .audit_b1_white_rims_grid import measures
from .train_b1_joint_region_colour import joint_region_loss
from .train_b1_boundary_profile import boundary_profile
from . import train_b1_minimum_area as m
from .runtime import write_json,file_hash

GROUPS=((0,7),(7,11),(11,15))


@torch.no_grad()
def install(net,setup):
    matrix=net.weight['rgb_head'].new_tensor(setup['matrix'])
    for key in ('up2_low_pw','up2_high_dw','rgb_head'):
        net.weight[key].zero_();net.bias[key].zero_()
    net.up2_low_pw_m.fill_(1);net.up2_high_dw_m.fill_(1);net.rgb_head_m.fill_(65536)
    kernel=matrix.new_tensor([[1,2,1],[2,4,2],[1,2,1]])
    for c,(lo,hi) in enumerate(GROUPS):
        n=hi-lo;mult=round(131072/(n*64))
        net.up2_low_pw_m[:,c]=mult
        net.weight['up2_low_pw'][c,lo:hi,0,0]=64*mult/131072
        mult=round(131072*(2*n)/(121*16));net.up2_high_dw_m[:,c]=mult
        net.weight['up2_high_dw'][c,0]=kernel*mult/131072
        net.weight['rgb_head'][:,c,0,0]=matrix[:,c]
    net.bias['rgb_head'].copy_(matrix.new_tensor(setup['bias']))
    return dict(matrix=setup['matrix'],bias=setup['bias'],maxima=[14,8,8],
                final_quantization='one_round_per_aggregate_at720',added_inference_ops=0)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--device',default='cpu');a=ap.parse_args()
    torch.set_num_threads(2);dest=Path(a.out);dest.mkdir(exist_ok=False)
    state,base=load(a.checkpoint);base=base.to(a.device);net=copy.deepcopy(base);setup=install(net,state['packed720'])
    levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float().to(a.device)
    pal=(levels@levels.new_tensor(setup['matrix']).T+levels.new_tensor(setup['bias'])+128).clamp(0,255)
    codes=torch.arange(1215,device=a.device)[:,None,None]
    fields=F.pad(net.round(canonical_states(codes)*63.5),(0,0,0,0,0,1))
    got=fragment(net,fields,'up1_high_dw')['rgb_head'][:,:,0,0]+128
    assert torch.equal(got,pal),float((got-pal).abs().max())
    gen=torch.Generator().manual_seed(418)
    z=torch.randint(0,3,(4096,16,1,1),generator=gen).float().to(a.device);z[:,15]=0
    indices=(81*z[:,:7].sum(1)+9*z[:,7:11].sum(1)+z[:,11:15].sum(1)).long()[:,0,0]
    got=fragment(net,net.round(z*63.5),'up1_high_dw')['rgb_head'][:,:,0,0]+128
    assert torch.equal(got,pal[indices])
    raw=net.export();assert len(raw)==20480;(dest/'initial.bin').write_bytes(raw)
    saved=copy.deepcopy(state);saved.update(model={k:v.cpu() for k,v in net.state_dict().items()},
        aggregate720=setup,ternary_packed720=False,visual_acceptance=False,step=0)
    torch.save(saved,dest/'initial.pt')
    actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device=a.device)
    source,tests=test_inputs();source=source[::3].to(a.device);tests=tests[::3]
    synthetic={}
    for name,n in (('paired',base),('aggregate',net)):
        rows=[]
        for i,src in enumerate(source):
            p=n(src[None]/128-1,all_nodes=True)
            if name=='aggregate':
                ref,_=actual(src[None]-128,True)
                assert all(torch.equal(v,ref[k]) for k,v in p.items())
            ends=torch.stack((src[:,0,0],src[:,0,-1]))[:,:,None,None].expand(2,3,48,48)
            endpoint=n(ends/128-1)[0][:,:,24,24]*128+128
            rows+=metrics(p['rgb_head'][...,16:-16,32:-32]+128,endpoint[:1],endpoint[1:])
        synthetic[name]=dict(records=rows,mean={k:float(np.mean([r[k] for r in rows])) for k in rows[0]})
    print('AGGREGATE720_SYNTHETIC', {k:v['mean'] for k,v in synthetic.items()},flush=True)
    images={};records=[];data=m.load_data(a.data)['validation']
    for row in data:
        def read(path):return torch.from_numpy(m.t.r.read(path).transpose(2,0,1).copy())[None].float().to(a.device)
        src,t=read(row['path']),read(row['target']);labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None].to(a.device)
        nodes=net(src/128-1,all_nodes=True);ref,_=actual(src-128,True)
        assert all(torch.equal(v,ref[k]) for k,v in nodes.items())
        record=dict(id=row['id'],variants={})
        for name,p in [('paired',base(src/128-1)[0]),('aggregate',nodes['rgb_head']/128)]:
            _,v=joint_region_loss(p,t/128-1,labels);_,prof=boundary_profile(p,t/128-1,labels)
            extra,_=measures(p.cpu()*128+128,t.cpu(),src.cpu())
            record['variants'][name]=dict(**{k:float(x) for k,x in (v|prof).items()},**extra)
            im=m.t.r.c.rgb_image(p[0]);images[row['id'],name]=im;im.save(dest/(row['id']+'_'+name+'.png'))
        records.append(record);print('AGGREGATE720_IMAGE',row['id'], {k:(v['region_flatness'],v['rgb_profile_shape'],v['bright_target_excess_fraction']) for k,v in record['variants'].items()},flush=True)
    byid={r['id']:r for r in data};board=Image.new('RGB',(1260,1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        choices=[('目标',Image.open(byid[rid]['target'])),('分组取整',images[rid,'paired']),('合并后取整',images[rid,'aggregate'])]
        for j,(name,im) in enumerate(choices):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+name,font=font,fill='white')
    board.save(dest/'comparison.png')
    write_json(dest/'review.json',dict(constant_palette_error=0,random_permutations=4096,byte_size=len(raw),
        all_six720_node_error=0,synthetic=synthetic,records=records,visual_acceptance=False,
        source_sha256=file_hash(a.checkpoint),new_operators=0))


if __name__=='__main__':main()
