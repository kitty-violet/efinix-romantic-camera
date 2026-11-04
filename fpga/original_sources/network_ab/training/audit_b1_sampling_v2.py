"""Freeze the user-designated fallback and measure B1's real sampling path."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont
from . import train_b1_learned_colour as colour
from .runtime import configure_workspace, contract_guard, file_hash, write_json

ROOT=colour.ROOT
PARENT=ROOT/'b1-rank-blocks-20261005-v1/luma-regions/stable-regions'
FALLBACK=ROOT/'b1-style-v1-fallback-20261005'
OUT=ROOT/'b1-sampling-colour-v2-20261005'


def freeze():
    FALLBACK.mkdir(exist_ok=False)
    files=['checkpoint_001800.pt','binding.json','preflight.json',
           'train_b1_learned_colour.py','train_b1_luma_regions.py','train_b1_stable_regions.py']
    files += ['step_001800/'+p for p in ['coefficients-padded.bin','coefficients.bin','load_words.hex',
              'verification.json','review.json','reviewed.png','region_targets.png']]
    files += ['videos/'+p for p in ['stairs_integer1080.mp4','plaza_integer1080.mp4','results.json']]
    records=[]
    for rel in files:
        src=PARENT/rel;dst=FALLBACK/rel;dst.parent.mkdir(exist_ok=True)
        shutil.copy2(src,dst);assert file_hash(src)==file_hash(dst)
        records.append(dict(path=rel,bytes=dst.stat().st_size,sha256=file_hash(dst)))
    write_json(FALLBACK/'manifest.json',dict(role='User-designated V1 last-resort fallback',
        source=str(PARENT),assessment={'continuity':'user accepts this version',
        'colours':'too few value levels','contours':'soft, corners insufficient',
        'people':'smeared','final_quality_accepted':False},files=records))
    (FALLBACK/'README.txt').write_text('第一版兜底，保留备用，不代表最终画质通过。\n'
        '用户本轮认可视频连续性；不认可色彩层次、轮廓、棱角与人物包浆。\n'
        '权重、训练断点、原训练源码及12秒/15秒实际整数包视频已复制并逐项SHA256核对。\n'
        '没有外部后处理；软件B1整数复演，不代表真实上板验证。\n',encoding='utf-8')


def profile_stats(a,scale,edge):
    """Project multichannel plateau contrast, preserving pixel-grid positions."""
    # a is C x W, centred row average. Projections quantify transition, not style.
    lo=a[:,:16].mean(1);hi=a[:,-16:].mean(1);direction=hi-lo
    power=float(np.dot(direction,direction))
    if power<1e-5:return dict(contrast=power,unresolved=True)
    q=((a-lo[:,None])*direction[:,None]).sum(0)/power
    positions=np.arange(q.size)*scale
    center=edge-.5
    crossings=[]
    for level in (.1,.5,.9):
        idx=np.where((q[:-1]<=level)&(q[1:]>=level))[0]
        if not len(idx):return dict(contrast=power,unresolved=True)
        j=min(idx,key=lambda j:abs(positions[j]-center))
        crossings.append(float(positions[j]+scale*(level-q[j])/(q[j+1]-q[j])))
    return dict(contrast=power,width_10_90=crossings[2]-crossings[0],
                centre_offset=crossings[1]-center,overshoot=float(max(0,q.max()-1,-q.min())))


@torch.inference_mode()
def audit(package=None,dest=None):
    dest=dest or OUT;package=package or PARENT/'step_001800/coefficients-padded.bin'
    dest.mkdir(exist_ok=True)
    assert not (dest/'sampling_audit.json').exists(), 'Refuse to replace an existing measurement'
    net=colour.ColourB1(revive=False,source=package).cuda().eval()
    net.quantized=True
    stages=['stem','down1.dw','down1.pw','down2.dw','down2.pw','res3_add',
            'up1_low_pw','up1_nn','up1_high_dw','up2_low_pw','up2_nn','up2_high_dw','rgb_head']
    records=[];panels=[]
    for axis in ('x','y'):
        for phase in range(4):
            x=torch.full((1,3,256,256),-64.,device='cuda');edge=128+phase
            if axis=='x':x[:,:,:,edge:]=64
            else:x[:,:,edge:,:]=64
            nodes=net(x/128,all_nodes=True)
            for name in stages:
                z=nodes[name][0]
                if axis=='y':z=z.transpose(-1,-2)
                h,w=z.shape[-2:];profile=z[:,h//2-2:h//2+2].mean(1).cpu().numpy()
                records.append(dict(axis=axis,phase=phase,node=name,**profile_stats(profile,256/w,edge)))
            if axis=='x':panels.append(colour.rgb_image(nodes['rgb_head'][0]/128).crop((96,96,160,160)).resize((256,256),Image.Resampling.NEAREST))
    kernels={}
    for key in ('down1__dw','down2__dw','up1_high_dw','up2_low_dw','up2_high_dw'):
        w=net.weight[key].detach().cpu().numpy()[:,0]
        kernels[key]=dict(coefficients=w.tolist(),negative_fraction=float((w<0).mean()),
                         sums=w.sum((1,2)).tolist(),absolute_sums=np.abs(w).sum((1,2)).tolist())
    # Oracle channel tests isolate sampling from learned region prediction.
    # The same layers/strides/padding/nearest operations are used, with specified kernels.
    g=torch.tensor([[1,2,1],[2,4,2],[1,2,1]],device='cuda',dtype=torch.float32)[None,None]/16
    cross=torch.tensor([[0,1,0],[1,4,1],[0,1,0]],device='cuda',dtype=torch.float32)[None,None]/8
    aligned=torch.tensor([[0,0,0],[0,.25,.25],[0,.25,.25]],device='cuda')[None,None]
    centre=torch.tensor([[0,0,0],[0,1,0],[0,0,0]],device='cuda',dtype=torch.float32)[None,None]
    conv=lambda x,k,s=1:F.conv2d(F.pad(x,(1,1,1,1),mode='replicate'),k,stride=s)
    oracle=[]
    for phase in range(4):
        x=torch.zeros((1,1,128,256),device='cuda');x[:,:,:,128+phase:]=1
        for name,d,u1,u2 in [('gaussian_sym',g,g,cross),('gaussian_aligned',g,aligned,aligned),
                             ('point_sym',centre,g,cross),('point_aligned',centre,aligned,aligned)]:
            z=conv(conv(x,d,2),d,2)
            z=conv(z.repeat_interleave(2,-1).repeat_interleave(2,-2),u1)
            z=conv(z.repeat_interleave(2,-1).repeat_interleave(2,-2),u2)
            oracle.append(dict(name=name,phase=phase,**profile_stats(z[0,:,64].cpu().numpy(),1,128+phase)))
    write_json(dest/'sampling_audit.json',dict(package=str(package),
        package_sha256=file_hash(package),
        contract_sha256=file_hash(Path(__file__).parents[1]/'model_b_contract.json'),
        actual_integer_node_profiles=records,actual_kernels=kernels,isolated_oracle_sampling=oracle,
        caveat='Synthetic step measurements diagnose these weights, not a proof of B1 capacity.'))
    img=Image.new('RGB',(1024,288),'#15202a');d=ImageDraw.Draw(img)
    for i,p in enumerate(panels):img.paste(p,(256*i,32));d.text((256*i+4,8),f'Actual INT8 step / phase {i}',fill='white')
    img.save(dest/'sampling_steps.png')
    for name in stages:
        rows=[r for r in records if r['node']==name and not r.get('unresolved')]
        if rows:print(name,'width',round(np.mean([r['width_10_90'] for r in rows]),3),'offset',round(np.mean([r['centre_offset'] for r in rows]),3),flush=True)
    for name in ('gaussian_sym','gaussian_aligned','point_sym','point_aligned'):
        rows=[r for r in oracle if r['name']==name]
        print(name,'width',np.mean([r['width_10_90'] for r in rows]),'offsets',[round(r['centre_offset'],3) for r in rows],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--audit-only',action='store_true')
    ap.add_argument('--package',type=Path);ap.add_argument('--out',type=Path);args=ap.parse_args()
    configure_workspace();contract_guard();torch.set_num_threads(4)
    if args.package:assert args.audit_only and args.out
    if not args.audit_only:freeze()
    else:assert (FALLBACK/'manifest.json').exists()
    audit(args.package,args.out);print('FALLBACK_FROZEN_AND_SAMPLING_AUDITED',flush=True)
