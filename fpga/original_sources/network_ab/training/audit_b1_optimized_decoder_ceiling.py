"""Target-only latent fitting to isolate fixed decoder geometry.

These images are NOT source predictions or deployable outputs. Optimize the
ordered30-field180p input of the existing fixed decoder on four target crops.
The real source encoder and all exported coefficients remain unchanged.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .train_b1_recover_boundary_gradients import recover_gradient
from .train_b1_ternary_regions import region_codes
from .build_b1_thirty_binary_regions import code_bits,GROUPS30
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

def decode(net,z):
    nodes={'up1_low_dw':net.round(z)};active=False
    for r in net.plan['nodes']:
        name=r['id'];key=name.replace('.','__')
        if name=='up1_low_dw':active=True;continue
        if not active:continue
        x=nodes[r['inputs'][0]]
        if key in net.weight:
            scale=(getattr(net,key+'_m')/131072).float().flatten()
            w=net.round(net.weight[key]/scale[:,None,None,None]).clamp(-127,127)
            b=net.round(net.bias[key]/scale).double()[None,:,None,None]
            if r['kernel']==3:x=F.pad(x,(1,1,1,1),mode='replicate')
            dot=F.conv2d(x,w,stride=r['stride'],groups=r['groups'])
            pre=net.round((dot.double()+b).clamp(-2**31,2**31-1)*getattr(net,key+'_m')/131072).float()
            y=pre.clamp(-128,127)
            if r['activation']=='relu':y=y.clamp_min(0)
            if name=='up2_low_dw' and torch.is_grad_enabled():y=recover_gradient(y,pre)
        else:
            assert r['kind']=='nearest_x2';y=x.repeat_interleave(2,-2).repeat_interleave(2,-1)
        nodes[name]=y
    return nodes['rgb_head']/128

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=500);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);net.requires_grad_(False);raw=net.export();dest=Path(a.out);dest.mkdir(exist_ok=False)
    setup=state['binary_coverage720'];levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float()
    pal=(levels@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255);pk=tuple(map(tuple,pal.tolist()))
    rows={r['id']:r for r in m.load_data(a.data)['validation']};zs=[];ts=[];source=[];names=[]
    for rid,box,label in m.t.r.c.CROPS:
        row=rows[rid];x0,y0,_,_=box;x0=x0//4*4;y0=y0//4*4;sl=np.s_[y0:y0+280,x0:x0+420]
        target=torch.from_numpy(m.t.r.read(row['target'])[sl].transpose(2,0,1).copy()).float()/128-1
        x=torch.from_numpy(m.t.r.read(row['path'])[sl].transpose(2,0,1).copy())[None].float()/128-1
        code=torch.from_numpy(region_codes(row['target'],row['region'],pk)[sl].astype('int64'))[None]
        zs.append(F.pad(F.avg_pool2d(code_bits(code)*127,4),(0,0,0,0,0,2))[0]);ts.append(target)
        with torch.no_grad():source.append(net(x)[0][0])
        names.append((rid,label))
    target=torch.stack(ts);z=torch.nn.Parameter(torch.stack(zs));opt=torch.optim.Adam([z],lr=.7)
    with torch.no_grad():
        initial=decode(net,z);assert torch.equal(initial,fragment(net,net.round(z),'up1_low_dw')['rgb_head']/128)
    history=[]
    for step in range(a.steps):
        p=decode(net,z);pixel=F.smooth_l1_loss(p,target,beta=.025)
        edge=sum(F.l1_loss(torch.diff(p,dim=d),torch.diff(target,dim=d)) for d in (-1,-2))/2
        loss=pixel+.5*edge;opt.zero_grad();loss.backward();assert torch.isfinite(z.grad).all();opt.step()
        with torch.no_grad():
            z.clamp_(0,127)
            for lo,hi in GROUPS30:z[:,lo:hi]=z[:,lo:hi].sort(dim=1,descending=True).values
            z[:,30:]=0
        if (step+1)%100==0:history.append(dict(step=step+1,pixel=float(pixel.detach()),edge=float(edge.detach())));print('DECODER_ONLY_LATENT',history[-1],flush=True)
    with torch.no_grad():
        final=decode(net,z);assert torch.equal(final,fragment(net,net.round(z),'up1_low_dw')['rgb_head']/128) and net.export()==raw
    board=Image.new('RGB',(1680,1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,(rid,label) in enumerate(names):
        for j,(name,p) in enumerate((('离线目标',target),('实际源图预测',torch.stack(source)),('真目标面积注入',initial),('优化隐变量·非源预测',final))):
            im=m.t.r.c.rgb_image(p[i]);im.save(dest/f'{i}_{j}.png');board.paste(im,(j*420,i*314+34));draw.text((j*420+3,i*314+4),label+' | '+name,font=font,fill='white')
    board.save(dest/'decoder_ceiling.png');write_json(dest/'review.json',dict(history=history,checkpoint_sha256=file_hash(a.checkpoint),
      source_prediction=False,latent_fitted_to_target=True,not_deployable=True,decoder_fragment_error=0,coefficients_unchanged=True,
      initial_mae=float((initial-target).abs().mean()*128),optimized_mae=float((final-target).abs().mean()*128),visual_acceptance=False))

if __name__=='__main__':main()
