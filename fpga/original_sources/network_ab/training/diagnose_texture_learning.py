"""Controlled fixed-patch loss ablation; no original run is modified."""
import copy
import gc
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F
from .coherent_surface_model import SurfaceAppearanceNet
from .coherent_shared_model import warp, temporal_residual_loss
from .engine import set_reproducible
from .losses import PerceptualLoss
from .train_clear_style import tensor, differences
from .runtime import configure_workspace, file_hash, write_json


def terms(net,vgg,batch):
    x,target,flow,mask=batch
    out=net(x);a,b=x.chunk(2);ta,tb=target.chunk(2);oa,ob=out.chunk(2)
    with torch.no_grad():tf=vgg.extract(F.avg_pool2d(target,2))
    of=vgg.extract(F.avg_pool2d(out,2))
    blur=lambda v:F.avg_pool2d(v,5,1,2)
    errors=(ob-tb)-warp(oa-ta,flow)
    values={'paired':F.l1_loss(out,target),
            'multiscale':F.l1_loss(F.avg_pool2d(out,8),F.avg_pool2d(target,8)),
            'detail':F.l1_loss(out-blur(out),target-blur(target)),
            'edge':sum(F.l1_loss(u,v) for u,v in zip(differences(out),differences(target))),
            'perceptual':torch.stack([F.mse_loss(of[k],tf[k])/(tf[k].square().mean()+1e-6) for k in ['relu2_2','relu3_3']]).mean(),
            'temporal_error':(errors.abs()*mask).sum()/(mask.sum()*3+1e-6),
            'temporal_style':temporal_residual_loss(a,b,oa,ob,flow,mask)}
    return out,values


def main():
    root=configure_workspace();parent=root/'runs/clear-style-control-20261003';run=root/'runs/texture-recovery-20261003'
    run.mkdir(exist_ok=True);out=run/'ablation';out.mkdir(exist_ok=True)
    if (out/'report.json').exists():raise FileExistsError('Ablation already completed')
    set_reproducible(202610031);cv2.setNumThreads(2)
    state=torch.load(parent/'student_train/checkpoint_002000.pt',map_location='cpu',weights_only=True)
    data=json.loads((parent/'paired_data/manifest.json').read_text(encoding='utf-8'))
    batches=[];specs=[]
    for ix,xy in [(3,(384,256)),(24,(512,192)),(48,(192,320)),(98,(800,128))]:
        row=next(r for r in data['pairs'] if r['parent_pair_index']==ix);x,y=xy
        for k in ['current','next','current_target','next_target','flow']:assert file_hash(row[k])==row[k+'_sha256']
        images=[np.array(Image.open(row[k]).convert('RGB'))[y:y+256,x:x+256] for k in ['current','next','current_target','next_target']]
        with np.load(row['flow']) as z:
            flow=(cv2.resize(z['backward'].astype(np.float32),(1280,720))*2)[y:y+256,x:x+256]
            mask=cv2.resize(z['mask'].astype(np.float32),(1280,720),interpolation=cv2.INTER_NEAREST)[y:y+256,x:x+256]
        yy,xx=np.mgrid[:256,:256];mask*=((xx+flow[:,:,0]>=4)&(xx+flow[:,:,0]<252)&(yy+flow[:,:,1]>=4)&(yy+flow[:,:,1]<252))
        flow=torch.from_numpy(flow.copy()).permute(2,0,1)[None].cuda();mask=torch.from_numpy(mask.copy())[None,None].cuda()
        batches.append((torch.cat([tensor(im) for im in images[:2]]),torch.cat([tensor(im) for im in images[2:]]),flow,mask))
        specs.append({'parent_pair_index':ix,'xy':xy,'size':256})
    vgg=PerceptualLoss(root).cuda();vgg.features=vgg.features[:16]
    original=state['config']['loss_weights'];variants={'control':dict(original),'no_temporal':dict(original),'texture_emphasis':dict(original)}
    variants['no_temporal'].update(temporal_error=0,temporal_style=0,shift=0)
    variants['texture_emphasis'].update(detail=6,edge=3,perceptual=.5)
    report={'status':'RUNNING','parent':str(parent),'parent_checkpoint_sha256':file_hash(parent/'student_train/checkpoint_002000.pt'),
            'specs':specs,'steps_each':160,'learning_rate':.0002,'variants':variants,'records':[],
            'limitations':'Four repeatedly fitted training patches only; controls diagnose fitting, not generalization or video acceptance.'}
    write_json(out/'config.json',report)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    outputs={}
    for name,weights in variants.items():
        net=SurfaceAppearanceNet(**state['config']['model']).cuda();net.load_state_dict(state['ema'])
        opt=torch.optim.Adam(net.parameters(),lr=.0002);begin=time.perf_counter();logs=[]
        for step in range(160):
            batch=batches[step%len(batches)];pred,values=terms(net,vgg,batch)
            loss=sum(weights[k]*v for k,v in values.items());opt.zero_grad(set_to_none=True);loss.backward()
            if step%4==0 and weights['shift']:
                a=batch[0][:1];actual=net(torch.roll(a,1,3))
                with torch.no_grad():expected=torch.roll(net(a),1,3)
                (weights['shift']*F.l1_loss(actual[:,:,16:-16,16:-16],expected[:,:,16:-16,16:-16])).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),10,error_if_nonfinite=True);opt.step()
            if (step+1)%40==0:print(name,step+1,float(loss.detach()),flush=True)
            logs.append({'step':step+1,**{k:float(v.detach()) for k,v in values.items()}})
        net.eval();scores=[];patches=[]
        with torch.no_grad():
            for x,target,flow,mask in batches:
                pred=net(x);hp=lambda v:v-F.avg_pool2d(v,5,1,2)
                scores.append({'mae':float(F.l1_loss(pred,target)*128),'highpass_mae':float(F.l1_loss(hp(pred),hp(target))*128)})
                patches.append((pred[0].permute(1,2,0).cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8))
        outputs[name]=patches
        dst=out/name;dst.mkdir(exist_ok=True)
        for case in ['quiet_stairs','plaza_stone','park_people']:
            im=np.array(Image.open(parent/'anime'/case/'source.png'))
            with torch.no_grad():pred=net(tensor(im))
            Image.fromarray((pred[0].permute(1,2,0).cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)).save(dst/(case+'.png'))
        torch.save({'model':state['config']['model'],'generator':net.cpu().state_dict(),'variant':name,'steps':160},dst/'probe.pt')
        write_json(dst/'metrics.json',logs)
        report['records'].append({'variant':name,'seconds':time.perf_counter()-begin,'fixed_patch_scores':scores})
        write_json(out/'report.partial.json',report)
        del net,opt,pred,values,loss;gc.collect();torch.cuda.empty_cache()
    for i,batch in enumerate(batches):
        board=Image.new('RGB',(5*256,300),(22,26,32));d=ImageDraw.Draw(board)
        images=[(batch[k][0].permute(1,2,0).cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8) for k in [0,1]]
        images+=[outputs[k][i] for k in variants]
        for j,(label,im) in enumerate(zip(['原图','教师','原损失续跑','去掉时序项','加强纹理项'],images)):
            d.text((j*256+6,8),label,font=font,fill='white');board.paste(Image.fromarray(im),(j*256,44))
        board.save(out/f'patch_{i}.png')
    report['status']='COMPLETED_DIAGNOSTIC';write_json(out/'report.json',report)
    print('ABLATION_COMPLETE',flush=True)


if __name__=='__main__':main()
