"""Counterfactual skip-texture audit; interventions are NOT deployable results."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .b1_capacity_component_net import CapacityComponentNet
from .component_region_net import decode
from .train_component_centers import target_components
from .train_b1_distillation import read_image,as_tensor,to_rgb
from .train_luma_regions import ROOT,RUN
from .runtime import configure_workspace,contract_guard,file_hash,write_json


def box(x,k):return F.avg_pool2d(F.pad(x,(k//2,)*4,mode='replicate'),k,1)


def edge_map(x):
    dx=F.pad((x[:,:,:,1:]-x[:,:,:,:-1]).abs().mean(1,keepdim=True),(0,1,0,0))
    dy=F.pad((x[:,:,1:]-x[:,:,:-1]).abs().mean(1,keepdim=True),(0,0,0,1))
    return torch.maximum(dx,dy)


def texture_scores(p,t):
    a,b=edge_map(p)*128,edge_map(t)*128
    # A five-pixel exclusion around real target boundaries permits alignment
    # error while measuring extra edges well INSIDE the teacher's regions.
    interior=F.max_pool2d((b>4).float(),5,1,2)==0
    real=b>8;near=F.max_pool2d(a,5,1,2)
    return dict(interior_fraction=float(interior.float().mean()),
        extra_edge_fraction=float(((a>4)*interior).sum()/interior.sum().clamp_min(1)),
        interior_gradient=float((a*interior).sum()/interior.sum().clamp_min(1)),
        boundary_recall=float(((near>8)*real).sum()/real.sum().clamp_min(1)),
        edge_fraction=float((a>4).float().mean()),rgb_mae=float((p-t).abs().mean()*128))


@torch.no_grad()
def main():
    configure_workspace();contract_guard();torch.set_num_threads(2)
    base=ROOT/'runs'/RUN;out=base/'capacity-texture-audit-v1';out.mkdir(exist_ok=False)
    ck=base/'b1-capacity-control-v2/capacity/fp32_checkpoint_002000.pt';s=torch.load(ck,map_location='cpu',weights_only=True)
    assert all(file_hash(k)==v for k,v in s['binding']['code'].items())
    net=CapacityComponentNet().cuda().eval();net.load_state_dict(s['model']);data=json.loads((ck.parent/'data.json').read_text(encoding='utf-8'));records=[]
    for r in data['validation']:
        x=as_tensor(read_image(r['source']))[None].cuda();t=as_tensor(read_image(base/'intermediate-target-screen'/f'{r["id"]}_0.008.png'))[None].cuda()
        target=decode(target_components(t));raw=net(x);baseline=decode(raw);Image.fromarray(to_rgb(target[0])).save(out/(r['id']+'_target.png'))
        cases={'baseline':baseline}
        for name,node,gain in [('skip_half','native.pw',.5),('skip_smooth','native.pw',0.),('context_smooth','context.up.pw',0.)]:
            hook=net.layers[node.replace('.','__')].register_forward_hook(lambda m,a,y:box(y,7)+gain*(y-box(y,7)))
            p=net(x);hook.remove();cases[name]=decode(p)
        for name,p in cases.items():
            Image.fromarray(to_rgb(p[0])).save(out/(r['id']+'_'+name+'.png'))
            records.append(dict(id=r['id'],case=name,**texture_scores(p,target)))
        print('TEXTURE_AUDIT',r['id'],[(q['case'],round(q['extra_edge_fraction'],4),round(q['boundary_recall'],4)) for q in records if q['id']==r['id']],flush=True)
    board=Image.new('RGB',(1600,894),'#18202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for j,(key,(x,y)) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400))]):
        for i,(name,title) in enumerate([('target','量化区域目标'),('baseline','当前网络'),('skip_half','诊断：旁路碎纹减半'),('skip_smooth','诊断：旁路碎纹清除'),('context_smooth','诊断：深层碎纹清除')]):
            draw.text((i*320+5,j*298+5),title,font=font,fill='white');board.paste(Image.open(out/(key+'_'+name+'.png')).crop((x,y,x+320,y+264)),(i*320,j*298+30))
    board.save(out/'comparison.png');write_json(out/'report.json',dict(records=records,checkpoint_sha256=file_hash(ck),script_sha256=file_hash(__file__),interventions_are_diagnostic_only=True,not_blind=True,style_accepted=False))


if __name__=='__main__':main()
