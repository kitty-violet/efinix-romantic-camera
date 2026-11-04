"""Isolate fixed decoder limits by injecting known region masks (diagnostic).

Every image is explicitly labelled as a target-injection diagnostic, never a
source prediction. The fragment is first compared with the full QAT graph.
"""
import argparse
import copy
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .train_b1_coverage_classifiers import native_coverage
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible


def palette(net):
    states=round(float(net.weight['up2_high_dw'][0,0].sum()*127))
    idx=torch.cartesian_prod(torch.arange(9,device='cuda'),torch.arange(5,device='cuda'),torch.arange(5,device='cuda'))
    features=torch.stack([(idx[:,c]-2*g).clamp(0,2)*(states/2) for c,groups in enumerate((4,2,2)) for g in range(groups)],1)
    s=(net.rgb_head_m/131072).float().flatten()
    wi=net.round(net.weight['rgb_head'][:,:,0,0]/s[:,None]).clamp(-127,127);bi=net.round(net.bias['rgb_head']/s)
    return net.round((features@wi.T+bi)*s).clamp(-128,127)/128,idx


def fragment(net,z,start_node='res3_conv1.pw'):
    nodes={'res1_add':torch.zeros_like(z),'res2_add':torch.zeros_like(z),start_node:z};active=False
    for r in net.plan['nodes']:
        name=r['id'];key=name.replace('.','__')
        if name==start_node:
            active=True
            continue
        if not active:continue
        x=nodes[r['inputs'][0]]
        if key in net.weight:
            w,b=net.weight[key],net.bias[key];mult=getattr(net,key+'_m');s=(mult/131072).float().flatten()
            wi=net.round(w/s[:,None,None,None]).clamp(-127,127);bi=net.round(b/s).double()[None,:,None,None]
            if r['kernel']==3:x=F.pad(x,(1,1,1,1),mode='replicate')
            dot=F.conv2d(x,wi,stride=r['stride'],groups=r['groups'])
            y=net.round((dot.double()+bi).clamp(-2**31,2**31-1)*mult/131072).clamp(-128,127).float()
            if r['activation']=='relu':y=y.clamp_min(0)
        elif r['kind']=='nearest_x2':y=x.repeat_interleave(2,-2).repeat_interleave(2,-1)
        else:
            q=net.package['residual_encoding'][name]
            assert not q['main_scaled'] and q['multiplier']==1
            y=net.round(x.double()*q['post_multiplier']/131072).clamp(-128,127).float()
        nodes[name]=y
    return nodes


@torch.no_grad()
def geometry_variant(net,name):
    keys=[]
    if name in ('no-morph','no-morph-single-aa') or name.startswith('majority3'):
        for key in ('res3_conv2__dw','up1_low_dw'):
            net.weight[key].zero_();net.weight[key][:,0,1,1]=1;net.bias[key].zero_();keys.append(key)
    if name.startswith('majority3'):
        key='res3_conv2__dw';net.weight[key][:,0]=1;net.bias[key].fill_(-4*127)
        if name.startswith('majority3x2'):
            key='up1_low_dw';net.weight[key][:,0]=1;net.bias[key].fill_(-4*127)
    if name.endswith('single-aa'):
        key='up1_high_dw';net.weight[key].zero_();net.weight[key][:,0,1,1]=1;net.bias[key].zero_();keys.append(key)
    for key in keys:
        w=net.weight[key];s=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,key+'_m').copy_((s*131072).clamp(1,131071).double()[None,:,None,None])


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();set_reproducible(20261006203);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=state['residual_encoding'];initial=copy.deepcopy(net.state_dict());colours,idx=palette(net)
    rows={r['id']:r for r in m.load_data()['validation']};records=[]
    sample=m.t.r.tensor(m.t.r.read(next(iter(rows.values()))['path']))[None,...,:64,:64]
    full=net(sample,all_nodes=True);part=fragment(net,full['res3_conv1.pw'])
    errors={k:float((v-full[k]).abs().max()) for k,v in part.items() if k not in ('res2_add','res3_conv1.pw')}
    assert max(errors.values())==0;del full,part
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        target=m.t.r.read(rows[key]['target'])[box[1]:box[3],box[0]:box[2]];t=m.t.r.tensor(target)[None]
        coverage=native_coverage(t,colours,idx);z=F.pad((coverage*127).round(),(0,0,0,0,0,16))
        images=[Image.fromarray(target)]
        for name in ('current','no-morph','no-morph-single-aa'):
            net.load_state_dict(initial);geometry_variant(net,name);p=fragment(net,z)['rgb_head']/128
            im=m.t.r.c.rgb_image(p[0]);images.append(im);im.save(dest/(str(i)+'_'+key+'_'+name+'_injection.png'))
            records.append(dict(id=key,crop=label,variant=name,mae=float((p-t).abs().mean()*128)))
        for j,(im,title) in enumerate(zip(images,['离线目标','当前解码器·注入诊断','移除开闭·注入诊断','单层形状滤波·注入诊断'])):
            board.paste(im,(j*420,i*314+34));draw.text((j*420+4,i*314+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'diagnostic.png');write_json(dest/'results.json',dict(records=records,fragment_node_errors=errors,
        source_prediction=False,target_masks_injected=True,visual_acceptance=False,warm_sha256=file_hash(a.warm)))
    print('DECODER_COVERAGE_DIAGNOSTIC',records,flush=True)


if __name__=='__main__':main()
