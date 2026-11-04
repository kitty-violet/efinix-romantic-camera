"""Isolate parameter/activation quantization and recover the actual B1 bytes."""
import argparse
import copy
import json
import math
import shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch.nn import functional as F
from .probe_b1_single_crop_fit import MasterTiedStudent,inputs,objective,update,OUT as CROP_RUN
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .b1_trained_image import runtime_from_image,decode
from .probe_b1_cartoon_int8 import IntegerB1
from .engineer_b1_blocks import oracle_records
from network_ab.contract_reference import execute_integer

OUT=CROP_RUN.parent/'b1-crop-quantization-20261005-v1'


def split_forward(net,x,weights=False,activations=False,return_nodes=False):
    rw=net.round if weights else lambda a:a
    ra=net.round if activations else lambda a:a
    nodes={'input':ra(x*128).clamp(-128,127)}
    for r in net.plan['nodes']:
        name=r['id'];key=name.replace('.','__');a=nodes[r['inputs'][0]]
        if key in net.weight:
            w=rw(net.weight[key]).clamp(-127,127);b=rw(net.bias[key])[None,:,None,None]
            if r['kernel']==3:a=F.pad(a,(1,1,1,1),mode='replicate')
            dot=F.conv2d(a,w,None,stride=r['stride'],groups=r['groups'])
            y=ra((dot.double()+b).clamp(-(2**31),2**31-1)*getattr(net,key+'_m')/131072).clamp(-128,127).float()
            if r['activation']=='relu':y=y.clamp_min(0)
        elif r['kind']=='nearest_x2':y=a.repeat_interleave(2,-2).repeat_interleave(2,-1)
        else:
            q=net.package['residual_encoding'][name];z=nodes[r['inputs'][1]]
            small,other=(a,z) if q['main_scaled'] else (z,a)
            aligned=ra(small.double()*q['multiplier']/131072).clamp(-128,127)
            y=ra((aligned+other.double())*q['post_multiplier']/131072).clamp(-128,127).float()
        nodes[name]=y
        if name=='res2_add':break
    return nodes if return_nodes else nodes['res2_add'][:,24:25]


def calibrate_parameter_precision(net):
    changes=[]
    with torch.no_grad():
        for key,w in net.weight.items():
            if ('w',key) not in net.masks:continue
            active=getattr(net,'w_'+key+'_mask').flatten(1).any(1)
            m=getattr(net,key+'_m');b=net.bias[key]
            for c in range(len(w)):
                if not active[c]:continue
                maximum=float(w[c].abs().max());old=float(m[0,c,0,0])
                new=max(1,min(131071,math.ceil(old*max(maximum,1e-4)/112)))
                factor=old/new
                w[c].mul_(factor);b[c].mul_(factor);m[0,c,0,0]=new
                getattr(net,'w_'+key+'_initial')[c].mul_(factor);getattr(net,'b_'+key+'_initial')[c].mul_(factor)
                changes.append(dict(layer=key,channel=c,old_multiplier=old,new_multiplier=new,weight_max_before=maximum))
            net.package['parameters'][key.replace('__','.')]['multipliers']=m.flatten().to(torch.int64).tolist()
    return changes


def score(p,t):
    a=p.detach()[0,0].cpu().numpy().clip(0);desired=(t[0,0].cpu().numpy()*32-8).clip(0)
    a=a[8:-8,8:-8];desired=desired[8:-8,8:-8];m=a>4;n=desired>4
    return dict(loss=float(objective(p,t).detach()),precision=float((m&n).sum()/max(m.sum(),1)),
        recall=float((m&n).sum()/max(n.sum(),1)),mae=float(abs(a-desired).mean()))


def contact(images,path):
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17);board=Image.new('RGB',(320*len(images),354),'#17202a');draw=ImageDraw.Draw(board)
    for i,(label,a) in enumerate(images):
        if isinstance(a,Image.Image):im=a
        else:im=Image.fromarray(np.clip(a*255/24,0,255).astype(np.uint8)).resize((320,320),Image.Resampling.NEAREST).convert('RGB')
        board.paste(im,(320*i,34));draw.text((320*i+4,5),label,font=font,fill='white')
    board.save(path)


@torch.no_grad()
def inspect(net,x,t,rgb,tag):
    dest=OUT/tag;dest.mkdir(exist_ok=False);net.quantized=True
    raw=net.export();(dest/'coefficients-padded.bin').write_bytes(raw)
    model=IntegerB1(runtime_from_image((dest/'coefficients-padded.bin').read_bytes()))
    actual,_=model(x*128,True);qat=net(x,True,True)
    errors={k:float((v-actual[k]).abs().max()) for k,v in qat.items()};assert max(errors.values())==0
    gate=actual[net.first_gate_node][:,24:25]
    target=(t[0,0].cpu().numpy()*32-8).clip(0);a=gate[0,0].cpu().numpy()
    contact([('固定训练裁块',Image.fromarray(rgb)),('同网格目标',target),('权重字节重载输出',a)],dest/'actual_integer_comparison.png')
    Image.fromarray(np.clip(a*255/24,0,255).astype(np.uint8)).resize((320,320),Image.Resampling.NEAREST).save(dest/'gate.png')
    result=dict(score=score(actual['res2_add'][:,24:25],t),export_node_errors=errors,weight_sha256=file_hash(dest/'coefficients-padded.bin'))
    write_json(dest/'verification.json',result);print('INT8_CROP',tag,result['score'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['diagnose','qat']);ap.add_argument('--steps',type=int,default=1200);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100508);contract_guard();row,box,rgb,x,t=inputs()
    checkpoint=CROP_RUN/'fixed/checkpoint.pt';net=MasterTiedStudent().cuda().eval()
    net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']);net.quantized=False
    original=copy.deepcopy(net)
    with torch.no_grad():reference=net(x).clone()
    changes=calibrate_parameter_precision(net)
    with torch.no_grad():preservation=float((net(x)-reference).abs().max())
    assert preservation<.02,preservation
    if a.mode=='diagnose':
        OUT.mkdir(exist_ok=False);scores={};images=[('同网格目标',(t[0,0].cpu().numpy()*32-8).clip(0))];layers={}
        with torch.no_grad():
            for name,w,ac,label in [('fp32',False,False,'原浮点'),('weights_only',True,False,'只取整权重'),('activations_only',False,True,'只取整中间特征'),('both',True,True,'两者同时取整')]:
                p=split_forward(original,x,w,ac);scores[name]=score(p,t);images.append((label,p[0,0].cpu().numpy().clip(0)))
                if name in ('fp32','both'):layers[name]=split_forward(original,x,w,ac,True)
            p=split_forward(net,x,True,True);scores['calibrated']=score(p,t);images.append(('参数精度重新校准',p[0,0].cpu().numpy().clip(0)))
            net.quantized=True;assert float((net(x)-p).abs().max())==0
        contact(images,OUT/'quantization_causes.png')
        layer_errors={k:dict(mae=float((v-layers['both'][k]).abs().mean()),max_error=float((v-layers['both'][k]).abs().max())) for k,v in layers['fp32'].items()}
        write_json(OUT/'diagnosis.json',dict(scores=scores,layer_errors=layer_errors,precision_calibration=changes,
            fp32_preservation_error=preservation,checkpoint_sha256=file_hash(checkpoint),code_sha256=file_hash(__file__),training_crop_only=True))
        torch.save(dict(model=net.state_dict(),runtime=net.package),OUT/'calibrated_initial.pt')
        inspect(net,x,t,rgb,'initial_calibrated');print('QUANT_DIAGNOSED',scores,flush=True);return
    diagnostic=json.loads((OUT/'diagnosis.json').read_text());assert diagnostic['code_sha256']==file_hash(__file__) and diagnostic['checkpoint_sha256']==file_hash(checkpoint)
    net.quantized=True;opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=.04)
    # Check saving/reloading preserves the next actual integer training update.
    update(net,opt,x,t);saved=copy.deepcopy(net.state_dict());optimizer=copy.deepcopy(opt.state_dict())
    ref=update(net,opt,x,t);expected=copy.deepcopy(net.state_dict())
    net.load_state_dict(saved);opt.load_state_dict(optimizer);got=update(net,opt,x,t)
    resume=max(float((v-expected[k]).abs().max()) for k,v in net.state_dict().items());assert resume==0 and ref==got
    # Restart this short controlled run from the calibrated checkpoint.
    initial=torch.load(OUT/'calibrated_initial.pt',map_location='cpu',weights_only=True)
    net.load_state_dict(initial['model']);net.package=initial['runtime'];opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=.04)
    write_json(OUT/'qat_preflight.json',dict(resume_error=resume,first_actual_loss=ref,actual_integer_forward=True));shutil.copy2(__file__,OUT/Path(__file__).name)
    with (OUT/'qat_metrics.jsonl').open('x',encoding='utf-8',buffering=1) as f:
        for step in range(1,a.steps+1):
            loss=update(net,opt,x,t);f.write(json.dumps(dict(step=step,loss=loss))+'\n')
            if step%100==0:print('CROP_QAT',step,round(loss,5),flush=True)
            if step%400==0 or step==a.steps:inspect(net,x,t,rgb,f'qat_{step:06d}')
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),runtime=net.package,steps=a.steps),OUT/'checkpoint_final.pt')
    raw=net.export();p,_=decode(raw);op,rp=oracle_records(p);contract,_=contract_guard()
    sample=np.random.default_rng(29).integers(-128,128,(12,12,3),dtype=np.int16);reference=execute_integer(contract,sample.tolist(),op,rp)
    actual,_=IntegerB1(runtime_from_image(raw))(torch.tensor(sample.transpose(2,0,1)[None],device='cuda',dtype=torch.float32),True)
    error=max(int(np.abs(v[0].permute(1,2,0).cpu().numpy()-np.array(reference['tensors'][k])).max()) for k,v in actual.items());assert error==0
    write_json(OUT/'qat_result.json',dict(independent_integer_oracle_max_error=error,steps=a.steps,training_crop_only=True,generalization_accepted=False,complete_style=False))


if __name__=='__main__':main()
