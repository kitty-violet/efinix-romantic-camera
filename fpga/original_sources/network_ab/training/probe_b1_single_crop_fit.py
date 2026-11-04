"""Controlled fit: isolate the eight-copy contour readout projection defect.

The held-out scenes are never used here. Identical weights, crop and objective
are used in both arms; only the post-update tying rule differs.
"""
import argparse
import copy
import json
import shutil
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch.nn import functional as F
from .train_b1_deep_contour import DeepStudent,OUT as PARENT
from .train_b1_structural_gate import BASE
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .b1_trained_image import runtime_from_image,decode
from .probe_b1_cartoon_int8 import IntegerB1
from .engineer_b1_blocks import oracle_records
from network_ab.contract_reference import execute_integer

OUT=PARENT.parent/'b1-single-crop-fit-20261005-v1'


class LegacyMeanStudent(DeepStudent):
    """Keep the diagnosed old rule explicit after fixing the live trainer."""
    def project(self):
        with torch.no_grad():
            for kind,key in self.masks:
                value=(self.weight if kind=='w' else self.bias)[key]
                mask=getattr(self,kind+'_'+key+'_mask');base=getattr(self,kind+'_'+key+'_initial')
                value.copy_(value*mask+base*(1-mask))
                if kind=='w':value.clamp_(-127,127)
            for params in (self.weight,self.bias):
                for key in ('res1_conv2__pw','res2_conv2__pw'):
                    value=params[key]
                    value[24:32].copy_(value[24:32].mean(0,keepdim=True).expand_as(value[24:32]))


class MasterTiedStudent(DeepStudent):
    def project(self):
        with torch.no_grad():
            for kind,key in self.masks:
                value=(self.weight if kind=='w' else self.bias)[key]
                mask=getattr(self,kind+'_'+key+'_mask');base=getattr(self,kind+'_'+key+'_initial')
                value.copy_(value*mask+base*(1-mask))
                if kind=='w':value.clamp_(-127,127)
            # Only channel24 enters the loss. Replicas are deployment copies,
            # not seven additional estimates to average with its new update.
            for params in (self.weight,self.bias):
                value=params['res2_conv2__pw'];value[25:32].copy_(value[24:25].expand_as(value[25:32]))


def objective(p,t):
    p=p[:,:,8:-8,8:-8];t=t[:,:,8:-8,8:-8];pos=t>.12;neg=~pos
    error=F.smooth_l1_loss(p/16,(t*32-8)/16,reduction='none')
    positive=(error*pos).sum()/pos.sum().clamp_min(1);negative=(error*neg).sum()/neg.sum().clamp_min(1)
    false=(F.relu(p+2)/16*neg).sum()/neg.sum().clamp_min(1)
    return positive+1.5*negative+.4*false


def update(net,opt,x,t,trace=False):
    opt.zero_grad(set_to_none=True);loss=objective(net(x),t);loss.backward()
    for kind,key in net.masks:
        value=(net.weight if kind=='w' else net.bias)[key]
        if value.grad is not None:value.grad.mul_(getattr(net,kind+'_'+key+'_mask'))
    torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],10,error_if_nonfinite=True)
    p=net.weight['res2_conv2__pw'];before=p[24].detach().clone()
    gradient_rows=[float(v.abs().sum()) for v in p.grad[24:32]] if trace else None
    opt.step();candidate=p[24].detach().clone();net.project()
    if not trace:return float(loss.detach())
    raw=(candidate-before).norm().item();kept=(p[24].detach()-before).norm().item()
    return dict(loss=float(loss.detach()),gradient_by_copy=gradient_rows,
        optimizer_update_norm=raw,post_projection_update_norm=kept,retained_update_ratio=kept/max(raw,1e-30))


def inputs():
    data=json.loads((BASE/'data.json').read_text(encoding='utf-8'))
    pair=next(r for r in data['pairs'] if r['source'].endswith('VID_20260925_121509.mp4'))
    manifest=json.loads((PARENT/'data.json').read_text(encoding='utf-8'))
    row=next(r for r in manifest if r['source']==pair['current']);assert row['split']=='train'
    for key in ('source','target','label'):assert file_hash(row[key])==row[key+'_sha256']
    box=(448,320,768,640)
    rgb=np.array(Image.open(row['source']).convert('RGB').crop(box));label=np.load(row['label']).astype(np.float32)[320:640,448:768]
    x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),device='cuda',dtype=torch.float32)/128-1
    t=torch.tensor(label[::4,::4].copy(),device='cuda')[None,None]
    return row,box,rgb,x,t


@torch.no_grad()
def inspect(models,rgb,x,t,tag):
    dest=OUT/tag;dest.mkdir(exist_ok=False);images={'source':Image.fromarray(rgb)};scores={}
    desired=(t[0,0].cpu().numpy()*32-8).clip(0)
    show=lambda a:Image.fromarray(np.clip(a*255/24,0,255).astype(np.uint8)).resize((320,320),Image.Resampling.NEAREST).convert('RGB')
    images['target']=show(desired)
    for name,net in models.items():
        net.quantized=False;p=net(x);a=p[0,0].cpu().numpy().clip(0);images[name]=show(a)
        mask=np.zeros_like(a,dtype=bool);mask[8:-8,8:-8]=True
        active=(a>4)&mask;truth=(desired>4)&mask
        scores[name]=dict(loss=float(objective(p,t)),precision=float((active&truth).sum()/max(active.sum(),1)),
            recall=float((active&truth).sum()/max(truth.sum(),1)),mae=float(abs(a-desired)[mask].mean()))
        Image.fromarray(a.astype(np.float32),mode='F').save(dest/(name+'_gate.tiff'))
    titles={'source':'固定训练裁块','target':'实际粗轮廓目标','before':'训练前','old':'旧：八份平均','fixed':'修复：主通道复制'}
    board=Image.new('RGB',(320*len(images),354),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    for i,(name,im) in enumerate(images.items()):
        board.paste(im,(320*i,34));draw.text((320*i+5,5),titles.get(name,name),font=font,fill='white');im.save(dest/(name+'.png'))
    board.save(dest/'comparison.png');write_json(dest/'scores.json',scores)
    print('CROP_EVALUATE',tag,scores,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','train']);ap.add_argument('--steps',type=int,default=800);args=ap.parse_args()
    configure_workspace();set_reproducible(2026100507);contract_guard()
    row,box,rgb,x,t=inputs();checkpoint=PARENT/'checkpoint_002000.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=True)['model']
    before=DeepStudent().cuda().eval();before.load_state_dict(state)
    old=LegacyMeanStudent().cuda().eval();old.load_state_dict(state)
    fixed=MasterTiedStudent().cuda().eval();fixed.load_state_dict(state)
    models={'old':old,'fixed':fixed};opts={k:torch.optim.Adam([p for p in n.parameters() if p.requires_grad],lr=.025) for k,n in models.items()}
    if args.mode=='prepare':
        OUT.mkdir(exist_ok=False)
        binding=dict(source=row,crop=list(box),parent_checkpoint=str(checkpoint),parent_sha256=file_hash(checkpoint),
            code_sha256=file_hash(__file__),steps=args.steps,scope='Single training crop diagnostic, no claim of generalization')
        write_json(OUT/'binding.json',binding)
        inspect({'before':before},rgb,x,t,'before')
        result={k:update(n,opts[k],x,t,True) for k,n in models.items()}
        assert abs(result['old']['retained_update_ratio']-.125)<.002
        assert abs(result['fixed']['retained_update_ratio']-1)<.002
        assert all(v==0 for v in result['old']['gradient_by_copy'][1:])
        write_json(OUT/'one_step_reproduction.json',result);print('TIE_DEFECT_CONFIRMED',result,flush=True)
        return
    binding=json.loads((OUT/'binding.json').read_text(encoding='utf-8'));assert binding['code_sha256']==file_hash(__file__) and binding['parent_sha256']==file_hash(checkpoint)
    shutil.copy2(__file__,OUT/Path(__file__).name)
    start=time.perf_counter()
    with (OUT/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as f:
        for step in range(1,args.steps+1):
            losses={k:update(n,opts[k],x,t) for k,n in models.items()}
            f.write(json.dumps(dict(step=step,loss=losses))+'\n')
            if step%100==0:print('SINGLE_CROP',step,losses,round(time.perf_counter()-start,1),flush=True)
            if step%400==0 or step==args.steps:inspect({'before':before,**models},rgb,x,t,f'step_{step:06d}')
    results={}
    contract,_=contract_guard()
    for name,net in models.items():
        dest=OUT/name;dest.mkdir();torch.save(dict(model=net.state_dict(),optimizer=opts[name].state_dict(),steps=args.steps,binding=binding),dest/'checkpoint.pt')
        raw=net.export();(dest/'coefficients-padded.bin').write_bytes(raw);integer=IntegerB1(runtime_from_image((dest/'coefficients-padded.bin').read_bytes()))
        net.quantized=True
        with torch.no_grad():
            nodes=net(x,True,True);reference,_=integer(x*128,True)
            error=max(float((v-reference[k]).abs().max()) for k,v in nodes.items());assert error==0
            gate=reference[net.first_gate_node][0,24].cpu().numpy()
            Image.fromarray(np.clip(gate*255/24,0,255).astype(np.uint8)).resize((320,320),Image.Resampling.NEAREST).save(dest/'actual_integer_gate.png')
            p,_=decode(raw);op,rp=oracle_records(p);sample=np.random.default_rng(9).integers(-128,128,(12,12,3),dtype=np.int16)
            scalar=execute_integer(contract,sample.tolist(),op,rp);got,_=integer(torch.tensor(sample.transpose(2,0,1)[None],device='cuda',dtype=torch.float32),True)
            oracle_error=max(int(np.abs(v[0].permute(1,2,0).cpu().numpy()-np.array(scalar['tensors'][k])).max()) for k,v in got.items());assert oracle_error==0
        results[name]=dict(export_all_node_max_error=error,independent_oracle_max_error=oracle_error,weight_sha256=file_hash(dest/'coefficients-padded.bin'))
    write_json(OUT/'result.json',dict(arms=results,seconds=time.perf_counter()-start,trained_crop_only=True,style_accepted=False))


if __name__=='__main__':main()
