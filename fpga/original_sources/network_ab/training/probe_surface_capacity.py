"""Explicit single TRAINING crop capacity check, never a style deliverable."""
import json,time
import torch,numpy as np
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .coarse_skip_net import CoarseSkipNet
from .surface32_net import Surface32Net
from .train_luma_regions import ROOT,RUN,SLICES,labels,display,expected
from .train_b1_distillation import as_tensor,read_image,to_rgb,edges
from .runtime import configure_workspace,contract_guard,write_json,file_hash
from .engine import set_reproducible


def main():
    configure_workspace();contract_guard();set_reproducible(2026100447);root=ROOT/'runs'/RUN;out=root/'surface-capacity-fit-v2';out.mkdir(exist_ok=False)
    source=root/'ordinal-regions-fp32/folded_002000.pt';state=torch.load(source,map_location='cpu',weights_only=True)
    data=json.loads((source.parent/'data.json').read_text(encoding='utf-8'));r=data['pairs'][40];xy=(320,240);x,y=xy
    images=[read_image(r[k])[y:y+320,x:x+320] for k in ['current','current_target']];xx,tt=[as_tensor(v)[None].cuda() for v in images];lab=labels(tt);records=[]
    for kind in ['coarse8','surface32']:
        torch.manual_seed(2026100447);torch.cuda.manual_seed_all(2026100447);base=CoarseSkipNet().eval();base.load_state_dict(state['model'])
        net=base
        if kind=='surface32':
            net=Surface32Net().eval();net.expand_from(base)
            with torch.no_grad():
                for n in net.nodes:
                    if n['kernel'] and net.conv(n['id']).weight.shape!=base.conv(n['id']).weight.shape:net.conv(n['id']).weight.add_(torch.randn_like(net.conv(n['id']).weight)*.001)
        net=net.cuda();opt=torch.optim.Adam(net.parameters(),lr=.0005);start=time.perf_counter();trace=[]
        for step in range(601):
            logits=net(xx);pred=expected(logits);cut=lambda v:v[...,48:-48,48:-48]
            ce=sum(F.cross_entropy(cut(logits[:,s]),cut(l)) for s,l in zip(SLICES,lab))/3
            rgb=F.l1_loss(cut(pred),cut(tt));edge=sum(F.l1_loss(cut(a),cut(b)) for a,b in zip(edges(pred),edges(tt)))
            loss=ce+4*rgb+2*edge
            if step in [0,100,300,600]:
                with torch.no_grad():hard=display(logits);mae=float((cut(hard)-cut(tt)).abs().mean()*128)
                trace.append(dict(step=step,loss=float(loss.detach()),rgb_mae=mae));Image.fromarray(to_rgb(hard[0])).save(out/f'{kind}_{step:03d}.png');print('SURFACE_CAPACITY',kind,trace[-1],flush=True)
            if step==600:break
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True);opt.step()
        torch.save(dict(model=net.cpu().state_dict(),diagnostic_only=True,training_crop_only=True,steps=600,source=str(source),source_sha256=file_hash(source)),out/f'{kind}_diagnostic.pt')
        records.append(dict(kind=kind,seconds=time.perf_counter()-start,trace=trace));del net,opt,base;torch.cuda.empty_cache()
    for name,v in zip(['source','target'],images):Image.fromarray(v).save(out/(name+'.png'))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17);board=Image.new('RGB',(1280,354),'#18202a');d=ImageDraw.Draw(board)
    for i,(name,title) in enumerate([('source','训练裁切原图'),('target','区域目标'),('coarse8_600','当前网络单块拟合'),('surface32_600','扩展上下文单块拟合')]):d.text((320*i+5,5),title,font=font,fill='white');board.paste(Image.open(out/(name+'.png')),(320*i,34))
    board.save(out/'comparison.png');write_json(out/'report.json',dict(status='TRAINING_CROP_CAPACITY_ONLY',records=records,sample=r,xy=xy,source_sha256=file_hash(source),script_sha256=file_hash(__file__),generalization=False,style_accepted=False))


if __name__=='__main__':main()
