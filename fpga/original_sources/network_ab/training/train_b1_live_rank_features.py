"""Function-preserving activation of spare nonlinear B1 feature channels."""
import argparse
import json
import shutil
from pathlib import Path

import torch

from . import train_b1_rank_colour as rank
from . import train_b1_learned_colour as colour
from .seed_b1_rank_blocks import OUT as SEEDS
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible

OUT=SEEDS/'live-features'


@torch.no_grad()
def activate(net):
    records=[];generator=torch.Generator(device='cpu').manual_seed(2026100521)
    for r in net.plan['nodes']:
        key=r['id'].replace('.','__')
        if key not in net.weight or r['activation']!='relu':continue
        w,b=net.weight[key],net.bias[key]
        dormant=((w.abs().flatten(1).amax(1)==0)&(b<=0)).nonzero().flatten().tolist()
        if not dormant:continue
        for c in dormant:
            if r['kind']=='dw3x3':w[c,0,1,1]=1
            elif key=='stem':w[c,c%3,1,1]=.5;b[c]=64
            else:w[c,c%w.shape[1],0,0]=.2;b[c]=8
        records.append(dict(layer=key,revived=dormant))
    # Opposite outgoing coefficients cancel each pair exactly at initialization,
    # while giving both incoming feature paths nonzero, opposite gradients.
    for stage in range(1,4):
        key=f'res{stage}_conv1__pw';out=f'res{stage}_conv2__pw'
        w,b=net.weight[key],net.bias[key];last=net.weight[out]
        for c in range(16,32,2):
            feature=torch.randn(w[c].shape,generator=generator)*.04
            bias=torch.tensor(16.+(c-16)*1.5)
            share=torch.randn(last.shape[0],generator=generator)*.025
            w[c].copy_(feature);w[c+1].copy_(feature);b[c]=bias;b[c+1]=bias
            last[:,c,0,0]=share;last[:,c+1,0,0]=-share
    return records


def make(cache,selected,steps):
    tr=rank.make(cache,selected,steps)
    activate(tr.net)
    return tr


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=3000);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100521);contract_guard();OUT.mkdir(exist_ok=False)
    data,selected,cache=colour.load_data();tr=make(cache,selected,a.steps)
    original=colour.ColourB1(revive=False,source=SEEDS/'blocks/coefficients-padded.bin').cuda().eval()
    errors=[]
    for s in selected:
        item=next(v for v in cache if v['row']['id']==s['id']);x0,y0,x1,y1=s['box'];x=item['source'][None,:,y0:y1,x0:x1]
        with torch.no_grad():errors.append(float((original(x)[0]-tr.net(x)[0]).abs().max()))
    assert max(errors)<.002,errors
    item=next(v for v in cache if v['row']['id']==selected[0]['id']);x=item['source'][None,:,:320,:320]
    p,_=tr.net(x);p.square().mean().backward();gradient={}
    for stage in range(1,4):
        key=f'res{stage}_conv1__pw';g=tr.net.weight[key].grad[16:32]
        gradient[key]=float(g.abs().sum());assert gradient[key]>0
    del original
    write_json(OUT/'activation.json',dict(initial_rgb_errors=errors,spare_feature_gradient=gradient,
        method='Identical feature pairs + opposite outgoing weights; unchanged initial image, nonzero feature gradients'))
    write_json(OUT/'binding.json',dict(steps=a.steps,selected=selected,seed=str(SEEDS/'blocks/coefficients-padded.bin'),
        seed_sha256=file_hash(SEEDS/'blocks/coefficients-padded.bin'),all_B1=True,external_inference=False,
        code={str(p):file_hash(p) for p in [Path(__file__),Path(rank.__file__),Path(colour.__file__)]}))
    for src in (__file__,rank.__file__,colour.__file__):shutil.copy2(src,OUT/Path(src).name)
    native=tr.update(True);crop=tr.update();tr.save(OUT/'resume_probe.pt');ref=tr.update()
    expected={k:v.detach().clone() for k,v in tr.net.state_dict().items()}
    expected_d={k:v.detach().clone() for k,v in tr.critic.state_dict().items()}
    fresh=make(cache,selected,a.steps);fresh.restore(OUT/'resume_probe.pt');got=fresh.update()
    error=max(float((v-expected[k]).abs().max()) for k,v in fresh.net.state_dict().items())
    derror=max(float((v-expected_d[k]).abs().max()) for k,v in fresh.critic.state_dict().items())
    assert error<1e-5 and derror<1e-5 and ref['crop']==got['crop']
    write_json(OUT/'preflight.json',dict(status='PASS',native=native,crop=crop,resume_error=error,critic_resume_error=derror))
    print('LIVE_FEATURE_PREFLIGHT_PASS',errors,error,derror,flush=True)
    del tr,fresh;torch.cuda.empty_cache();torch.manual_seed(2026100521);torch.cuda.manual_seed_all(2026100521)
    tr=make(cache,selected,a.steps);colour.review(tr,data,selected,OUT/'initial',oracle_comparison=False)
    with (OUT/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as f:
        while tr.step<a.steps:
            row=tr.update();f.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('LIVE_FEATURE',tr.step,round(row['loss'],4),round(row['discriminator'],4),flush=True)
            if tr.step%1000==0 or tr.step==a.steps:
                tr.save(OUT/f'checkpoint_{tr.step:06d}.pt');colour.review(tr,data,selected,OUT/f'step_{tr.step:06d}',oracle_comparison=False)
    print('LIVE_FEATURE_COMPLETE',flush=True)


if __name__=='__main__':main()
