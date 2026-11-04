"""Control for MSE averaging: source-pixel plus source-gradient reconstruction.

Same B1, no sharpening at inference and no teacher. Keep the pure-MSE control.
"""
import copy
import json
from pathlib import Path
import torch

from . import train_b1_reconstruction as r
from .engine import set_reproducible
from .runtime import configure_workspace,file_hash,write_json

OUT = r.OUT/'source-gradient-control'
EDGE = .04


def score(evaluation):
    return evaluation['mean_mse']/128**2 + EDGE*sum(x['edge_mae'] for x in evaluation['records'])/(128*len(evaluation['records']))


def run(tr, train, val, dest):
    dest.mkdir(exist_ok=False)
    first=r.evaluate(tr.net,val);best=score(first)
    history=[dict(step=0,score=best,evaluation=first)]
    tr.save(dest/'best.pt')
    with (dest/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        while tr.step<tr.max_steps:
            row=tr.update();log.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('RECON_GRAD',dest.name,tr.step,round(row['loss'],6),flush=True)
            if tr.step%300==0 or tr.step==tr.max_steps:
                e=r.evaluate(tr.net,val);s=score(e)
                history.append(dict(step=tr.step,score=s,evaluation=e))
                if s<best:
                    best=s;tr.save(dest/'best.pt');r.preview(tr.net,val,dest/'best')
                tr.save(dest/'last.pt');write_json(dest/'history.json',history)
                print('RECON_GRAD_EVAL',dest.name,tr.step,e['pooled_psnr'],s,flush=True)
    state=torch.load(dest/'best.pt',map_location='cpu',weights_only=True)
    tr.net.load_state_dict(state['model'])
    return state


def main():
    configure_workspace();set_reproducible(r.SEED+1)
    train,val=r.data();OUT.mkdir(exist_ok=False)
    parent=r.OUT/'fp32/best.pt';state=torch.load(parent,map_location='cpu',weights_only=True)
    write_json(OUT/'binding.json',dict(parent=str(parent),sha256=file_hash(parent),parent_steps=state['step'],
        objective='MSE(input,output)+0.04*L1(first_difference(input),first_difference(output))',
        fp32_steps=1200,qat_steps=900,hardware_changed=False,style=False,code_sha256=file_hash(__file__)))
    tr=r.Trainer(train,1200,False,EDGE);tr.net.load_state_dict(state['model'])
    tr.base_lrs=[.00012,.005]
    best=run(tr,train,val,OUT/'fp32')
    tr.net.calibrate();tr.net.quantized=True;state=copy.deepcopy(tr.net.state_dict())
    write_json(OUT/'ptq.json',r.evaluate(tr.net,val))
    del tr;torch.cuda.empty_cache()
    tr=r.Trainer(train,900,True,EDGE);tr.net.load_state_dict(state)
    best=run(tr,train,val,OUT/'qat')
    r.export_review(tr.net,val,OUT/'package')
    write_json(OUT/'completion.json',dict(fp32=1200,qat=900,qat_selected=best['step'],
        exact_same_deployed_graph=True,perceptual_superiority_not_assumed=True))
    print('RECON_GRAD_COMPLETE',flush=True)


if __name__=='__main__':main()
