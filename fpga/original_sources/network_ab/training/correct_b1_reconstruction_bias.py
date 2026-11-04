"""Training-set-only sequential bias calibration before true B1 QAT."""
import argparse
import copy
from pathlib import Path
import torch
from . import train_b1_reconstruction as r
from . import research_b1_reconstruction as s
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--balance-root',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--targets',nargs='+',type=int,default=[0,80,112]);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100568);contract_guard();train,val=r.data()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    samples=train[::max(1,len(train)//6)][:6]
    xs=[r.tensor(r.read(z['path']))[None] for z in samples]
    trials=[]
    for target in a.targets:
        source=Path(a.balance_root)/f'target_{target}.pt'
        state=torch.load(source,map_location='cpu',weights_only=True)
        net=r.initialize();net.load_state_dict(state['model']);net.quantized=False
        wanted={}
        for x in xs:
            nodes=net(x,all_nodes=True)
            for row in net.plan['nodes']:
                if row['kernel']:
                    key=row['id'];value=nodes[key].mean((0,2,3))
                    wanted[key]=wanted.get(key,torch.zeros_like(value))+value/len(xs)
        net.quantized=True;before=s.evaluate(net,val)
        best=before['pooled_psnr'];chosen=copy.deepcopy(net.state_dict());corrections=[]
        for sweep in range(2):
            for row in net.plan['nodes']:
                if not row['kernel']:continue
                mean=torch.zeros_like(wanted[row['id']]);active=torch.zeros_like(mean)
                for x in xs:
                    z=net(x,all_nodes=True)[row['id']]
                    mean+=z.mean((0,2,3))/len(xs)
                    valid=(z>0)&(z<127) if row['activation']=='relu' else (z>-128)&(z<127)
                    active+=valid.float().mean((0,2,3))/len(xs)
                delta=((wanted[row['id']]-mean)/active.clamp_min(.2)).clamp(-8,8)
                net.bias[row['id'].replace('.','__')].add_(delta)
                corrections.append(dict(sweep=sweep,node=row['id'],delta=delta.cpu().tolist()))
            evaluation=s.evaluate(net,val)
            if evaluation['pooled_psnr']>best:
                best=evaluation['pooled_psnr'];chosen=copy.deepcopy(net.state_dict())
            print('BIAS_CORRECTION',target,sweep,before['pooled_psnr'],evaluation['pooled_psnr'],flush=True)
        net.load_state_dict(chosen);evaluation=s.evaluate(net,val)
        path=out/f'target_{target}.pt'
        torch.save(dict(model=chosen,step=state['step'],qat=True,contract_sha256=r.CONTRACT_SHA,
            parent=str(source),parent_sha256=file_hash(source),calibration_only=True),path)
        trials.append(dict(target=target,checkpoint=str(path),before=before,after=evaluation,corrections=corrections))
        write_json(out/'results.json',dict(trials=trials,calibration_sources=samples,
            hardware_changed=False,original_image_target=True,final_test_not_read=True))
        del net;torch.cuda.empty_cache()
    chosen=max(trials,key=lambda z:z['after']['pooled_psnr'])
    write_json(out/'selection.json',dict(checkpoint=chosen['checkpoint'],target=chosen['target'],
        psnr=chosen['after']['pooled_psnr'],rule='Reviewed-validation PTQ PSNR; test set unused'))


if __name__=='__main__':main()
