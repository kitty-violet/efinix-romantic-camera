"""Bind final C3 QAT parameters and independently verify integer arithmetic."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from PIL import Image
from .component_region_net import ComponentNet,decode
from .train_luma_regions import ROOT,RUN
from .train_b1_distillation import as_tensor,read_image,to_rgb
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .probe_region_class_int8 import execute


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args();configure_workspace();contract_guard();set_reproducible(2026100453)
    out=ROOT/'runs'/RUN/a.name;out.mkdir(exist_ok=False);state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);assert state['mode']=='qat' and all(file_hash(k)==v for k,v in state['binding']['code'].items())
    net=ComponentNet().cuda().eval();net.load_state_dict(state['model']);net.scales=state['scales'];assert net.scales['rgb']==1/128
    package=net.integer_package();package.update(checkpoint=str(a.checkpoint),checkpoint_sha256=file_hash(a.checkpoint),trained_steps=state['step'],training_mode='qat');write_json(out/'runtime_parameters.json',package)
    rng=np.random.default_rng(2026100453);arrays=[rng.integers(-128,128,(1,3,16,24),dtype=np.int64),np.full((1,3,8,8),-128,np.int64),np.full((1,3,8,8),127,np.int64)];tests=[]
    for i,arr in enumerate(arrays):
        cpu=execute(package,arr,False);x=torch.tensor(arr,device='cuda',dtype=torch.float32);gpu=execute(package,x,True);qat=net(x/128,True);errors={}
        for k,v in cpu.items():
            errors[k]=int(np.max(np.abs(v-gpu[k].cpu().numpy())));err=int(np.max(np.abs(v-(qat[k]/net.scales[k]).round().cpu().numpy())));assert errors[k]==err==0,(i,k,errors[k],err)
        tests.append(dict(index=i,shape=list(arr.shape),layer_errors=errors))
    data=json.loads((a.checkpoint.parent/'data.json').read_text(encoding='utf-8'));rows=[]
    for r in data['validation']:
        arr=read_image(r['source']);x=as_tensor(arr)[None].cuda();pred=net(x);q=execute(package,torch.tensor((arr.astype(np.float32)-128).transpose(2,0,1)[None],device='cuda'),True)['rgb'];err=float((pred*128-q).abs().max());assert err==0
        Image.fromarray(to_rgb(decode(pred)[0])).save(out/(r['id']+'.png'));np.save(out/(r['id']+'_components.npy'),q[0].permute(1,2,0).cpu().numpy().astype(np.int16));rows.append(dict(id=r['id'],integer_error=err));print('C3_QAT_INTEGER',r['id'],err,flush=True)
    write_json(out/'verification.json',dict(status='C3_QAT_INTEGER_NUMERIC_PASS_NOT_STYLE',tests=tests,native720=rows,checkpoint_sha256=file_hash(a.checkpoint),package_sha256=file_hash(out/'runtime_parameters.json'),script_sha256=file_hash(__file__),style=False,rtl=False,board=False))


if __name__=='__main__':main()
