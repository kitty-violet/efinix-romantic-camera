"""Function-preserving hidden-channel balancing, then C3 fixed-scale PTQ.

No logit common-mode removal: these three outputs are actual color components.
"""
import argparse,copy,json
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
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100451);out=ROOT/'runs'/RUN/a.name;out.mkdir(exist_ok=False)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);assert state['scales'] is None and all(file_hash(k)==v for k,v in state['binding']['code'].items())
    net=ComponentNet().cuda().eval();net.load_state_dict(state['model']);original=copy.deepcopy(net);data=json.loads((a.checkpoint.parent/'data.json').read_text(encoding='utf-8'));maxima={}
    for i,r in enumerate(data['pairs']):
        assert file_hash(r['current'])==r['current_sha256'];values=net(as_tensor(read_image(r['current']))[None].cuda(),True)
        for k,z in values.items():v=z.abs().amax((0,2,3)).cpu().numpy();maxima[k]=np.maximum(maxima.get(k,v),v)
        if (i+1)%20==0:print('COMPONENT_BALANCE',i+1,flush=True)
    parent={k:k for k in maxima}
    def find(k):
        while parent[k]!=k:k=parent[k]
        return k
    for n in net.nodes:
        if n['kind'] in ('nearest_x2','residual_add'):
            for k in n['inputs']:parent[find(k)]=find(n['id'])
    pooled={}
    for k,v in maxima.items():g=find(k);pooled[g]=np.maximum(pooled.get(g,v),v)
    factors={k:np.clip(1/np.maximum(pooled[find(k)],1e-3),.001,100.) for k in maxima};factors['input']=np.ones(3);factors['rgb']=np.ones(3)
    for n in net.nodes:
        if not n['kernel']:continue
        c=net.conv(n['id']);fi=torch.tensor(factors[n['inputs'][0]],device='cuda');fo=torch.tensor(factors[n['id']],device='cuda')
        c.weight.mul_(fo[:,None,None,None]);c.bias.mul_(fo);c.weight.div_(fi[:,None,None,None] if n['kind']=='dw3x3' else fi[None,:,None,None])
    fp=[]
    for r in data['validation']:
        x=as_tensor(read_image(r['source']))[None].cuda();old=original(x);new=net(x);error=float((old-new).abs().max());fraction=float((decode(old)!=decode(new)).any(1).float().mean());assert error<1e-4 and fraction<1e-4
        fp.append(dict(id=r['id'],component_max_error=error,decoded_pixel_change=fraction))
    groups={}
    for i,r in enumerate(data['pairs']):groups.setdefault(r['source'],[]).append(i)
    indices=[j for g in groups.values() for j in (g[0],g[len(g)//2],g[-1])];cal=net.calibrate(as_tensor(read_image(data['pairs'][i]['current']))[None].cuda() for i in indices);assert net.scales['rgb']==1/128
    package=net.integer_package();package.update(checkpoint=str(a.checkpoint),checkpoint_sha256=file_hash(a.checkpoint),trained_steps=state['step'],training_mode=state['mode']);write_json(out/'runtime_parameters.json',package)
    rng=np.random.default_rng(2026100451);tests=[]
    arrays=[rng.integers(-128,128,(1,3,8,16),dtype=np.int64),np.full((1,3,8,8),-128,np.int64),np.full((1,3,8,8),127,np.int64)]
    for i,arr in enumerate(arrays):
        cpu=execute(package,arr,False);x=torch.tensor(arr,device='cuda',dtype=torch.float32);gpu=execute(package,x,True);qat=net(x/128,True);errors={}
        for key,v in cpu.items():
            errors[key]=int(np.max(np.abs(v-gpu[key].cpu().numpy())));error=int(np.max(np.abs(v-(qat[key]/net.scales[key]).round().cpu().numpy())));assert errors[key]==error==0,(i,key,errors[key],error)
        tests.append(dict(case=i,shape=list(arr.shape),layer_errors=errors))
    rows=[]
    for r in data['validation']:
        arr=read_image(r['source']);x=as_tensor(arr)[None].cuda();pred=net(x);q=execute(package,torch.tensor((arr.astype(np.float32)-128).transpose(2,0,1)[None],device='cuda'),True)['rgb'];err=float((pred*128-q).abs().max());assert err<.0001
        rgb=to_rgb(decode(pred)[0]);Image.fromarray(rgb).save(out/(r['id']+'.png'));f=to_rgb(decode(original(x))[0]);rows.append(dict(id=r['id'],native720_integer_error=err,ptq_rgb_mae=float(np.abs(rgb.astype(float)-f).mean())))
    binding=copy.deepcopy(state['binding']);binding['code'][str(Path(__file__))]=file_hash(__file__)
    torch.save(dict(model=net.state_dict(),scales=net.scales,calibration=cal,mode='component_ptq',step=state['step'],binding=binding,config=state['config'],source=str(a.checkpoint),source_sha256=file_hash(a.checkpoint)),out/'equalized.pt')
    write_json(out/'report.json',dict(status='C3_HIDDEN_BALANCE_INTEGER_NUMERIC_PASS',fp32=fp,tests=tests,native720=rows,calibration=cal,factors={k:v.tolist() for k,v in factors.items()},output_common_mode_removed=False,style=False,rtl=False,fit=False,timing=False));print('COMPONENT_PTQ_PASS',rows,flush=True)


if __name__=='__main__':main()
