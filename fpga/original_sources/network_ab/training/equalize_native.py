"""Training-source cross-layer balancing for the checked ContextNet graph."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .skip_filter_net import SkipFilterNet
from .train_luma_regions import ROOT,RUN,SLICES,display
from .train_b1_distillation import as_tensor,read_image,to_rgb
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


class EqualizedNative(SkipFilterNet):
    @torch.no_grad()
    def calibrate(self,inputs):
        inputs=list(inputs);report=super().calibrate(inputs);old=self.scales['rgb'];saved=self.scales;self.scales=None;peak=0
        for x in inputs:
            p=self(x)
            for s in SLICES:peak=max(peak,float(p[:,s].topk(2,dim=1).values.abs().max()))
        self.scales=saved;self.scales['rgb']=max(peak*1.12/127,1/512)
        report.update(scales=self.scales,head_fullrange_scale=old,head_top2_peak=peak,head_top2_scale=self.scales['rgb'])
        return report


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100439);root=ROOT/'runs'/RUN;out=root/a.name;out.mkdir(exist_ok=False)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    assert state['scales'] is None and all(file_hash(k)==v for k,v in state['binding']['code'].items())
    net=EqualizedNative().cuda().eval();net.load_state_dict(state['model']);original=copy.deepcopy(net)
    data=json.loads((a.checkpoint.parent/'data.json').read_text(encoding='utf-8'));xx=np.zeros((9,9),np.float64);xy=np.zeros((9,3),np.float64);maxima={}
    for i,r in enumerate(data['pairs']):
        assert file_hash(r['current'])==r['current_sha256']
        x=as_tensor(read_image(r['current']))[None].cuda();v=net(x,True)
        for k,z in v.items():
            c=z.abs().amax((0,2,3)).cpu().numpy();maxima[k]=np.maximum(maxima.get(k,c),c)
        f=v['finish.dw'][0,:,::16,::16].reshape(8,-1).T.cpu().double().numpy();f=np.concatenate((f,np.ones((len(f),1))),1)
        p=v['rgb'][0,:,::16,::16].reshape(32,-1);top=np.stack([p[s].max(0).values.cpu().double().numpy() for s in SLICES],1)
        xx+=f.T@f;xy+=f.T@top
        if (i+1)%20==0:print('NATIVE_EQUALIZE',i+1,flush=True)
    coef=np.linalg.lstsq(xx+np.eye(9)*1e-8,xy,rcond=1e-12)[0];head=net.conv('rgb')
    for i,s in enumerate(SLICES):
        head.weight[s,:,0,0]-=torch.tensor(coef[:8,i],device='cuda',dtype=torch.float32)[None];head.bias[s]-=float(coef[8,i])
    parent={n:n for n in maxima}
    def find(x):
        while parent[x]!=x:x=parent[x]
        return x
    def union(a,b):parent[find(a)]=find(b)
    for n in net.nodes:
        if n['kind'] in ('nearest_x2','residual_add'):
            for k in n['inputs']:union(n['id'],k)
    pooled={}
    for k,v in maxima.items():g=find(k);pooled[g]=np.maximum(pooled.get(g,v),v)
    factors={k:np.clip(1/np.maximum(pooled[find(k)],1e-3),.001,100.) for k in maxima};factors['input']=np.ones(3);factors['rgb']=np.ones(32)
    for n in net.nodes:
        if not n['kernel']:continue
        c=net.conv(n['id']);fi=torch.tensor(factors[n['inputs'][0]],device='cuda');fo=torch.tensor(factors[n['id']],device='cuda')
        c.weight.mul_(fo[:,None,None,None]);c.bias.mul_(fo);c.weight.div_(fi[:,None,None,None] if n['kind']=='dw3x3' else fi[None,:,None,None])
    checks=[]
    for r in data['validation']:
        x=as_tensor(read_image(r['source']))[None].cuda();a0=original(x);b=net(x);errs=[]
        for s in SLICES:errs.append(float(((a0[:,s]-a0[:,s].mean(1,keepdim=True))-(b[:,s]-b[:,s].mean(1,keepdim=True))).abs().max()))
        fraction=float(np.any(to_rgb(display(a0)[0])!=to_rgb(display(b)[0]),2).mean());assert max(errs)<.0005 and fraction<1e-4,(r['id'],errs,fraction)
        checks.append(dict(id=r['id'],common_centered_max_errors=errs,pixel_change_fraction=fraction))
    groups={}
    for i,r in enumerate(data['pairs']):groups.setdefault(r['source'],[]).append(i)
    indices=[j for g in groups.values() for j in (g[0],g[len(g)//2],g[-1])]
    cal=net.calibrate(as_tensor(read_image(data['pairs'][i]['current']))[None].cuda() for i in indices)
    for r in data['validation']:Image.fromarray(to_rgb(display(net(as_tensor(read_image(r['source']))[None].cuda()))[0])).save(out/(r['id']+'.png'))
    binding=copy.deepcopy(state['binding']);binding['code'][str(Path(__file__))]=file_hash(__file__)
    torch.save(dict(model=net.state_dict(),scales=net.scales,calibration=cal,mode='equalized_ptq',step=state['step'],binding=binding,config=state['config'],source=str(a.checkpoint),source_sha256=file_hash(a.checkpoint)),out/'equalized.pt')
    write_json(out/'runtime_parameters.json',net.integer_package());write_json(out/'report.json',dict(fp32_checks=checks,calibration=cal,factors={k:v.tolist() for k,v in factors.items()},head_common_mode=coef.tolist(),source=str(a.checkpoint),source_sha256=file_hash(a.checkpoint),calibration_train_frames=len(data['pairs']),style_accepted=False))
    print('NATIVE_EQUALIZE_DONE',cal['head_fullrange_scale'],cal['head_top2_scale'],flush=True)


if __name__=='__main__':main()
