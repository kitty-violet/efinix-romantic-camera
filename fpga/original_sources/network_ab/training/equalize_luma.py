"""Positive diagonal feature balancing and class-common logit centering.

Both transformations preserve the FP32 classifier (not bit exact floats).
All scaling is folded into existing weights. Calibration uses TRAIN sources.
"""
import argparse,copy,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .train_luma_regions import ROOT,RUN,LumaNet,SLICES,display
from .train_b1_distillation import as_tensor,read_image,to_rgb
from .runtime import configure_workspace,file_hash,write_json
from .engine import set_reproducible

class EqualizedNet(LumaNet):
    @torch.no_grad()
    def calibrate(self,inputs):
        inputs=list(inputs);report=super().calibrate(inputs);old=self.scales['rgb'];saved=self.scales;self.scales=None;peak=0
        for x in inputs:
            p=self(x)
            for s in SLICES:
                top=p[:,s].topk(2,dim=1).values;peak=max(peak,float(top.abs().max()))
        self.scales=saved;self.scales['rgb']=max(peak*1.12/127,1/512)
        report.update(scales=self.scales,head_fullrange_scale=old,head_top2_peak=peak,head_top2_scale=self.scales['rgb'],
            note='Very negative losing logits may clip; selected/top2 logits retain calibrated range.')
        return report

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='equalized-v2');a=p.parse_args()
    configure_workspace();set_reproducible(2026100428);root=ROOT/'runs'/RUN;out=root/a.name;out.mkdir(exist_ok=False)
    ck=root/'luma/fp32_checkpoint_003000.pt';state=torch.load(ck,map_location='cpu',weights_only=True);net=EqualizedNet().cuda().eval();net.load_state_dict(state['model'])
    original=copy.deepcopy(net);data=json.loads((root/'luma/data.json').read_text(encoding='utf-8'))
    # All 95 TRAIN current frames, stride16 feature sampling for normal equations.
    xx=np.zeros((9,9),np.float64);xy=np.zeros((9,3),np.float64);maxima={}
    for i,r in enumerate(data['pairs']):
        x=as_tensor(read_image(r['current']))[None].cuda();v=net(x,True)
        for k,a in v.items():
            c=a.abs().amax((0,2,3)).cpu().numpy();maxima[k]=np.maximum(maxima.get(k,c),c)
        f=v['finish.dw'][0,:,::16,::16].reshape(8,-1).T.cpu().double().numpy();f=np.concatenate((f,np.ones((len(f),1))),1)
        p=v['rgb'][0,:,::16,::16].reshape(32,-1);top=np.stack([p[s].max(0).values.cpu().double().numpy() for s in SLICES],1)
        xx+=f.T@f;xy+=f.T@top
        if (i+1)%20==0:print('EQUALIZE_CALIB',i+1,flush=True)
    coef=np.linalg.lstsq(xx+np.eye(9)*1e-8,xy,rcond=1e-12)[0];head=net.conv('rgb')
    for i,s in enumerate(SLICES):
        head.weight[s,:,0,0]-=torch.tensor(coef[:8,i],device='cuda',dtype=torch.float32)[None]
        head.bias[s]-=float(coef[8,i])
    # Tie all channel transforms connected by an add or a view.
    parent={n:n for n in maxima}
    def find(x):
        while parent[x]!=x:x=parent[x]
        return x
    def union(a,b):parent[find(a)]=find(b)
    for n in net.nodes:
        if n['kind'] in ('nearest_x2','residual_add'):
            for p in n['inputs']:union(n['id'],p)
    pooled={}
    for k,v in maxima.items():g=find(k);pooled[g]=np.maximum(pooled.get(g,v),v)
    factors={k:np.clip(1/np.maximum(pooled[find(k)],1e-3),.001,100.) for k in maxima}
    factors['input']=np.ones(3);factors['rgb']=np.ones(32)
    for n in net.nodes:
        if not n['kernel']:continue
        conv=net.conv(n['id']);fi=torch.tensor(factors[n['inputs'][0]],device='cuda');fo=torch.tensor(factors[n['id']],device='cuda')
        conv.weight.mul_(fo[:,None,None,None]);conv.bias.mul_(fo)
        conv.weight.div_(fi[:,None,None,None] if n['kind']=='dw3x3' else fi[None,:,None,None])
    comparisons=[]
    for r in data['validation']:
        x=as_tensor(read_image(r['source']))[None].cuda();a=original(x);b=net(x);errs=[]
        for s in SLICES:
            ca=a[:,s]-a[:,s].mean(1,keepdim=True);cb=b[:,s]-b[:,s].mean(1,keepdim=True);errs.append(float((ca-cb).abs().max()))
        before=to_rgb(display(a)[0]);after=to_rgb(display(b)[0]);fraction=float(np.any(before!=after,2).mean())
        assert max(errs)<.0005 and fraction<1e-4,(r['id'],errs,fraction)
        comparisons.append(dict(id=r['id'],common_centered_max_errors=errs,fp32_pixel_change_fraction=fraction))
    groups={}
    for i,r in enumerate(data['pairs']):groups.setdefault(r['source'],[]).append(i)
    indices=[j for g in groups.values() for j in (g[0],g[len(g)//2],g[-1])]
    calibration=net.calibrate(as_tensor(read_image(data['pairs'][i]['current']))[None].cuda() for i in indices)
    for r in data['validation']:
        x=as_tensor(read_image(r['source']))[None].cuda();pred=display(net(x));Image.fromarray(to_rgb(pred[0])).save(out/(r['id']+'.png'))
    # Do not present this reparameterization as new training steps.
    payload=dict(model=net.state_dict(),scales=net.scales,calibration=calibration,step=state['step'],mode='equalized_ptq',
        source=str(ck),source_sha256=file_hash(ck),equalization_sha256=file_hash(__file__),binding=state['binding'],config=state['config'])
    torch.save(payload,out/'equalized.pt');write_json(out/'runtime_parameters.json',net.integer_package())
    write_json(out/'report.json',dict(fp32_checks=comparisons,source=str(ck),source_sha256=file_hash(ck),
        calibration=calibration,factors={k:v.tolist() for k,v in factors.items()},head_common_mode=coef.tolist(),
        calibration_train_frames=95,requires_new_op=False,fit=False,temporal=False))
    print('EQUALIZATION_DONE',calibration['head_fullrange_scale'],calibration['head_top2_scale'],flush=True)

if __name__=='__main__':main()
