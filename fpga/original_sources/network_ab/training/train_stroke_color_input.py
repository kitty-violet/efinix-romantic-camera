"""Color-preserving entrance experiment; keep the first graph/run immutable."""
import json
from pathlib import Path
import shutil
import sys
import torch
from . import train_stroke_color as base
from .stroke_color_input_net import StrokeColorInputNet
from .runtime import file_hash,write_json
from .train_b1_distillation import as_tensor,read_image


class Training(base.Training):
    def __init__(self,out):
        super().__init__(out)
        heads=[];backbone=[]
        for name,p in self.net.named_parameters():
            (heads if name.startswith(('color_context.','ink_features.','render.','rgb.','color_input.')) else backbone).append(p)
        self.opt=torch.optim.Adam([dict(params=backbone,lr=.000225,mult=.15),dict(params=heads,lr=.0015,mult=1.)])


@torch.no_grad()
def initialize(out):
    cfg=json.loads((out/'config.json').read_text(encoding='utf-8'))
    source=base.ROOT/'native-stroke-color-v1/checkpoint_000800.pt'
    net=StrokeColorInputNet().cuda().eval();s=torch.load(source,map_location='cpu',weights_only=True)
    keys=net.load_state_dict(s['model'],strict=False)
    assert keys.missing_keys==['color_input.weight','color_input.bias'] and not keys.unexpected_keys
    data=json.loads((out/'data.json').read_text(encoding='utf-8'));groups={}
    for i,r in enumerate(data['pairs']):groups.setdefault(r['source'],[]).append(i)
    gram=torch.zeros(9,9,dtype=torch.float64,device='cuda');cross=torch.zeros(9,3,dtype=torch.float64,device='cuda');records=[]
    # Fit only the last linear head on training crops for a sensible color start.
    captured=[];hook=net.render.register_forward_hook(lambda m,x,y:captured.append(y))
    for group in groups.values():
        for i in (group[0],group[len(group)//2],group[-1]):
            r=data['pairs'][i]
            for xx,yy in [(128,128),(640,384)]:
                x=as_tensor(read_image(r['current'])[yy:yy+256,xx:xx+256])[None].cuda()
                t=as_tensor(read_image(r['current_target'])[yy:yy+256,xx:xx+256])[None].cuda()
                net(x);f=captured.pop()[0,:,24:-24:4,24:-24:4].flatten(1).T.double()
                f=torch.cat([f,torch.ones_like(f[:,:1])],1);z=t[0,:,24:-24:4,24:-24:4].flatten(1).T.double()
                gram+=f.T@f;cross+=f.T@z;records.append(dict(pair=i,x=xx,y=yy,pixels=len(f)))
    hook.remove();coef=torch.linalg.solve(gram+torch.eye(9,device='cuda',dtype=torch.float64)*.01,cross).float()
    net.rgb.weight.copy_(coef[:8].T[:,:,None,None]);net.rgb.bias.copy_(coef[8])
    path=out/'color_initial.pt';torch.save(dict(model=net.cpu().state_dict(),source=str(source),source_sha256=file_hash(source),head_fit_training_only=records),path)
    cfg.update(initial=str(path),initial_sha256=file_hash(path),graph='native-stroke-color-input-v2',
        renderer='native contour + quarter color context + learned RGB1x1 color entrance -> DW/PW renderer -> RGB; no direct output bypass')
    write_json(out/'config.json',cfg);b=json.loads((out/'binding.json').read_text())
    for p in [Path(__file__),Path(__file__).with_name('stroke_color_input_net.py')]:
        b['code'][str(p)]=file_hash(p);shutil.copy2(p,out/'source'/p.name)
    b['config']=file_hash(out/'config.json');write_json(out/'binding.json',b)
    write_json(out/'initialization.json',dict(method='final linear RGB ridge regression on 42 training crops; no validation fitted',training_crops=records,
        previous=str(source),previous_sha256=file_hash(source),new_color_entrance=True))
    print('COLOR_INPUT_INITIALIZED',len(records),flush=True)


def main():
    base.StrokeColorNet=StrokeColorInputNet;base.Training=Training
    if '--name' not in sys.argv:sys.argv.extend(['--name','native-stroke-color-input-v2'])
    if sys.argv[1]=='prepare':
        base.main();initialize(base.ROOT/sys.argv[sys.argv.index('--name')+1]);return
    base.main()


if __name__=='__main__':main()
