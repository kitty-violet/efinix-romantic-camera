"""Use all three existing residual stages for contour context inside B1.

The unchanged decoder layers implement the contour-constrained colour update.
This moves the contour readout from support41 to support57 without new nodes.
"""
import json
import math
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from . import train_b1_structural_gate as gate
from .train_b1_deep_contour import DeepStudent,DeepTraining,OUT as LABELS
from . import resume_b1_contour_precision as general
from .probe_b1_crop_quantization import calibrate_parameter_precision
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible

OUT=LABELS.parent/'b1-late-contour-context-20261005-v1'
PARENT=LABELS.parent/'b1-feature-flow-repair-20261005-v1/reconnected_001000.pt'


class LateStudent(DeepStudent):
    first_gate_node='up1_low_dw'
    rectified_gate_node='up1_low_dw'

    def __init__(self):
        super().__init__()
        state=torch.load(PARENT,map_location='cpu',weights_only=True)
        self.load_state_dict(state['model']);self.package=state['runtime']
        with torch.no_grad():
            for suffix in ('conv1__dw','conv1__pw','conv2__dw','conv2__pw'):
                dst,src='res3_'+suffix,'res2_'+suffix
                self.weight[dst].copy_(self.weight[src]);self.bias[dst].copy_(self.bias[src])
                getattr(self,dst+'_m').copy_(getattr(self,src+'_m'))
            # New middle residual starts as an exact integer identity. Its
            # feature transform is trainable, with no old gate skip remaining.
            self.weight['res2_conv2__pw'].zero_();self.bias['res2_conv2__pw'].zero_()
            self.weight['res3_conv2__pw'][26:].zero_();self.bias['res3_conv2__pw'][26:].zero_()
            for kind,parameters in [('w',self.weight),('b',self.bias)]:
                for key,value in parameters.items():
                    if not key.startswith(('res2_','res3_')):continue
                    value.requires_grad_(True);mask=torch.zeros_like(value)
                    if key=='res3_conv2__pw':
                        if kind=='w':mask[24:26,6:24]=1
                        else:mask[24:26]=1
                    elif kind=='w' and key.endswith('__pw'):mask[6:24,6:24]=1
                    else:mask[6:24]=1
                    for suffix,tensor in [('mask',mask),('initial',value.detach().clone())]:
                        name=kind+'_'+key+'_'+suffix
                        if hasattr(self,name):getattr(self,name).copy_(tensor)
                        else:self.register_buffer(name,tensor)
                    self.masks[(kind,key)]=True
            self._decoder()
            for key in self.weight:
                self.package['parameters'][key.replace('__','.')]['multipliers']=getattr(self,key+'_m').flatten().long().tolist()
        self.quantized=False

    def _decoder(self):
        def clear(key):
            w,b,m=self.weight[key],self.bias[key],getattr(self,key+'_m')
            w.zero_();b.zero_();m.fill_(65536)
            w.requires_grad_(False);b.requires_grad_(False)
            return w,b,m
        ident=torch.tensor([[0,0,0],[0,2,0],[0,0,0]],dtype=torch.float32)
        gauss=torch.tensor([[1,2,1],[2,4,2],[1,2,1]],dtype=torch.float32)
        w,b,m=clear('up1_low_dw')
        for c in (0,1,2,24):w[c,0]=ident
        for c in (3,4,5):w[c,0]=gauss;m[0,c]=8192
        # ReLU has already made E>=0. The two following positive differences
        # preserve a boundary and replace unprotected interior detail by mean.
        w,b,m=clear('up1_low_pw')
        for c in range(3):
            w[c,c]=2
            w[3+c,c]=2;w[3+c,3+c]=-2;w[3+c,24]=-2
            w[6+c,c]=-2;w[6+c,3+c]=2;w[6+c,24]=-2
        for key in ('up1_high_dw','up2_low_dw','up2_high_dw'):
            w,b,m=clear(key)
            for c in range(len(w)):w[c,0]=ident
        w,b,m=clear('up2_low_pw');levels=[6,3,3];phases=[0,2,2]
        for c in range(3):
            w[c,c]=1;w[c,3+c]=-1;w[c,6+c]=1;b[c]=phases[c]
            m[0,c]=round(131072/levels[c])

    def project(self):
        # Only the supervised master gate is optimized; copy its deployment
        # replica after updating, without averaging in an unused gradient.
        with torch.no_grad():
            for kind,key in self.masks:
                value=(self.weight if kind=='w' else self.bias)[key]
                mask=getattr(self,kind+'_'+key+'_mask');base=getattr(self,kind+'_'+key+'_initial')
                value.copy_(value*mask+base*(1-mask))
                if kind=='w':value.clamp_(-127,127)
            for parameters in (self.weight,self.bias):
                p=parameters['res3_conv2__pw'];p[25].copy_(p[24])

    def forward(self,x,all_nodes=False,full=False):
        nodes=gate.Student.forward(self,x,True,(full or all_nodes),stop_node='res3_add')
        return nodes if all_nodes else nodes['res3_add'][:,24:25]


class Trainer(DeepTraining):
    def __init__(self,data):
        super().__init__(data);self.net=LateStudent().cuda().eval()
        self.opt=torch.optim.Adam([p for p in self.net.parameters() if p.requires_grad],lr=.025)


def main():
    configure_workspace();set_reproducible(2026100515);contract_guard();OUT.mkdir(exist_ok=False);general.OUT=OUT
    data=json.loads((LABELS/'data.json').read_text(encoding='utf-8'))
    for r in data:
        for key in ('source','target','label'):assert file_hash(r[key])==r[key+'_sha256']
    for name in ('train_b1_late_contour.py','train_b1_structural_gate.py','train_b1_structural_features.py','train_b1_deep_contour.py','resume_b1_contour_precision.py'):
        shutil.copy2(Path(__file__).with_name(name),OUT/name)
    write_json(OUT/'binding.json',dict(parent=str(PARENT),parent_sha256=file_hash(PARENT),
        readout='res3_add instead of res2_add',contour_rf=57,previous_rf=41,
        colour='Same graph decoder layers now consume the later gate; explicit coefficients, no postprocessing',
        fp32_steps=1600,qat_steps=600,data_sha256=file_hash(LABELS/'data.json')))
    tr=Trainer(data);original=DeepStudent().cuda().eval();state=torch.load(PARENT,map_location='cpu',weights_only=True)
    original.load_state_dict(state['model']);original.package=state['runtime']
    r=next(r for r in data if r['split']=='train');rgb=np.array(Image.open(r['source']))[160:480,320:640]
    x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),device='cuda',dtype=torch.float32)/128-1
    with torch.no_grad():
        original.quantized=tr.net.quantized=True
        integer_difference=float((original(x)-tr.net(x)).abs().max());assert integer_difference==0
        original.quantized=tr.net.quantized=False
        float_difference=float((original(x)-tr.net(x)).abs().max());assert float_difference<.1
    tr.net.zero_grad(set_to_none=True);short=tr.net(x);short.square().mean().backward()
    grads={k:v.grad.clone() for k,v in tr.net.named_parameters() if v.grad is not None}
    tr.net.zero_grad(set_to_none=True);full=tr.net(x,True,True)['res3_add'][:,24:25];full.square().mean().backward()
    ge=max(float((p.grad-grads[k]).abs().max()) for k,p in tr.net.named_parameters() if k in grads)
    assert ge==0 and torch.equal(short,full)
    native=tr.update(True);crop=tr.update();general.save(tr,OUT/'resume_probe.pt');ref=tr.update()
    expected={k:v.clone() for k,v in tr.net.state_dict().items()};fresh=Trainer(data);general.restore(fresh,OUT/'resume_probe.pt');got=fresh.update()
    resume=max(float((v-expected[k]).abs().max()) for k,v in fresh.net.state_dict().items());assert resume==0 and ref['crop']==got['crop']
    write_json(OUT/'preflight.json',dict(initial_gate_integer_difference=integer_difference,
        initial_gate_fp32_difference=float_difference,full_path_gradient_error=ge,disk_resume_error=resume,native=native,crop=crop))
    print('LATE_CONTOUR_PREFLIGHT_PASS',integer_difference,float_difference,ge,resume,flush=True)
    del original,tr,fresh;torch.cuda.empty_cache();tr=Trainer(data)
    general.review(tr,data,'initial_integer',True)
    with (OUT/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as f:
        while tr.step<2200:
            if tr.step==1600:
                write_json(OUT/'calibration.json',calibrate_parameter_precision(tr.net));tr.net.quantized=True
                tr.opt=torch.optim.Adam([p for p in tr.net.parameters() if p.requires_grad],lr=.012)
            lr=.025-.01*tr.step/1600 if tr.step<1600 else .002+.01*.5*(1+math.cos(math.pi*(tr.step-1600)/600))
            for group in tr.opt.param_groups:group['lr']=lr
            row=tr.update((tr.step+1)%100==0);f.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('LATE_CONTOUR',tr.step,round(row['loss'],5),flush=True)
            if tr.step in (800,1600,1900,2200):
                general.save(tr,OUT/f'checkpoint_{tr.step:06d}.pt');general.review(tr,data,f'step_{tr.step:06d}',tr.net.quantized)
    print('LATE_CONTOUR_COMPLETE',flush=True)


if __name__=='__main__':main()
