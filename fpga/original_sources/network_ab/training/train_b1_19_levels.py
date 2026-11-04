"""User-requested increase from thirteen to nineteen learned luminance levels."""
import argparse
import copy
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_sampling_colour_v2 as base
from . import train_b1_v2_temporal_margin as stable
from . import train_b1_learned_colour as c
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible

OUT=base.OUT/'nineteen-levels'
PARENT=stable.OUT/'checkpoint_001200.pt'
INCREMENTS=(16,16,16,16,20,24)
BASES=(8,24,40,56,72,92)
LEVELS=[8.]
for lo,inc in zip(BASES,INCREMENTS):LEVELS.extend([lo+inc*j/3 for j in (1,2,3)])


@torch.no_grad()
def quantize(net):
    if net.quantized:return copy.deepcopy(net)
    q=copy.deepcopy(net)
    q.weight['up2_high_dw'][:6]*=3/127;q.weight['up2_high_dw'][6:]/=2
    q.weight['rgb_head'][:,:6]*=127/3;q.weight['rgb_head'][:,6:]*=2
    q.calibrate();q.up2_high_dw_m[:,:6].fill_(32);q.up2_high_dw_m[:,6:].fill_(1024)
    # A 2/15 scale represents 32/3, 40/3, 16 and chroma coefficient4 with
    # integer weights 80,100,120,30. The Q17 approximation is tested separately.
    q.rgb_head_m.fill_(17476);q.quantized=True
    return q


class Trainer(stable.Trainer):
    max_region_state=3
    def __init__(self,cache,selected,steps,qat=False):
        super().__init__(cache,selected,steps)
        self.net.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=True)['model'])
        with torch.no_grad():
            self.net.weight['up2_high_dw'][:6]*=127/2;self.net.weight['up2_high_dw'][6:]*=2
            self.net.weight['rgb_head'][:,:6]/=127/2;self.net.weight['rgb_head'][:,6:]/=2
        self.net.quantized=False
        if qat:self.net=quantize(self.net)
        self.new_optimizer()

    def make_targets(self,t):
        v=(t+1)*64;r,g,b=v[:,0:1],v[:,1:2],v[:,2:3]
        y=(r+2*g+b)/4;out=[]
        for lo,inc in zip(BASES,INCREMENTS):
            planes=torch.cat([(y>lo+inc*(j+.5)/3).float() for j in range(3)],1)
            planes=(F.avg_pool2d(F.pad(planes,(1,1,1,1),mode='replicate'),3,1)>=.5).float()
            out.append(planes.sum(1,keepdim=True)*(127/3))
        return torch.cat(out+[(r-b)/2+64,(-r+2*g-b)/4+64],1)

    def get_features(self,nodes):
        f=base.features(self.net,nodes)
        if self.net.quantized:f=torch.cat((f[:,:6]*(2/3),f[:,6:]),1)
        return f

    def region_logits(self,f):
        return torch.stack([(f[:,:6]-(j+.5)*127/3)/6 for j in range(3)],2).flatten(1,2)

    def region_truth(self,t):
        return torch.stack([(t[:,:6]>(j+.5)*127/3).float() for j in range(3)],2).flatten(1,2)

    def lattice_loss(self,nodes,t):
        f=nodes['up2_low_pw'][:,:,20:-20,20:-20];label=t[:,:,::2,::2][:,:,20:-20,20:-20]
        return .5*F.binary_cross_entropy_with_logits(self.region_logits(f),self.region_truth(label))+.5*self.colour_loss(f,label)


@torch.inference_mode()
def review(tr,data,selected,dest):
    saved=tr.net;tr.net=quantize(saved);c.review(tr,data,selected,dest,integer=True,oracle_comparison=False)
    board=Image.new('RGB',(1680,1248),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    old=c.IntegerB1(c.runtime_from_image((base.PARENT/'step_001800/coefficients-padded.bin').read_bytes()));records=[]
    for i,(key,box,label) in enumerate(c.CROPS):
        row=next(v for v in tr.cached if v['row']['id']==key);x=row['source'][None];nodes=tr.net(x,all_nodes=True)
        gate=nodes['up2_high_dw'][:,:6];assert gate.min()>=0 and gate.max()<=3
        choices=[('冻结教师',row['target']),('V1兜底（7层）',old(x*128)[0]/128),
                 ('实际小网络（19层）',nodes['rgb_head'][0]/128),('原片',x[0])]
        for j,(name,z) in enumerate(choices):
            board.paste(c.rgb_image(z).crop(box),(420*j,312*i+32));draw.text((420*j+3,312*i+4),name,font=font,fill='white')
        records.append(dict(id=key,class_accuracy=float((gate==tr.targets[key][:,:6].mul(3/127).round()).float().mean()),
            occupancy=[float((gate==k).float().mean()) for k in range(4)]))
    board.save(dest/'v1_v2_teacher.png')
    write_json(dest/'regions.json',dict(luminance_levels_8bit=[2*v for v in LEVELS],records=records,visual_accepted=False))
    # Exhaustive ordered luminance states x deployed chroma grid, not a few random colours.
    runtime=c.runtime_from_image(tr.net.export());p=runtime['parameters']['rgb_head']
    rows=[]
    for value in range(19):
        for co in range(65):
            for cg in range(65):rows.append([min(3,max(0,value-3*k)) for k in range(6)]+[co,cg])
    x=np.array(rows,np.int64);wi=np.array(p['weights'],np.int64);bias=np.array(p['bias'],np.int64);m=np.array(p['multipliers'],np.int64)
    accum=(x@wi.T+bias)*m
    rounded=np.sign(accum)*((np.abs(accum)+65536)//131072);actual=rounded.clip(-128,127)
    idealw=tr.net.weight['rgb_head'].detach().cpu().numpy()[:,:,0,0].astype(np.float64)
    idealb=tr.net.bias['rgb_head'].detach().cpu().numpy().astype(np.float64)
    ideal=x@idealw.T+idealb;expected=(np.sign(ideal)*np.floor(np.abs(ideal)+.5)).clip(-128,127)
    error=int(np.abs(actual-expected).max());assert error<=1
    delta_w=wi*m[:,None]/131072-idealw;delta_b=bias*m/131072-idealb
    bound=float((np.abs(delta_w)@np.array([3]*6+[64,64])+np.abs(delta_b)).max());assert bound<.05
    write_json(dest/'colour_head_proof.json',dict(cases=len(rows),maximum_output_error=error,scale_q17=17476,
        mean_signed_error=(actual-expected).mean(0).tolist(),all_state_continuous_error_bound=bound,
        explanation='Q17 approximation of 2/15 for thirds; independent integer rounding and clipping.'))
    tr.net=saved


def preflight(cache,selected,steps,path,qat=False,state=None):
    tr=Trainer(cache,selected,steps,qat)
    if state is not None:tr.net.load_state_dict(state)
    native=tr.update(True);crop=tr.update();tr.save(path/'resume_probe.pt');ref=tr.update();expected={k:v.detach().clone() for k,v in tr.net.state_dict().items()}
    fresh=Trainer(cache,selected,steps,qat);fresh.restore(path/'resume_probe.pt');got=fresh.update()
    error=max(float((v-expected[k]).abs().max()) for k,v in fresh.net.state_dict().items())
    assert error<1e-5 and ref['crop']==got['crop'] and ref['samples']==got['samples']
    write_json(path/'preflight.json',dict(status='PASS',native=native,pair_crop=crop,resume_error=error,quantized=qat));print('LEVEL19_PREFLIGHT_PASS',qat,error,flush=True)
    del tr,fresh;torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--qat',type=int,default=1000);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100531);contract_guard();OUT.mkdir(exist_ok=False);data,selected,cache=c.load_data()
    write_json(OUT/'binding.json',dict(parent=str(PARENT),parent_sha256=file_hash(PARENT),parent_cumulative_steps=11700,
        new_fp32_steps=a.steps,new_qat_steps=a.qat,requested_change='User asks more colour levels again: 13 to19',
        actual_B1=True,external_inference=False,levels=[2*v for v in LEVELS],
        codes={str(p):file_hash(p) for p in [Path(__file__),Path(base.__file__),Path(stable.__file__)]}))
    for p in (__file__,base.__file__,stable.__file__):shutil.copy2(p,OUT/Path(p).name)
    preflight(cache,selected,a.steps,OUT);tr=Trainer(cache,selected,a.steps)
    with (OUT/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        while tr.step<a.steps:
            row=tr.update();log.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('LEVEL19_FP32',tr.step,round(row['loss'],4),flush=True)
        tr.save(OUT/f'checkpoint_{tr.step:06d}.pt');review(tr,data,selected,OUT/f'step_{tr.step:06d}')
        q=quantize(tr.net);state=copy.deepcopy(q.state_dict());del tr,q;torch.cuda.empty_cache();qdir=OUT/'qat';qdir.mkdir()
        preflight(cache,selected,a.qat,qdir,True,state);tr=Trainer(cache,selected,a.qat,True);tr.net.load_state_dict(state)
        while tr.step<a.qat:
            row=tr.update();log.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('LEVEL19_QAT',tr.step,round(row['loss'],4),flush=True)
        tr.save(qdir/f'checkpoint_{tr.step:06d}.pt');review(tr,data,selected,qdir/'package')
    print('LEVEL19_COMPLETE',flush=True)


if __name__=='__main__':main()
