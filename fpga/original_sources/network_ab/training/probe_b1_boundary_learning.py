"""Bounded training-only capacity diagnostic for the current integer B1.

Two fixed crops from train sources, never validation, separate plain boundary
learning from the full competing objective. These fitted outputs are not a
generalized model or deliverable. No graph or inference operation changes.
"""
import argparse
import copy
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont
from . import train_b1_minimum_area as m
from .train_b1_balanced_region_boundaries import BalancedBoundaryTraining
from .train_b1_independent_region_boundaries import distances
from .fit_b1_sixteen_regions import region_class_map
from .train_b1_sixteen_regions import colour_target
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .engine import set_reproducible


class FixedTraining(BalancedBoundaryTraining):
    def prepare(self, data):
        self.fixed_data=[]
        for suffix,xy in [('121509',(768,288)),('121831',(416,224))]:
            row=next(r for r in data['training'] if Path(r['video']).stem.endswith(suffix))
            assert row['split']=='train'
            x,y=xy;side=320
            code=region_class_map(row['target'],row['region'],self.pk)[y:y+side,x:x+side]
            label=m.load_region(row['region'])[y:y+side,x:x+side]
            source=m.t.r.tensor(m.t.r.read(row['path'])[y:y+side,x:x+side])
            self.fixed_data.append((source,torch.from_numpy(code.astype(np.int64)).cuda(),
                torch.from_numpy(label.astype(np.int64)).cuda(),dict(id=row['id'],xy=list(xy),wh=[side,side],flip=0)))
        self.x=torch.stack([v[0] for v in self.fixed_data])
        self.fixed_codes=torch.stack([v[1] for v in self.fixed_data])
        self.fixed_labels=torch.stack([v[2] for v in self.fixed_data])
        self.target=colour_target(self.fixed_codes,self.palette)
        bits=self.bit_table[self.fixed_codes].permute(0,3,1,2)
        self.wanted=F.avg_pool2d(bits,2)[...,20:-20,20:-20]
        self.distance_target=distances(bits[:,:,::4,::4])[...,10:-10,10:-10]

    def batch_style(self,native):
        assert not native
        self.class_labels=self.fixed_codes;self.labels=self.fixed_labels
        return self.x,self.target,[v[3] for v in self.fixed_data],None,None

    def input_stability_loss(self,x,pred,native):
        if self.args.diagnostic_objective=='plain':return pred.new_zeros(()),{}
        return super().input_stability_loss(x,pred,native)

    def additional_loss(self,p,target,native):
        if self.args.diagnostic_objective=='full':return super().additional_loss(p,target,native)
        key='up1_low_pw';w,b=self.net.weight[key],self.net.bias[key]
        s=(getattr(self.net,key+'_m')/131072).float().flatten()
        qw=self.net.round(w/s[:,None,None,None]).clamp(-127,127)*s[:,None,None,None]
        qb=self.net.round(b/s)*s
        raw=(F.conv2d(self.nodes['up1_low_dw'],qw,qb)[:,:15]-63.5)/8
        logits=(self.nodes['up1_high_dw'][:,:15]-63.5)/4
        cls=self.classification_loss(logits[...,20:-20,20:-20],self.wanted)
        distance=self.distance_loss(raw[...,10:-10,10:-10],self.distance_target)
        return cls+distance,dict(plain_classification=cls,plain_distance=distance)


@torch.inference_mode()
def report(tr,dest,step,input_title='训练原片',output_title='实际 B1：拟合诊断'):
    nodes=tr.net(tr.x,all_nodes=True);p=nodes['rgb_head']/128
    logits=(nodes['up1_high_dw'][:,:15,...,20:-20,20:-20]-63.5)/4
    interior=((tr.wanted<.01)|(tr.wanted>.99))
    correct=((logits>0)==(tr.wanted>.5))&interior
    actual_codes=torch.stack([tr.predicted_codes(v) for v in logits])
    wanted_codes=torch.stack([tr.predicted_codes(v-.5) for v in tr.wanted])
    pure_colour=interior.all(1)
    values=dict(step=step,mae=float((p[...,40:-40,40:-40]-tr.target[...,40:-40,40:-40]).abs().mean()*128),
        interior_bit_accuracy=float(correct.sum()/interior.sum()),
        interior_joint_colour_accuracy=float(((actual_codes==wanted_codes)&pure_colour).sum()/pure_colour.sum().clamp_min(1)),
        interior_joint_bit_accuracy=float((correct.all(1)&pure_colour).sum()/pure_colour.sum().clamp_min(1)),
        bit_accuracy=[float(correct[:,i].sum()/interior[:,i].sum().clamp_min(1)) for i in range(15)])
    board=Image.new('RGB',(720,278*len(p)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    for i in range(len(p)):
        for j,(z,title) in enumerate([(tr.x[i],input_title),(tr.target[i],'固定区域目标'),(p[i],output_title)]):
            board.paste(m.t.r.c.rgb_image(z[...,40:-40,40:-40]),(240*j,278*i+38))
            draw.text((240*j+4,278*i+7),title,font=font,fill='white')
    board.save(dest/f'comparison_{step:06d}.png')
    write_json(dest/f'metrics_{step:06d}.json',values)
    print('BOUNDARY_CAPACITY',tr.args.diagnostic_objective,values,flush=True)
    return values


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--steps',type=int,default=700);ap.add_argument('--objective',choices=['plain','full'],required=True)
    a=ap.parse_args();configure_workspace();contract_guard();set_reproducible(20261006111);torch.set_num_threads(2)
    state=torch.load(a.warm,map_location='cpu',weights_only=True)
    args=argparse.Namespace(**state['args']);args.warm=a.warm;args.steps=a.steps;args.lr=.001
    args.diagnostic_objective=a.objective;args.shift=0 if a.objective=='plain' else .3;args.temporal=0
    args.training_class='FixedTraining';args.resume_from=''
    dest=m.OUT/a.name;dest.mkdir(exist_ok=False);data=m.load_data()
    tr=FixedTraining(args,data);tr.prepare(data)
    write_json(dest/'config.json',dict(args=vars(args),fitted_training_crops=[v[3] for v in tr.fixed_data],
        source_sha256=file_hash(Path(__file__)),parent_sha256=file_hash(a.warm),
        capacity_diagnostic_only=True,heldout_not_trained=True,hardware_changed=False))
    shutil.copy2(__file__,dest/Path(__file__).name)
    # Verify the same new diagnostic objective resumes the next actual update.
    tr.update(False);tr.save(dest/'preflight.pt');expected=tr.update(False)
    weights=copy.deepcopy(tr.net.state_dict());other=FixedTraining(args,data);other.prepare(data);other.restore(dest/'preflight.pt')
    actual=other.update(False);error=max(float((v-weights[k]).abs().max()) for k,v in other.net.state_dict().items())
    assert error<1e-5 and expected['samples']==actual['samples'];write_json(dest/'preflight.json',dict(resume_error=error,status='PASS'))
    del tr,other;torch.cuda.empty_cache();tr=FixedTraining(args,data);tr.prepare(data)
    start=time.perf_counter();history=[report(tr,dest,0)]
    with (dest/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        for _ in range(a.steps):
            row=tr.update(False);log.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('BOUNDARY_CAPACITY_STEP',a.objective,tr.step,row['loss'],flush=True)
            if tr.step%350==0 or tr.step==a.steps:
                history.append(report(tr,dest,tr.step));tr.save(dest/f'candidate_{tr.step:06d}.pt')
    write_json(dest/'completion.json',dict(history=history,seconds=time.perf_counter()-start,visual_acceptance=False,
        diagnostic_only=True,no_generalization_claim=True))


if __name__=='__main__':main()
