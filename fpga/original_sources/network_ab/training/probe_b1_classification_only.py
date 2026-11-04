"""Fixed TRAIN crops, classification only: isolate learning from loss conflict.

The older 'plain' diagnostic still inherited a pixelwise RGB appearance loss.
This experiment explicitly zeros appearance, temporal, input and SDF losses.
Target-as-input is a decoder/representation diagnostic, never a style result.
"""
import argparse
import copy
import json
import shutil
import time
from pathlib import Path
import torch
from .probe_b1_boundary_learning import FixedTraining,report
from .train_b1_count_regions import count_targets,count_loss
from . import train_b1_minimum_area as m
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


class ClassificationOnlyTraining(FixedTraining):
    @staticmethod
    def appearance_loss(p,t,mode):return p.sum()*0,{}

    def prepare(self,data):
        super().prepare(data)
        if self.args.diagnostic_input=='target':self.x=self.target.detach().clone()
        self.count_wanted=[v[...,20:-20,20:-20] for v in count_targets(self.fixed_codes)]

    def additional_loss(self,p,target,native):
        logits=(self.nodes['up1_high_dw'][:,:15,20:-20,20:-20]-63.5)/4
        cls=count_loss(logits,self.count_wanted) if self.args.diagnostic_coding=='counts' else self.classification_loss(logits,self.wanted)
        return cls,dict(classification_only=cls)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--steps',type=int,default=3000);ap.add_argument('--input',choices=['source','target'],default='source')
    ap.add_argument('--coding',choices=['bits','counts'],default='bits')
    ap.add_argument('--fp32',action='store_true');a=ap.parse_args()
    configure_workspace();contract_guard();set_reproducible(20261006111);torch.set_num_threads(2)
    parent=torch.load(a.warm,map_location='cpu',weights_only=True);args=argparse.Namespace(**parent['args'])
    args.warm=a.warm;args.steps=a.steps;args.lr=.001;args.shift=0;args.temporal=0
    args.diagnostic_objective='plain';args.diagnostic_input=a.input;args.qat=not a.fp32
    args.diagnostic_coding=a.coding
    args.training_class='ClassificationOnlyTraining';args.resume_from=''
    dest=m.OUT/a.name;dest.mkdir(exist_ok=False);data=m.load_data()
    tr=ClassificationOnlyTraining(args,data);tr.prepare(data)
    write_json(dest/'config.json',dict(args=vars(args),training_crops=[v[3] for v in tr.fixed_data],
        only_loss='balanced count NLL' if a.coding=='counts' else 'balanced bit BCE',no_rgb_loss=True,no_distance_loss=True,no_stability_loss=True,
        source_sha256=file_hash(__file__),warm_sha256=file_hash(a.warm),
        qat_forward=args.qat,diagnostic_only=True,validation_never_fitted=True,teacher_input=a.input=='target'))
    shutil.copy2(__file__,dest/Path(__file__).name)
    for name in ('probe_b1_boundary_learning.py','train_b1_balanced_region_boundaries.py','train_b1_count_regions.py'):
        shutil.copy2(Path(__file__).with_name(name),dest/name)
    tr.update(False);tr.save(dest/'preflight.pt');expected=tr.update(False);weights=copy.deepcopy(tr.net.state_dict())
    other=ClassificationOnlyTraining(args,data);other.prepare(data);other.restore(dest/'preflight.pt');actual=other.update(False)
    error=max(float((v-weights[k]).abs().max()) for k,v in other.net.state_dict().items())
    assert error<1e-5 and actual['samples']==expected['samples']
    assert set(expected['values'])=={'classification_only'}
    write_json(dest/'preflight.json',dict(status='PASS',resume_error=error,loss_terms=list(expected['values'])))
    print('CLASSIFICATION_ONLY_PREFLIGHT',error,flush=True)
    del tr,other;torch.cuda.empty_cache();tr=ClassificationOnlyTraining(args,data);tr.prepare(data)
    def review(step):
        result=report(tr,dest,step,input_title='目标作为输入（诊断）' if a.input=='target' else '训练原片',
            output_title='FP32拟合诊断' if a.fp32 else '真实整数B1：拟合诊断')
        if a.fp32:
            # The same FP32-learned coefficients, switched to unchanged integer
            # arithmetic. No recalibration/retraining is hidden in this control.
            folder=dest/'integer-replay';folder.mkdir(exist_ok=True)
            tr.net.quantized=True
            try:result['same_weights_integer_replay']=report(tr,folder,step,
                input_title='固定训练输入',output_title='同权重整数计算·诊断')
            finally:tr.net.quantized=False
        return result
    history=[review(0)];start=time.perf_counter()
    with (dest/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        while tr.step<a.steps:
            row=tr.update(False);log.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('CLASSIFICATION_ONLY_STEP',tr.step,row['loss'],flush=True)
            if tr.step%500==0 or tr.step==a.steps:
                history.append(review(tr.step));tr.save(dest/f'candidate_{tr.step:06d}.pt')
                if (dest/'STOP_REQUEST.json').exists():break
    write_json(dest/'completion.json',dict(executed_steps=tr.step,history=history,seconds=time.perf_counter()-start,
        diagnostic_only=True,visual_acceptance=False,no_generalization_claim=True,teacher_input=a.input=='target'))


if __name__=='__main__':main()
