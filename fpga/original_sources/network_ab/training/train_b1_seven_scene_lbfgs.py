"""Full-scene FP32 fit with bounded coefficients and a line-search optimizer.

Only seven identified TRAIN frames are fitted. Integer previews, saved plain
weights and the separate resumable optimizer state must never be confused.
"""
import argparse
import copy
import json
import shutil
import time
from pathlib import Path
import torch
from torch.nn.utils import parametrize
from .probe_b1_bounded_lbfgs import BoundedRows,BiasUnits
from .train_b1_seven_scene_batch import SevenSceneBatchTraining
from .train_b1_count_regions import count_loss
from .train_b1_ordered_regions import evaluate,preview
from . import train_b1_minimum_area as m
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


class BoundedSevenFit:
    def __init__(self,args,data):
        self.args=args;self.step=0;self.calls=0
        state=torch.load(args.warm,map_location='cpu',weights_only=True)
        setup=argparse.Namespace(**state['args'])
        setup.warm=args.warm;setup.steps=args.steps;setup.shift=0;setup.temporal=0
        setup.lr_schedule='constant';setup.qat=True;setup.scale_lr=0
        self.training=SevenSceneBatchTraining(setup,data)
        self.net=self.training.net;self.initial=copy.deepcopy(state)
        self.keys=[k for k,w in self.net.weight.items() if w.requires_grad]
        self.original_keys=list(self.net.state_dict())
        initial_blob=self.net.export()
        for key in self.keys:
            limit=127*(getattr(self.net,key+'_m')/131072).float().flatten()[:,None,None,None]
            parametrize.register_parametrization(self.net.weight,key,BoundedRows(limit))
            parametrize.register_parametrization(self.net.bias,key,BiasUnits())
        assert self.net.export()==initial_blob
        self.parameters=[p for p in self.net.parameters() if p.requires_grad]
        self.opt=torch.optim.LBFGS(self.parameters,lr=1,max_iter=3,max_eval=5,history_size=40,
            line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-10)
        self.net.quantized=False

    @torch.no_grad()
    def plain_model(self):
        state=self.net.state_dict();plain={}
        for key in self.original_keys:
            if key.startswith('weight.'):value=self.net.weight[key[7:]]
            elif key.startswith('bias.'):value=self.net.bias[key[5:]]
            else:value=state[key]
            plain[key]=value.detach().clone()
        return plain

    def update(self,crop=False):
        started=time.perf_counter();before=self.calls;values=[]
        def closure():
            self.net.quantized=False;self.opt.zero_grad(set_to_none=True)
            total=0.;values.clear()
            for key,x,truth in self.training.fit_cache:
                if crop:
                    xx=x[...,192:512,384:704];tt=[v[...,116:236,212:332] for v in truth]
                else:xx=x;tt=truth
                nodes=self.net(xx,all_nodes=True)
                logits=(nodes['up1_high_dw'][:,:15]-63.5)/4
                if crop:logits=logits[...,20:-20,20:-20]
                loss=count_loss(logits,tt)/len(self.training.fit_cache)
                loss.backward();total+=loss.detach();values.append(float(loss.detach())*7)
                del nodes,logits,loss
            self.calls+=1
            assert torch.isfinite(total) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.parameters)
            return total
        initial_loss=self.opt.step(closure);self.step+=1;torch.cuda.synchronize()
        self.training.step=self.step
        return dict(step=self.step,loss_before_step=float(initial_loss),last_closure_loss=sum(values)/7,
            closure_calls=self.calls-before,cumulative_closures=self.calls,native720=not crop,
            fit_ids=self.training.fit_ids,seconds=time.perf_counter()-started)

    def save(self,path):
        state=dict(self.initial);setup=dict(vars(self.training.args));setup.update(qat=False,
            training_class='BoundedSevenFit',steps=self.args.steps,warm=self.args.warm)
        state.update(model=self.plain_model(),bounded_training_model=copy.deepcopy(self.net.state_dict()),
            optimizer=self.opt.state_dict(),args=setup,step=self.step,closure_calls=self.calls,
            qat=False,bounded_fp32_lbfgs=True,seven_scene_fit_diagnostic=True,fit_ids=self.training.fit_ids,
            native_coverage_schedule='all seven fixed full720 TRAIN images; FP32 bounded L-BFGS',
            objective='mean of seven class-balanced count cross-entropies',
            group_decoder_training=True,training_only_parameterization=True,
            minimum_area_inference_guarantee=False,visual_acceptance=False,generalization_claim=False,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            temporal_class_manifest_sha256=self.training.class_manifest_sha256)
        torch.save(state,path)

    def restore(self,path):
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert state['bounded_fp32_lbfgs'] and state['fit_ids']==self.training.fit_ids
        assert state['temporal_class_manifest_sha256']==self.training.class_manifest_sha256
        self.net.load_state_dict(state['bounded_training_model']);self.opt.load_state_dict(state['optimizer'])
        self.step=state['step'];self.calls=state['closure_calls'];self.training.step=self.step
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        self.net.quantized=False

    @torch.inference_mode()
    def report(self,data,dest):
        self.net.quantized=True
        try:
            fitted=self.training.report_training_fit();quality=evaluate(self.net,data['validation'])
            preview(self.net,data['validation'],dest/f'integer_preview_{self.step:06d}')
            blob=self.net.export();(dest/f'coefficients_{self.step:06d}.bin').write_bytes(blob)
            # A plain deployment instance must reproduce the training model
            # despite removal of the bounded training parameterization.
            plain=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin').cuda()
            plain.load_state_dict(self.plain_model());plain.quantized=True
            plain.package['residual_encoding']=self.net.package['residual_encoding']
            assert plain.export()==blob
            x=self.training.fit_cache[0][1]
            expected=self.net(x,all_nodes=True);actual=plain(x,all_nodes=True)
            error=max(float((v-actual[k]).abs().max()) for k,v in expected.items());assert error==0
            result=dict(step=self.step,closure_calls=self.calls,training_fit=fitted,validation=quality,
                removed_parameterization_all_node_error=error,bytes=len(blob),
                coefficient_sha256=file_hash(dest/f'coefficients_{self.step:06d}.bin'))
            print('BOUNDED_SEVEN_REPORT',self.step,fitted['mean_accuracy'],quality['mean'],flush=True)
            return result
        finally:self.net.quantized=False


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--steps',type=int,default=150);ap.add_argument('--resume',default='')
    a=ap.parse_args();configure_workspace();contract_guard();set_reproducible(20261006149)
    torch.set_num_threads(2);data=m.load_data();dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    here=Path(__file__).parent
    sources=[Path(__file__),here/'probe_b1_bounded_lbfgs.py',here/'train_b1_seven_scene_batch.py',
        here/'train_b1_seven_scene_fit.py',here/'train_b1_count_regions.py',here/'train_b1_learned_colour.py']
    for path in sources:shutil.copy2(path,dest/path.name)
    write_json(dest/'config.json',dict(args=vars(a),warm_sha256=file_hash(a.warm),
        code={str(p):file_hash(p) for p in sources},data_sha256=file_hash(m.OUT/'data.json'),
        fp32_no_amp_no_tf32=True,integer_preview_only=True,architecture_changed=False))
    tr=BoundedSevenFit(a,data)
    if a.resume:tr.restore(a.resume)
    first=tr.update();crop=tr.update(True);tr.save(dest/'preflight.pt')
    expected=tr.update();reference=copy.deepcopy(tr.net.state_dict())
    other=BoundedSevenFit(a,data);other.restore(dest/'preflight.pt');got=other.update()
    error=max(float((v-reference[k]).abs().max()) for k,v in other.net.state_dict().items())
    assert error<1e-5 and got['closure_calls']==expected['closure_calls']
    write_json(dest/'preflight.json',dict(status='PASS',native=first,crop=crop,resumed=got,weight_error=error))
    print('BOUNDED_SEVEN_PREFLIGHT_PASS',error,flush=True)
    del tr,other;torch.cuda.empty_cache();tr=BoundedSevenFit(a,data)
    if a.resume:tr.restore(a.resume)
    history=[tr.report(data,dest)];tr.save(dest/f'candidate_{tr.step:06d}.pt');start=time.perf_counter()
    with (dest/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        while tr.step<a.steps:
            assert shutil.disk_usage(dest).free>512*1024**2
            record=tr.update();log.write(json.dumps(record)+'\n')
            if tr.step%10==0:print('BOUNDED_SEVEN_STEP',record,flush=True)
            if tr.step%50==0 or tr.step==a.steps:
                history.append(tr.report(data,dest));tr.save(dest/f'candidate_{tr.step:06d}.pt')
                tr.save(dest/'last.pt');write_json(dest/'history.json',history)
                if (dest/'STOP_REQUEST.json').exists():break
    write_json(dest/'completion.json',dict(executed_steps=tr.step,closures=tr.calls,seconds=time.perf_counter()-start,
        visual_acceptance=False,generalization_claim=False,history=history))


if __name__=='__main__':main()
