"""Learn clean regions using retained reconstruction features at360p.

Supervision uses the unclipped common brightness coordinate: hard RGB output
must not be the only training signal through saturated ordinal bands. Only
ordinary B1 weights are exported. No early binary geometry is imposed.
"""
import argparse
import copy
import json
import shutil
import sys
import inspect
import time
from pathlib import Path
import cv2
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from .seed_b1_clear_regions import BASES,INCREMENTS
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


def executed_sources(training_factory):
    """Capture imported project dependencies and actual MRO method resolution.

    Hand-maintained extra_sources missed inherited optimizers/losses. The
    resolved methods identify what actually runs rather than stale class flags.
    """
    root=Path(__file__).resolve().parents[2]
    paths=[]
    for name,module in list(sys.modules.items()):
        if not (name=='network_ab' or name.startswith('network_ab.')):continue
        filename=getattr(module,'__file__',None)
        if not filename:continue
        p=Path(filename).resolve()
        if p.suffix=='.py' and p.is_file() and p.is_relative_to(root):paths.append(p)
    methods={}
    for method in ('__init__','batch_style','update','appearance_loss','additional_loss',
                   'count_supervision','count_logits','input_stability_loss','save','restore'):
        fn=getattr(training_factory,method,None)
        if fn is not None:
            # A small objective-only factory inherits object.__init__, whose
            # wrapper descriptor has no Python module/source file.
            python_callable=inspect.isfunction(fn) or inspect.ismethod(fn)
            methods[method]=dict(module=getattr(fn,'__module__','builtins'),name=fn.__qualname__,file=inspect.getsourcefile(fn) if python_callable else None)
    return sorted(set(paths)),dict(mro=[c.__module__+'.'+c.__name__ for c in training_factory.__mro__],methods=methods)


class LateB1(m.t.r.c.ColourB1):
    def forward(self,x,gate=None,mix=0.,all_nodes=False):
        nodes=super().forward(x,gate=gate,mix=mix,all_nodes=True)
        if nodes['up2_low_pw'].requires_grad:
            nodes['up2_low_pw'].register_hook(lambda g:g*.02)
        return nodes if all_nodes else (nodes['rgb_head']/128,nodes['res3_add'][:,24:25])


class LateTraining(m.AreaTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        self.net.__class__=LateB1
        self.project()
        def remember(module,inputs,result):
            if isinstance(result,dict):self.nodes=result
        self.net.register_forward_hook(remember)

    @torch.no_grad()
    def project(self):
        w,b=self.net.weight['up2_low_pw'],self.net.bias['up2_low_pw']
        s=w.new_tensor(INCREMENTS)/127;offset=b.new_tensor(BASES)
        common_w=(w[:6]*s[:,None,None,None]).mean(0)
        common_b=(b[:6]*s+offset).mean()
        w[:6]=common_w[None]/s[:,None,None,None];b[:6]=(common_b-offset)/s

    def additional_loss(self,p,target,native):
        area,values=super().additional_loss(p,target,native)
        feature=self.nodes['up2_low_dw']
        if not native:feature=feature[...,20:-20,20:-20]
        w,b=self.net.weight['up2_low_pw'],self.net.bias['up2_low_pw']
        s=w.new_tensor(INCREMENTS)/127;offset=b.new_tensor(BASES)
        yw=(w[:6]*s[:,None,None,None]).mean(0,keepdim=True)
        yb=(b[:6]*s+offset).mean()[None]
        coord=F.conv2d(feature,torch.cat((yw,w[6:]),0),torch.cat((yb,b[6:])))
        z=F.avg_pool2d((target+1)*64,2);r,g,blue=z[:,0:1],z[:,1:2],z[:,2:3]
        truth=torch.cat(((r+2*g+blue)/4,(r-blue)/2+64,(-r+2*g-blue)/4+64),1)
        thresholds=coord.new_tensor([lo+inc*(j+.5)/3 for lo,inc in zip(BASES,INCREMENTS) for j in range(3)])[None,:,None,None]
        ordinal=F.binary_cross_entropy_with_logits((coord[:,:1]-thresholds)/2.5,(truth[:,:1]>thresholds).float())
        colour=F.smooth_l1_loss(coord/32,truth/32)
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        labels=labels[:,::2,::2]
        flat,flat_values=m.region_loss(coord/64,truth/64,labels)
        loss=ordinal+2*colour+.7*flat
        values.update(late_ordinal=ordinal,late_colour=colour,late_region_flat=flat_values['area_flatness'])
        return area+self.args.classification*loss,values

    def update(self,native=None):
        result=super().update(native);self.project();return result


def preflight(args,d,dest,factory=LateTraining):
    a=factory(args,d)
    if getattr(args,'resume_from',''):a.restore(args.resume_from)
    native=a.update(True);crop=a.update(False);a.update(False);a.update(False)
    extra=a.preflight_extras() if hasattr(a,'preflight_extras') else None
    a.save(dest/'preflight.pt')
    resume_native=bool(getattr(a,'preflight_native_resume',False))
    expected=a.update(resume_native);state=copy.deepcopy(a.net.state_dict())
    heads=copy.deepcopy(a.heads.state_dict())
    critic=copy.deepcopy(a.critic.state_dict()) if a.critic is not None else None
    b=factory(args,d);b.restore(dest/'preflight.pt');actual=b.update(resume_native)
    error=max(float((v-state[k]).abs().max()) for k,v in b.net.state_dict().items())
    assert error<1e-5 and actual['samples']==expected['samples']
    head_error=max([float((v-heads[k]).abs().max()) for k,v in b.heads.state_dict().items()]+[0.])
    critic_error=max([float((v-critic[k]).abs().max()) for k,v in b.critic.state_dict().items()]+[0.]) if critic is not None else None
    loss_error=abs(actual['loss']-expected['loss']);grad_error=abs(actual['grad']-expected['grad'])
    assert head_error<1e-5 and (critic_error is None or critic_error<1e-5) and loss_error<1e-6 and grad_error<1e-5
    write_json(dest/'preflight.json',dict(status='PASS',native=native,crop=crop,extra=extra,resume_error=error,
        head_resume_error=head_error,critic_resume_error=critic_error,loss_resume_error=loss_error,gradient_resume_error=grad_error))
    print('LATE_REGION_PREFLIGHT',error,flush=True)
    del a,b;torch.cuda.empty_cache()


def main(training_factory=LateTraining,extra_sources=(),evaluation=m.evaluate,preview=m.t.preview):
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--steps',type=int,default=4000)
    ap.add_argument('--warm',default=str(m.t.OUT/'clear-regions/best.pt'))
    ap.add_argument('--lr',type=float,default=.00025);ap.add_argument('--classification',type=float,default=2.)
    ap.add_argument('--lr-schedule',choices=('cosine','constant'),default='cosine')
    ap.add_argument('--interval',type=int,default=500)
    ap.add_argument('--fp32',action='store_true')
    ap.add_argument('--teacher-checkpoint',default='')
    ap.add_argument('--data-manifest',default='')
    ap.add_argument('--unfreeze-head',action='store_true')
    ap.add_argument('--class-manifest',default='',help='Optional immutable training-only palette label revision')
    ap.add_argument('--resume-from',default='',help='Resume model, optimizer and RNG in a new run directory')
    ap.add_argument('--shift',type=float,default=.15)
    ap.add_argument('--temporal',type=float,default=.25)
    ap.add_argument('--temporal-mask',choices=('target','source'),default='target')
    ap.add_argument('--adversarial',type=float,default=0.)
    ap.add_argument('--critic-bands',action='store_true')
    ap.add_argument('--critic-artifacts',action='store_true')
    ap.add_argument('--critic-balance',choices=('appearance','actual_output'),default='appearance')
    ap.add_argument('--critic-warm',default='')
    ap.set_defaults(seed=2026100679,minimum_area=256,area_weight=.7,freeze_head=True,projection=0.,
        init='identity',aux=0,perceptual=0,qat=True,scale_lr=0,crop=320,batch=3,hard=False,shift=.15,temporal=.25,
        mode='direct',adversarial=0,regions=0,qat_teacher='',critic_warm='',critic_bands=False,tied_regions=False,
        colour=False,rotate=False,latent_refine=0)
    args=ap.parse_args();configure_workspace();set_reproducible(args.seed);cv2.setNumThreads(2);contract_guard()
    args.qat=not args.fp32
    if args.unfreeze_head:args.freeze_head=False
    args.training_class=training_factory.__name__
    args.evaluation_function=evaluation.__module__+'.'+evaluation.__name__
    d=m.load_data(args.data_manifest or None);dest=m.OUT/args.name;dest.mkdir(exist_ok=False)
    paths=[Path(__file__),Path(m.__file__),Path(m.t.__file__),Path(m.t.s.__file__),Path(m.t.r.__file__),Path(m.t.r.c.__file__)]
    paths.extend(extra_sources)
    imported,execution=executed_sources(training_factory)
    paths=sorted(set(paths+imported))
    for p in paths:shutil.copy2(p,dest/p.name)
    frozen={str(p):file_hash(p) for p in paths}
    def verify_code():
        assert all(file_hash(p)==digest for p,digest in frozen.items()),'Executing source changed during run'
    write_json(dest/'config.json',dict(args=vars(args),code={str(p):file_hash(p) for p in paths},
        data_sha256=file_hash(args.data_manifest or m.OUT/'data.json'),warm_sha256=file_hash(args.warm),contract_sha256=m.t.r.CONTRACT_SHA,
        precision='FP32_without_AMP_or_TF32',inference='Unchanged B1; no postfilter',
        teacher_sha256=file_hash(args.teacher_checkpoint) if args.teacher_checkpoint else None,
        class_manifest_sha256=file_hash(args.class_manifest) if args.class_manifest else None,
        resume_sha256=file_hash(args.resume_from) if args.resume_from else None,execution_resolution=execution))
    if args.resume_from:
        resumed=torch.load(args.resume_from,map_location='cpu',weights_only=True)
        assert resumed['args'].get('training_class')==training_factory.__name__,'Resume requires the same training objective'
        assert resumed['step']<args.steps
    preflight(args,d,dest,training_factory);tr=training_factory(args,d)
    if args.resume_from:tr.restore(args.resume_from)
    start_step=tr.step;e=evaluation(tr.net,d['validation']);best=e['mean']['score']
    history=[dict(step=tr.step,validation=e)];tr.save(dest/'best.pt');preview(tr.net,d['validation'],dest/'initial-preview')
    if hasattr(tr,'report_training_fit'):history[-1]['training_fit']=tr.report_training_fit()
    def integer_replay(step):
        if args.qat:return None
        tr.net.quantized=True
        try:
            result=evaluation(tr.net,d['validation'])
            preview(tr.net,d['validation'],dest/f'integer_preview_{step:06d}')
            return result
        finally:tr.net.quantized=False
    history[-1]['same_weights_integer_replay']=integer_replay(tr.step)
    started=time.perf_counter();stop=None
    with (dest/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        while tr.step<args.steps:
            assert shutil.disk_usage(m.OUT).free>512*1024**2
            row=tr.update();log.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('LATE_REGION_STEP',tr.step,round(row['loss'],5),flush=True)
            if tr.step%args.interval==0 or tr.step==args.steps:
                verify_code()
                e=evaluation(tr.net,d['validation']);history.append(dict(step=tr.step,validation=e))
                if hasattr(tr,'report_training_fit'):history[-1]['training_fit']=tr.report_training_fit()
                tr.save(dest/f'candidate_{tr.step:06d}.pt');tr.save(dest/'last.pt');preview(tr.net,d['validation'],dest/f'preview_{tr.step:06d}')
                history[-1]['same_weights_integer_replay']=integer_replay(tr.step)
                if e['mean']['score']<best:best=e['mean']['score'];tr.save(dest/'best.pt')
                write_json(dest/'history.json',history);print('LATE_REGION_EVAL',tr.step,e['mean'],flush=True)
                req=dest/'STOP_REQUEST.json'
                if req.exists():stop=json.loads(req.read_text(encoding='utf-8'));assert stop.get('reason');break
    state=torch.load(dest/'best.pt',map_location='cpu',weights_only=True);tr.net.load_state_dict(state['model'])
    preview(tr.net,d['validation'],dest/'preview')
    write_json(dest/'completion.json',dict(executed_steps=tr.step,start_step=start_step,new_updates=tr.step-start_step,
        selected_step=state['step'],elapsed_seconds=time.perf_counter()-started,
        validation=evaluation(tr.net,d['validation']),stop_evidence=stop,visual_acceptance=False))
    print('LATE_REGION_COMPLETE',tr.step,state['step'],flush=True)


if __name__=='__main__':main()
