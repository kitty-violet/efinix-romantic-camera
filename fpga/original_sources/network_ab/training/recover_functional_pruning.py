"""Recover a causally screened physical pruning candidate, true FP32."""
import argparse
import gc
import json
import random
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .audit_style_branches import RUN,REPAIR,REPAIR_SHA,load,intervene,predict
from .functional_pruned_model import build,remove_attention
from .prune_functional_channels import data,measure,compare
from .prune_animation_teacher import profile
from .repair_style_retention import Trainer as PreviousTrainer,bands,gray
from .recover_pruned_teacher import evaluate
from .train_b1_distillation import read_image,as_tensor,to_rgb
from .losses import PerceptualLoss
from .runtime import configure_workspace,write_json,file_hash,contract_guard
from .engine import set_reproducible


class Trainer(PreviousTrainer):
    def __init__(self,root,cfg,manifest):
        self.config=cfg;self.step=0;self.rng=random.Random(cfg['seed']);self.groups={}
        for row in manifest['pairs']:
            assert manifest['source_splits'][row['source']]['split']=='train'
            self.groups.setdefault(row['source'],[]).append(row)
        path=Path(cfg['initial']);assert file_hash(path)==cfg['initial_sha256']
        state=torch.load(path,map_location='cpu',weights_only=True);self.indices=state['indices']
        self.net=build(root,state).cuda().eval();self.opt=torch.optim.Adam(self.net.parameters(),lr=cfg['lr'])
        self.vgg=PerceptualLoss(root).cuda();self.vgg.features=self.vgg.features[:9]

    def spatial(self,p,t,s):
        loss,terms=super().spatial(p,t,s)
        p,t=p[:,:,8:-8,8:-8],t[:,:,8:-8,8:-8]
        bp,bt=bands(p,1),bands(t,1)
        # Learn to remove only unsupported fine detail inside teacher-smooth
        # regions. No image blur or material classifier is added to inference.
        with torch.no_grad():
            g=gray(t);dx=F.pad((g[:,:,:,1:]-g[:,:,:,:-1]).abs(),(0,1,0,0));dy=F.pad((g[:,:,1:]-g[:,:,:-1]).abs(),(0,0,0,1))
            edge=F.max_pool2d(dx+dy,9,1,4)
            mask=((bt.abs()<3/128)&(edge<5/128)).float()
        surplus=(F.relu(bp.abs()-bt.abs()-.5/128)*mask).sum()/mask.sum().clamp_min(1)
        loss=loss+self.config['surplus_weight']*surplus;terms['surplus']=surplus
        return loss,terms


def prepare(root,run):
    out=run/'recovery'
    if out.exists():
        assert (out/'structural_preflight.json').exists() and not (out/'config.json').exists()
        shutil.copy2(out/'structural_preflight.json',out/'failed_pre_recovery_screen.json')
    out.mkdir(exist_ok=True)
    selection=json.loads((run/'channel_trials/selection.json').read_text());assert selection['status']=='COMPLETE'
    path=run/'channel_trials/candidate_fp32.pt' if selection['accepted'] else root/'runs'/REPAIR
    state=torch.load(path,map_location='cpu',weights_only=True);net=build(root,state).cuda().eval()
    # Compare an actual dead-code-removed graph against a zero attention gate.
    assets,flow=data(run,True);references={};handle=intervene(net,'attention',0)
    for row,s,t in assets:
        if row['split']=='seen_regression':references[row['id']]=predict(net,s,'repair')
    handle.remove();remove_attention(net);errors=[]
    for row,s,t in assets:
        if row['id'] in references:errors.append(int(np.abs(predict(net,s,'repair').astype(int)-references[row['id']].astype(int)).max()))
    assert max(errors)==0
    baseline_net=load(root,'repair');baseline=measure(baseline_net,assets,flow);del baseline_net;torch.cuda.empty_cache()
    validation=out/'validation_000000';validation.mkdir(exist_ok=True);values=measure(net,assets,flow,validation);checked=compare(values,baseline)
    write_json(out/'structural_preflight.json',dict(gate0_vs_physical_six_pixel_errors=errors,comparison=checked,measurements=values,baseline=baseline))
    # A failed visual screen is NOT waived. Recovery is a bounded experiment;
    # the identical original thresholds will be applied again after recovery.
    write_json(out/'candidate_status.json',dict(status='PRE_RECOVERY_SCREEN_PASS' if checked['screen_pass'] else 'NEEDS_RECOVERY_NOT_ACCEPTED',violations=checked['violations']))
    pr=profile(net,as_tensor(assets[0][1])[None].cuda());initial=out/'initial_fp32.pt'
    torch.save(dict(indices=state['indices'],removed_attention=True,generator={k:v.cpu() for k,v in net.state_dict().items()},parent_sha256=file_hash(path),production=False),initial)
    manifest=root/'runs/style-retention-repair-20261003/data_manifest.json'
    cfg=dict(mode='repair',seed=2026100403,max_steps=240,lr=.00010,lr_floor=.00002,
        weights=dict(rgb=1.,edge=2.,perceptual=.1,planes=.5),temporal=.2,change_weight=.15,surplus_weight=4.,
        initial=str(initial),initial_sha256=file_hash(initial),data_manifest=str(manifest),removed_attention=True,
        target='Original teacher legacy PNG targets; preserve structures and suppress unsupported detail in teacher-smooth regions',
        pipeline='No added inference postprocessing; dead attention branch physically removed')
    write_json(out/'config.json',cfg)
    names=['recover_functional_pruning.py','functional_pruned_model.py','repair_style_retention.py','importance_pruned_teacher.py','trainable_animegan.py','recover_pruned_teacher.py','prune_functional_channels.py','audit_style_branches.py']
    binding=dict(config_sha256=file_hash(out/'config.json'),manifest_sha256=file_hash(manifest),code={str(Path(__file__).with_name(n)):file_hash(Path(__file__).with_name(n)) for n in names})
    write_json(out/'binding.json',binding);write_json(out/'profile_initial.json',pr)
    (out/'inference_graph.txt').write_text(str(net.generator.graph),encoding='utf-8')
    print('PHYSICAL_EQUIVALENCE_PASS',pr['parameters'],pr['conv_macs'],'pre_recovery_screen',checked['screen_pass'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','preflight','train']);ap.add_argument('--stop-after',type=int,default=240);ap.add_argument('--resume',type=Path);args=ap.parse_args()
    root=configure_workspace();contract_guard();set_reproducible(2026100403);torch.set_num_threads(3);cv2.setNumThreads(2);run=root/'runs'/RUN
    if args.action=='prepare':prepare(root,run);return
    out=run/'recovery';cfg=json.loads((out/'config.json').read_text());binding=json.loads((out/'binding.json').read_text())
    assert all(file_hash(p)==h for p,h in binding['code'].items()) and file_hash(out/'config.json')==binding['config_sha256']
    assert file_hash(cfg['data_manifest'])==binding['manifest_sha256']
    manifest=json.loads(Path(cfg['data_manifest']).read_text(encoding='utf-8'));trainer=Trainer(root,cfg,manifest)
    if args.action=='preflight':
        folder=out/'preflight';folder.mkdir(exist_ok=False);full=trainer.update()
        (a,b,ta,tb),_,_,_=trainer.data();x=F.interpolate(torch.cat([a,b]),size=(360,640),mode='area');t=F.interpolate(torch.cat([ta,tb]),size=(360,640),mode='area')
        trainer.opt.zero_grad(set_to_none=True);loss,_=trainer.spatial(trainer.net(x),t,x);loss.backward();trainer.opt.step()
        torch.save(trainer.state(binding),folder/'resume.pt');reference=trainer.update();expected={k:v.cpu().clone() for k,v in trainer.net.state_dict().items()}
        del trainer;gc.collect();torch.cuda.empty_cache();trainer=Trainer(root,cfg,manifest)
        trainer.restore(torch.load(folder/'resume.pt',map_location='cpu',weights_only=True),binding);replay=trainer.update()
        error=max(float((v.cpu()-expected[k]).abs().max()) for k,v in trainer.net.state_dict().items() if v.is_floating_point());assert error<1e-5
        assert reference['source']==replay['source'] and reference['indices']==replay['indices']
        write_json(folder/'result.json',dict(status='PASS',binding=binding,full720_update=full,batch2_loss=float(loss.detach()),restore_error=error))
        print('PREFLIGHT_PASS',error,flush=True);return
    pre=json.loads((out/'preflight/result.json').read_text());assert pre['status']=='PASS' and pre['binding']==binding
    if args.resume:trainer.restore(torch.load(args.resume,map_location='cpu',weights_only=True),binding)
    else:assert not (out/'metrics.jsonl').exists()
    write_json(out/f'launch_{trainer.step:06d}.json',dict(step=trainer.step,resume=str(args.resume),binding=binding))
    with (out/'metrics.jsonl').open('a',encoding='utf-8',buffering=1) as log:
        while trainer.step<min(args.stop_after,cfg['max_steps']):
            row=trainer.update();log.write(json.dumps(row,ensure_ascii=False)+'\n')
            if trainer.step%20==0:print('RECOVERY',trainer.step,round(row['loss'],5),round(row['surplus'],6),flush=True)
            if trainer.step%120==0 or trainer.step==min(args.stop_after,cfg['max_steps']):
                torch.save(trainer.state(binding),out/f'checkpoint_{trainer.step:06d}.pt');evaluate(trainer.net,manifest,out/f'validation_{trainer.step:06d}')
    complete=trainer.step==cfg['max_steps'];write_json(out/'status.json',dict(status='COMPLETE' if complete else 'SEGMENT_COMPLETE',step=trainer.step))
    if complete:
        path=out/'functional_pruned_fp32.pt';torch.save(dict(schema='functional-pruned-teacher-v1',indices=trainer.indices,removed_attention=True,
            generator={k:v.cpu() for k,v in trainer.net.state_dict().items()},steps=trainer.step,binding=binding,production=False),path)
        reloaded=build(root,torch.load(path,map_location='cpu',weights_only=True)).cuda().eval();errors=[]
        with torch.inference_mode():
            for row in manifest['validation']:
                x=as_tensor(read_image(row['source']))[None].cuda();errors.append(float((reloaded(x)-trainer.net(x)).abs().max()))
        assert max(errors)==0;write_json(out/'export.json',dict(sha256=file_hash(path),reload_max_errors=errors,steps=trainer.step))
    print('SEGMENT_COMPLETE',trainer.step,flush=True)


if __name__=='__main__':main()
