"""FP32 recovery for dependency-group pruning, using the tested native720 loop."""
import argparse
import gc
import json
from pathlib import Path
import random
import shutil
import cv2
import torch
from torch.nn import functional as F
from .importance_pruned_teacher import RUN,build_group_pruned,WIDTHS
from .recover_pruned_teacher import Recovery,evaluate
from .losses import PerceptualLoss
from .train_b1_distillation import as_tensor,read_image
from .engine import set_reproducible
from .runtime import configure_workspace,write_json,file_hash,contract_guard


class GroupRecovery(Recovery):
    def __init__(self,root,run,cfg,manifest):
        self.config=cfg;self.step=0;self.rng=random.Random(cfg['seed']);self.groups={}
        for row in manifest['pairs']:
            assert manifest['source_splits'][row['source']]['split']=='train'
            self.groups.setdefault(row['source'],[]).append(row)
        p=run/cfg['candidate']/'initial_fp32.pt';assert file_hash(p)==cfg['initial_sha256']
        state=torch.load(p,map_location='cpu',weights_only=True);self.indices=state['indices']
        self.net=build_group_pruned(root,self.indices).cuda();self.net.load_state_dict(state['generator'])
        self.opt=torch.optim.Adam(self.net.parameters(),lr=cfg['lr'])
        self.vgg=PerceptualLoss(root).cuda();self.vgg.features=self.vgg.features[:9]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','preflight','train'])
    ap.add_argument('--candidate',choices=['cost18g','cost14g','cost10g'],required=True)
    ap.add_argument('--resume',type=Path);ap.add_argument('--stop-after',type=int,default=600);args=ap.parse_args()
    root=configure_workspace();run=root/'runs'/RUN;out=run/(args.candidate+'_recovery')
    contract_guard();set_reproducible(2026100312);cv2.setNumThreads(2)
    manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'))
    if args.mode=='prepare':
        out.mkdir(exist_ok=False)
        cfg=dict(candidate=args.candidate,seed=2026100312,max_steps=600,lr=.00035,lr_floor=.000075,
                 weights=dict(rgb=1.,edge=2.,perceptual=.1,planes=.5),temporal=.2,
                 initial_sha256=file_hash(run/args.candidate/'initial_fp32.pt'),
                 train_policy='Native whole720 adjacent frame pair, sequential gradients; FP32; no AMP/TF32/EMA',
                 target='Original full teacher + fixed accepted R48 targets; no inference guard')
        write_json(out/'config.json',cfg)
        paths=[Path(__file__),Path(__file__).with_name('importance_pruned_teacher.py'),Path(__file__).with_name('measure_pruning_importance.py'),
               Path(__file__).with_name('recover_pruned_teacher.py'),Path(__file__).with_name('prune_animation_teacher.py'),Path(__file__).with_name('trainable_animegan.py'),
               Path(__file__).with_name('train_b1_distillation.py'),Path(__file__).with_name('coherent_shared_model.py'),Path(__file__).with_name('losses.py')]
        binding=dict(config_sha256=file_hash(out/'config.json'),manifest_sha256=file_hash(run/'data_manifest.json'),
                     code={str(p):file_hash(p) for p in paths})
        write_json(out/'binding.json',binding);source=out/'source';source.mkdir()
        for p in paths:shutil.copy2(p,source/p.name)
        print('PREPARED',args.candidate,flush=True);return
    cfg=json.loads((out/'config.json').read_text());binding=json.loads((out/'binding.json').read_text())
    assert file_hash(out/'config.json')==binding['config_sha256'] and file_hash(run/'data_manifest.json')==binding['manifest_sha256']
    assert all(file_hash(p)==h for p,h in binding['code'].items())
    trainer=GroupRecovery(root,run,cfg,manifest)
    if args.mode=='preflight':
        p=out/'preflight';p.mkdir(exist_ok=False);full=trainer.update()
        (a,b,ta,tb),_,_,_=trainer.data()
        x=F.interpolate(torch.cat([a,b]),size=(360,640),mode='area');t=F.interpolate(torch.cat([ta,tb]),size=(360,640),mode='area')
        trainer.opt.zero_grad(set_to_none=True);loss=F.l1_loss(trainer.net(x),t);loss.backward();trainer.opt.step()
        torch.save(trainer.state(binding),p/'resume.pt');reference=trainer.update()
        expected={k:v.detach().cpu().clone() for k,v in trainer.net.state_dict().items()}
        del trainer;gc.collect();torch.cuda.empty_cache();trainer=GroupRecovery(root,run,cfg,manifest)
        trainer.restore(torch.load(p/'resume.pt',map_location='cpu',weights_only=True),binding);replayed=trainer.update()
        error=max(float((v.cpu()-expected[k]).abs().max()) for k,v in trainer.net.state_dict().items() if v.is_floating_point())
        assert reference['source']==replayed['source'] and reference['indices']==replayed['indices'] and error<1e-5,error
        write_json(p/'result.json',dict(status='PASS',binding=binding,full720_two_frame_update=full,batch2_360_loss=float(loss.detach()),restore_error=error))
        print('PREFLIGHT_PASS',args.candidate,error,flush=True);return
    pre=json.loads((out/'preflight/result.json').read_text());assert pre['status']=='PASS' and pre['binding']==binding
    if args.resume:
        trainer.restore(torch.load(args.resume,map_location='cpu',weights_only=True),binding)
        assert [json.loads(s)['step'] for s in (out/'metrics.jsonl').read_text().splitlines()]==list(range(1,trainer.step+1))
    else:
        if (out/'metrics.jsonl').exists():raise FileExistsError('Explicit resume required')
        evaluate(trainer.net,manifest,out/'validation_000000')
    write_json(out/f'launch_{trainer.step:06d}.json',dict(step=trainer.step,resume=str(args.resume),binding=binding))
    with (out/'metrics.jsonl').open('a',encoding='utf-8',buffering=1) as log:
        while trainer.step<min(args.stop_after,cfg['max_steps']):
            row=trainer.update();log.write(json.dumps(row,ensure_ascii=False)+'\n')
            if trainer.step%10==0:write_json(out/'progress.json',row);print('RECOVER',args.candidate,trainer.step,round(row['loss'],5),round(row['seconds'],3),flush=True)
            if trainer.step%100==0 or trainer.step==min(args.stop_after,cfg['max_steps']):
                torch.save(trainer.state(binding),out/f'checkpoint_{trainer.step:06d}.pt');evaluate(trainer.net,manifest,out/f'validation_{trainer.step:06d}')
    complete=trainer.step==cfg['max_steps'];write_json(out/'status.json',dict(status='COMPLETE' if complete else 'BOUNDED_SEGMENT_COMPLETE',step=trainer.step))
    if complete:
        path=out/'pruned_teacher_fp32.pt'
        torch.save(dict(schema='importance-group-pruned-recovered-fp32-v1',indices=trainer.indices,
                        generator={k:v.cpu() for k,v in trainer.net.state_dict().items()},steps=trainer.step,binding=binding,
                        production=False,inference_surface_guard=False),path)
        net=build_group_pruned(root,trainer.indices).cuda();net.load_state_dict(torch.load(path,weights_only=True)['generator'])
        x=as_tensor(read_image(manifest['validation'][0]['source']))[None].cuda()
        with torch.inference_mode():error=float((net(x)-trainer.net(x)).abs().max())
        assert error==0;write_json(out/'export.json',dict(path=str(path),sha256=file_hash(path),reload_error=error,steps=trainer.step))
    print('SEGMENT_COMPLETE',args.candidate,trainer.step,flush=True)


if __name__=='__main__':main()
