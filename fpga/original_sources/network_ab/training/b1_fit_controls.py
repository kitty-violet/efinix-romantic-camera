"""Four bounded FP32 single-patch controls; never a generalization acceptance run."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from network_ab.contract_torch import build_fp32_model
from .engine import set_reproducible
from .optimization_policy import initialize_rgb_path
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .train_b1_distillation import as_tensor, to_rgb


RUN = 'b1-fit-controls-20261003'
CASES = [('rgb_identity', 'rgb', 'identity'), ('perturbed_identity', 'perturbed', 'identity'),
         ('rgb_style', 'rgb', 'style'), ('perturbed_style', 'perturbed', 'style')]


def model_for(kind, seed):
    torch.manual_seed(seed)
    net = build_fp32_model().cuda().eval()
    initialize_rgb_path(net)
    if kind == 'perturbed':
        # Break duplicate channels / centre-only taps without changing any operator.
        g = torch.Generator(device='cuda').manual_seed(seed + 100)
        with torch.no_grad():
            for m in net.modules():
                if isinstance(m, torch.nn.Conv2d):
                    fan_in = m.weight[0].numel()
                    m.weight.add_(torch.randn(m.weight.shape, generator=g, device='cuda') * (.02/math.sqrt(fan_in)))
    return net


def gradient(x):
    return x[:, :, 1:, :] - x[:, :, :-1, :], x[:, :, :, 1:] - x[:, :, :, :-1]


def terms(y, target, margin):
    y, target = y[:, :, margin:-margin, margin:-margin], target[:, :, margin:-margin, margin:-margin]
    hi = lambda a: a - F.avg_pool2d(a, 5, 1, 2)
    return {'rgb': F.l1_loss(y, target),
            'edge': sum(F.l1_loss(a, b) for a, b in zip(gradient(y), gradient(target))),
            'detail': F.l1_loss(hi(y), hi(target))}


def update(net, opt, x, target, step, cfg):
    opt.zero_grad(set_to_none=True)
    lr = cfg['lr_floor'] + (cfg['lr']-cfg['lr_floor']) * .5 * (1+math.cos(math.pi*step/(cfg['steps']-1)))
    lr *= min(1., (step+1)/20)
    for group in opt.param_groups: group['lr'] = lr
    y = net(x)
    losses = terms(y, target, cfg['margin'])
    loss = sum(losses[k]*v for k, v in cfg['loss_weights'].items())
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in net.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all(): raise FloatingPointError(name)
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
    assert norm > 0
    opt.step()
    return {'step': step+1, 'loss': float(loss.detach()), 'grad_norm': float(norm), 'lr': lr,
            **{k: float(v.detach()) for k, v in losses.items()}}


def score(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    ga = [np.diff(a, axis=d) for d in (0,1)]
    gb = [np.diff(b, axis=d) for d in (0,1)]
    mse = float(np.square(a-b).mean())
    return {'mae_rgb255': float(np.abs(a-b).mean()), 'psnr_rgb_db': float(10*np.log10(255**2/max(mse,1e-12))),
            'gradient_mae': float(sum(np.abs(x-y).mean() for x,y in zip(ga,gb))/2),
            'output_gradient': float(sum(np.abs(x).mean() for x in ga)/2),
            'target_gradient': float(sum(np.abs(x).mean() for x in gb)/2)}


def prepare(root, run):
    run.mkdir(exist_ok=False)
    old = root/'runs/b1-distillation-20261003'
    data = json.loads((old/'data/manifest.json').read_text(encoding='utf-8'))
    selected = next(r for r in data['validation'] if r['id']=='val_stairs_090')
    assert file_hash(selected['source'])==selected['source_sha256']
    assert file_hash(selected['target'])==selected['target_sha256']
    cfg = {'seed': 202610034, 'steps': 1200, 'lr': .003, 'lr_floor': .0003,
           'xywh': [384,280,400,400], 'margin': 40, 'loss_weights': {'rgb':1., 'edge':2., 'detail':1.},
           'fp32':True, 'amp':False, 'tf32':False, 'bn':'frozen_running_stats_train_affine',
           'ema':False, 'temporal_loss':False, 'perceptual_loss':False,
           'purpose':'single-patch fitting diagnostic; not production training or blind validation',
           'cases':CASES, 'selected':selected, 'regression':data['validation'],
           'contract':contract_guard()[1]['contract_sha256'],
           'split_override':'val_stairs_090 now used for fitting in this run only; stairs clip is NOT held out'}
    paths = [Path(__file__), Path(__file__).with_name('optimization_policy.py'),
             Path(__file__).with_name('train_b1_distillation.py'), Path(__file__).with_name('engine.py'),
             Path(__file__).with_name('runtime.py'), Path(__file__).parents[1]/'contract_torch.py',
             Path(__file__).parents[1]/'model_b_contract.json']
    cfg['code'] = {str(p):file_hash(p) for p in paths}
    frozen = run/'source';frozen.mkdir()
    for p in paths: shutil.copy2(p, frozen/p.name)
    write_json(run/'config.json',cfg)
    for key in ('source','target'):
        shutil.copy2(selected[key], run/(key+'_720.png'))
    x,y,w,h = cfg['xywh']; m = cfg['margin']
    for key in ('source','target'):
        a = Image.open(run/(key+'_720.png'))
        a.crop((x,y,x+w,y+h)).save(run/(key+'_fit_input.png'))
        a.crop((x+m,y+m,x+w-m,y+h-m)).save(run/(key+'_fit_interior.png'))
    write_json(run/'status.json', {'status':'PREPARED','steps_per_case':cfg['steps']})
    print('PREPARED',run,flush=True)


def inputs(run, cfg):
    x,y,w,h = cfg['xywh']
    source = as_tensor(np.array(Image.open(run/'source_720.png')))[None].cuda()
    target = as_tensor(np.array(Image.open(run/'target_720.png')))[None].cuda()
    return source, target, source[:,:,y:y+h,x:x+w], target[:,:,y:y+h,x:x+w]


def preflight(run,cfg):
    source,target,x,t = inputs(run,cfg)
    net=model_for('perturbed',cfg['seed']);opt=torch.optim.Adam(net.parameters(),lr=cfg['lr'])
    begin=time.perf_counter()
    full=update(net,opt,source,target,0,cfg)
    rows=[update(net,opt,x,t,i,cfg) for i in range(1,11)]
    saved={'model':copy.deepcopy(net.state_dict()),'opt':copy.deepcopy(opt.state_dict())}
    update(net,opt,x,t,11,cfg)
    expected={k:v.clone() for k,v in net.state_dict().items()}
    net.load_state_dict(saved['model']);opt.load_state_dict(saved['opt'])
    update(net,opt,x,t,11,cfg)
    error=max(float((v-expected[k]).abs().max()) for k,v in net.state_dict().items())
    assert error<2e-6
    torch.cuda.synchronize()
    report={'status':'PASS','full720_backward':full,'crop_backward_count':12,'restore_error':error,
            'elapsed_seconds':time.perf_counter()-begin,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'config_sha256':file_hash(run/'config.json')}
    write_json(run/'preflight.json',report);print('PREFLIGHT',json.dumps(report),flush=True)


def evaluate(net,run,cfg,out):
    out.mkdir(exist_ok=True);records=[]
    full,teacher,patch,tpatch=inputs(run,cfg)
    x,y,w,h=cfg['xywh'];m=cfg['margin']
    with torch.inference_mode():
        full_y=net(full)
        patch_y=net(patch)
        error=float((full_y[:,:,y+m:y+h-m,x+m:x+w-m]-patch_y[:,:,m:-m,m:-m]).abs().max())
        assert error<5e-5, error
        for row in cfg['regression']:
            a=np.array(Image.open(row['source']));b=np.array(Image.open(row['target']))
            output=to_rgb(net(as_tensor(a)[None].cuda())[0])
            Image.fromarray(output).save(out/(row['id']+'.png'))
            records.append({'id':row['id'],'against_source':score(output,a),'against_teacher':score(output,b),
                            'role':'fitting_frame' if row['id']==cfg['selected']['id'] else 'reviewed_regression_not_blind'})
        rgb=to_rgb(full_y[0])[y+m:y+h-m,x+m:x+w-m]
        src=np.array(Image.open(run/'source_fit_interior.png'));tgt=np.array(Image.open(run/'target_fit_interior.png'))
        Image.fromarray(rgb).save(out/'fit_interior.png')
    return {'fit_source':score(rgb,src),'fit_teacher':score(rgb,tgt),'full_patch_interior_max':error,'records':records}


def train(run,cfg):
    pre=json.loads((run/'preflight.json').read_text());assert pre['status']=='PASS' and pre['config_sha256']==file_hash(run/'config.json')
    if (run/'results.json').exists():raise FileExistsError('This bounded run has finished; do not repeat')
    source,teacher,x,t=inputs(run,cfg);results=[]
    for name,kind,task in CASES:
        out=run/name;out.mkdir(exist_ok=False)
        net=model_for(kind,cfg['seed']);opt=torch.optim.Adam(net.parameters(),lr=cfg['lr'])
        target=x if task=='identity' else t
        initial=evaluate(net,run,cfg,out/'initial')
        torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        with (out/'metrics.jsonl').open('w',encoding='utf-8',buffering=1) as stream:
            for step in range(cfg['steps']):
                row=update(net,opt,x,target,step,cfg);stream.write(json.dumps(row)+'\n')
                if (step+1)%100==0:
                    torch.cuda.synchronize()
                    print('CONTROL',name,step+1,'loss',round(row['loss'],6),'seconds',round(time.perf_counter()-begin,2),flush=True)
                    write_json(run/'status.json',{'status':'TRAINING','case':name,'step':step+1,'completed':[r['case'] for r in results]})
                if (step+1)==600:
                    torch.save({'model':net.state_dict(),'optimizer':opt.state_dict(),'step':600,'config_sha256':file_hash(run/'config.json')},out/'checkpoint_600.pt')
        torch.cuda.synchronize();elapsed=time.perf_counter()-begin
        final=evaluate(net,run,cfg,out/'final')
        path=out/'weights_fp32.pt'
        torch.save({'schema':'b1-single-patch-control-v1','model':{k:v.detach().cpu() for k,v in net.state_dict().items()},
                    'case':name,'step':cfg['steps'],'contract':cfg['contract'],'config_sha256':file_hash(run/'config.json'),
                    'production_candidate':False},path)
        reloaded=build_fp32_model().cuda().eval();reloaded.load_state_dict(torch.load(path,weights_only=True)['model'])
        with torch.inference_mode(): reload_error=float((net(source)-reloaded(source)).abs().max())
        assert reload_error==0
        result={'case':name,'initialization':kind,'target':task,'steps':cfg['steps'],'initial':initial,'final':final,
                'training_seconds':elapsed,'peak_mib':torch.cuda.max_memory_allocated()/2**20,
                'weights':str(path),'sha256':file_hash(path),'reload_error':reload_error}
        results.append(result)
        write_json(run/'results.partial.json',{'records':results})
        del net,reloaded,opt;torch.cuda.empty_cache()
        print('FINISHED',name,json.dumps(final['fit_source'] if task=='identity' else final['fit_teacher']),flush=True)
    write_json(run/'results.json',{'status':'COMPLETE_DIAGNOSTIC_NOT_GENERALIZATION','records':results,'config_sha256':file_hash(run/'config.json')})
    write_json(run/'status.json',{'status':'COMPLETED','cases':4,'total_steps':4800})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','preflight','train']);args=ap.parse_args()
    root=configure_workspace();run=root/'runs'/RUN
    contract_guard();set_reproducible(202610034);cv2.setNumThreads(2)
    if args.mode=='prepare':prepare(root,run);return
    cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
    for name,digest in cfg['code'].items(): assert file_hash(name)==digest, name
    (preflight if args.mode=='preflight' else train)(run,cfg)


if __name__=='__main__':main()
