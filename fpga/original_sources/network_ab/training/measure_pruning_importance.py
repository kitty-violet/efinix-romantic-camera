"""Measure real full720 output damage for each removable dependency channel."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from .importance_pruned_teacher import RUN,PARENT,PARENT_SHA,WIDTHS,full_plan,parent,build_group_pruned,transfer,cost
from .prune_animation_teacher import profile
from .train_b1_distillation import as_tensor,to_rgb,read_image,read_flow,edges
from .coherent_shared_model import warp
from .recover_pruned_teacher import evaluate
from .runtime import configure_workspace,write_json,file_hash,contract_guard
from .engine import set_reproducible


def terms(a,b):
    a,b=a[:,:,8:-8,8:-8],b[:,:,8:-8,8:-8];d=(a-b).abs()
    tile=F.avg_pool2d(d.mean(1,keepdim=True),32,32).flatten()
    return dict(rgb=d.mean(),edge=sum((u-v).abs().mean() for u,v in zip(edges(a),edges(b))),
                planes=(F.avg_pool2d(a,16)-F.avg_pool2d(b,16)).abs().mean(),
                local=torch.topk(tile,max(1,tile.numel()//10)).values.mean())


def capture_data(root,run,net,collect=False):
    rows=json.loads((run/'measurement_sources.json').read_text(encoding='utf-8'))['pairs']
    values=[];moments={};counts={};hooks=[]
    if collect:
        def capture(name):
            def hook(module,args):
                a=args[0].detach()[0].flatten(1);n=min(a.shape[1],8192)
                ids=torch.linspace(0,a.shape[1]-1,n,device=a.device).long();a=a[:,ids].double()
                xx=(a@a.T).cpu();moments[name]=moments.get(name,torch.zeros_like(xx))+xx;counts[name]=counts.get(name,0)+n
            return hook
        hooks=[m.register_forward_pre_hook(capture(n)) for n,m in net.named_modules() if isinstance(m,nn.Conv2d)]
    with torch.inference_mode():
        for row in rows:
            a,b=[as_tensor(read_image(row[k]))[None].cuda() for k in ('current','next')]
            pa,pb=net(a),net(b);f,m=read_flow(row['flow'])
            flow=torch.from_numpy(f).permute(2,0,1)[None].cuda();mask=torch.from_numpy(m)[None,None].cuda()
            temporal=(pb-b)-warp(pa-a,flow)
            values.append((a,b,pa,pb,flow,mask,temporal,row['source']))
    for h in hooks:h.remove()
    if collect:torch.save({k:xx/counts[k] for k,xx in moments.items()},run/'parent_input_second_moments.pt')
    return values


def damage(net,values):
    records=[]
    with torch.inference_mode():
        for a,b,ra,rb,flow,mask,base_t,source in values:
            pa,pb=net(a),net(b);aa=terms(pa,ra);bb=terms(pb,rb)
            scalar={k:float((v+bb[k])*.5) for k,v in aa.items()}
            delta=(pb-b)-warp(pa-a,flow)-base_t
            scalar['temporal']=float((delta.abs()*mask).sum()/(3*mask.sum()+1e-6))
            scalar['total']=scalar['rgb']+2*scalar['edge']+.5*scalar['planes']+.5*scalar['local']+.2*scalar['temporal']
            records.append(dict(source=source,**scalar))
    mean=sum(r['total'] for r in records)/len(records);worst=max(r['total'] for r in records)
    return dict(mean=mean,worst=worst,score=mean+.5*worst,scenes=records)


def prepare(root,run):
    run.mkdir(exist_ok=False)
    manifest=json.loads((root/'runs/teacher-pruning-calibrated-20261003/data_manifest.json').read_text(encoding='utf-8'))
    write_json(run/'data_manifest.json',manifest)
    groups={}
    for r in manifest['pairs']:
        assert manifest['source_splits'][r['source']]['split']=='train'
        groups.setdefault(r['source'],[]).append(r)
    chosen=[rows[len(rows)//2] for _,rows in sorted(groups.items())]
    assert len(chosen)==7
    bindings={row[k]:file_hash(row[k]) for row in chosen for k in ('current','next','flow')}
    write_json(run/'measurement_sources.json',dict(pairs=chosen,input_hashes=bindings,selection='one middle adjacent pair from each of seven training-only sources; no held-out frames'))
    base=copy.deepcopy(parent(root)).cuda();net=build_group_pruned(root,full_plan()).cuda()
    errors=[]
    with torch.inference_mode():
        for row in manifest['validation']:
            x=as_tensor(read_image(row['source']))[None].cuda();errors.append(float((net(x)-base(x)).abs().max()))
        batch=F.interpolate(torch.cat([x,x.flip(-1)]),size=(360,640),mode='area')
        batch_error=float((net(batch)-base(batch)).abs().max())
    assert max(errors)<5e-5 and batch_error<5e-5,(errors,batch_error)
    prof=profile(net,x);spatial={r['name']:r['shape'][0]*r['shape'][2]*r['shape'][3] for r in prof['convs']}
    assert cost(full_plan(),spatial)['conv_macs']==prof['conv_macs']
    probe={g:list(range(max(2,c//2))) for g,c in WIDTHS.items()};probe['attn_projection']=list(range(24));probe['attn_memory']=list(range(12))
    small=build_group_pruned(root,probe).cuda();y=small(x);loss=y.square().mean();loss.backward()
    assert torch.isfinite(y).all() and all(p.grad is not None and torch.isfinite(p.grad).all() for p in small.parameters())
    small_prof=profile(small,x);assert cost(probe,spatial)['conv_macs']==small_prof['conv_macs']
    write_json(run/'equivalence.json',dict(status='PASS',six_full720_errors=errors,batch2_360_error=batch_error,
        nonuniform_graph_forward_backward=True,nonuniform_probe_profile=small_prof,baseline_profile=prof))
    write_json(run/'spatial.json',spatial)
    paths=[Path(__file__),Path(__file__).with_name('importance_pruned_teacher.py'),Path(__file__).with_name('prune_animation_teacher.py'),Path(__file__).with_name('trainable_animegan.py')]
    source=run/'source';source.mkdir()
    for p in paths:shutil.copy2(p,source/p.name)
    write_json(run/'binding.json',dict(parent=str(root/'runs'/PARENT),parent_sha256=PARENT_SHA,
        code={str(p):file_hash(p) for p in paths},measurement_sources_sha256=file_hash(run/'measurement_sources.json')))
    print('GROUP_GRAPH_PREPARED',max(errors),batch_error,flush=True)


def oracle(root,run):
    binding=json.loads((run/'binding.json').read_text());assert all(file_hash(p)==h for p,h in binding['code'].items())
    assert not (run/'importance.jsonl').exists()
    spatial=json.loads((run/'spatial.json').read_text());base=copy.deepcopy(parent(root)).cuda();base.eval()
    values=capture_data(root,run,base,collect=True);cpu=parent(root)
    zero=damage(base,values);assert zero['score']<1e-7
    del base;torch.cuda.empty_cache();records=[];begin=time.perf_counter()
    with (run/'importance.jsonl').open('w',encoding='utf-8',buffering=1) as log:
        for group,width in WIDTHS.items():
            plan=full_plan();plan[group]=list(range(1,width));candidate=build_group_pruned(root,plan).cuda()
            saving=cost(full_plan(),spatial)['conv_macs']-cost(plan,spatial)['conv_macs']
            for channel in range(width):
                plan[group]=[i for i in range(width) if i!=channel];transfer(candidate,cpu,plan)
                result=damage(candidate,values)
                record=dict(group=group,channel=channel,mac_saved=saving,**result);records.append(record);log.write(json.dumps(record,ensure_ascii=False)+'\n')
                if (channel+1)%8==0:print('IMPORTANCE',group,channel+1,'/',width,'total',len(records),round(time.perf_counter()-begin,1),flush=True)
            del candidate;torch.cuda.empty_cache()
            write_json(run/'importance_progress.json',dict(group_completed=group,count=len(records),seconds=time.perf_counter()-begin))
    write_json(run/'importance.json',dict(status='COMPLETE',parent_sha256=PARENT_SHA,records=records,
        seconds=time.perf_counter()-begin,zero_control=zero,
        method='Physical deletion of one dependent channel at a time; whole720 adjacent pairs from all 7 training sources. RGB + 2*edge + .5*16x16 planes + .5*top10percent32x32 tile + .2*flow-warped temporal perturbation. Rank mean + .5 worst source.',
        limitations=['Individual removal cost does not add linearly when many channels are deleted.','Measures preservation of the accepted parent, not proof of a perfect style or unseen-source generalization.']))
    print('IMPORTANCE_COMPLETE',len(records),flush=True)


def select(root,run):
    raw=json.loads((run/'importance.json').read_text());assert raw['status']=='COMPLETE'
    spatial=json.loads((run/'spatial.json').read_text());moments=torch.load(run/'parent_input_second_moments.pt',weights_only=True)
    rankings={g:sorted([r for r in raw['records'] if r['group']==g],key=lambda r:r['score']) for g in WIDTHS}
    summary=[]
    for budget in (18,14,10):
        plan=full_plan();removed={g:0 for g in WIDTHS};history=[]
        while cost(plan,spatial)['conv_macs']>budget*1e9:
            choices=[]
            for g,c in WIDTHS.items():
                if len(plan[g])-2<max(4,c//4):continue
                cut=rankings[g][removed[g]:removed[g]+2];trial=copy.deepcopy(plan);trial[g]=[i for i in plan[g] if i not in [r['channel'] for r in cut]]
                gain=cost(plan,spatial)['conv_macs']-cost(trial,spatial)['conv_macs']
                choices.append((sum(r['score'] for r in cut)/gain,g,trial,gain,[r['channel'] for r in cut]))
            assert choices,'No feasible group'
            ratio,g,plan,gain,cut=min(choices,key=lambda x:x[0]);removed[g]+=2;history.append(dict(group=g,channels=cut,gain=gain,damage_per_mac=ratio))
        name=f'cost{budget}g';out=run/name;out.mkdir(exist_ok=False);net=build_group_pruned(root,plan).cuda()
        transfer(net,parent(root),plan,moments)
        p=out/'initial_fp32.pt';torch.save(dict(schema='importance-group-pruned-teacher-v1',indices=plan,generator={k:v.cpu() for k,v in net.state_dict().items()},parent_sha256=PARENT_SHA,production=False),p)
        manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'));evaluate(net,manifest,out/'initial_validation')
        write_json(out/'plan.json',dict(plan=plan,profile=cost(plan,spatial),greedy_history=history,initial_sha256=file_hash(p),input_compensation=True))
        summary.append(dict(name=name,profile=cost(plan,spatial),widths={g:len(v) for g,v in plan.items()}))
        print('CANDIDATE',name,summary[-1],flush=True);del net;torch.cuda.empty_cache()
    write_json(run/'candidates.json',dict(records=summary,status='INITIAL_CANDIDATES_READY_NOT_QUALITY_ACCEPTED'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','oracle','select']);args=ap.parse_args()
    root=configure_workspace();contract_guard();set_reproducible(2026100311);cv2.setNumThreads(2)
    globals()[args.mode](root,root/'runs'/RUN)


if __name__=='__main__':main()
