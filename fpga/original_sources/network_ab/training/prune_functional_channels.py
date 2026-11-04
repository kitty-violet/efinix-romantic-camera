"""Small physical channel trials with independent, per-scene quality gates."""
import argparse
import copy
import json
import time
import cv2
import numpy as np
import torch
from PIL import Image
from .audit_style_branches import RUN,REPAIR,REPAIR_SHA,load,predict,spatial,motion
from .importance_pruned_teacher import build_group_pruned,transfer
from .prune_animation_teacher import profile
from .train_b1_distillation import as_tensor,read_image
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


def data(run,full=False):
    rows=json.loads((run/'assets.json').read_text())['rows'];result=[];counts={}
    for row in rows:
        seq=row.get('sequence')
        if not full:
            if row['split']=='seen_regression':continue
            if row['split']=='train' and not row['id'].endswith('current'):continue
            if seq:
                counts[seq]=counts.get(seq,0)+1
                if counts[seq]>2:continue
        result.append((row,read_image(row['source']),read_image(row['target'])))
    flow={};prev={}
    for row,s,t in result:
        if row.get('sequence'):
            seq=row['sequence']
            if seq in prev:flow[row['id']]=motion(prev[seq],s)
            prev[seq]=s
    return result,flow


def measure(net,assets,flow,folder=None):
    rows=[];time_rows=[];prev={}
    for row,s,t in assets:
        p=predict(net,s,'repair');rows.append(dict(id=row['id'],split=row['split'],**spatial(p,t,s)))
        seq=row.get('sequence')
        if folder and row['split']=='seen_regression':Image.fromarray(p).save(folder/(row['id']+'.png'))
        if seq:
            # RGB residual avoids hiding chromatic flicker in a luma metric.
            error=p.astype(np.float32)-t;effect=p.astype(np.float32)-s
            if seq in prev:
                mx,my,mask=flow[row['id']];pe,px=prev[seq]
                warp=lambda v:cv2.remap(v,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                time_rows.append(dict(id=row['id'],sequence=seq,teacher_relative=float(np.abs(error-warp(pe))[mask].mean()),effect=float(np.abs(effect-warp(px))[mask].mean())))
            prev[seq]=(error,effect)
    means={k:float(np.mean([r[k] for r in rows if r['split']=='train'])) for k in rows[0] if k not in ('id','split')}
    return dict(train=means,rows=rows,temporal=time_rows)


def compare(value,base):
    # Fixed conservative SCREENING tolerances, not user's visual acceptance.
    # Require every sampled scene, not just a good pooled average.
    rowbase={r['id']:r for r in base['rows']};tb={r['id']:r for r in base['temporal']}
    violations=[]
    for r in value['rows']:
        b=rowbase[r['id']]
        for key,relative,absolute in [('fine_error',.015,.025),('excess_detail',.02,.015),('structural_edge_error',.02,.05),('rgb_mae',.02,.1)]:
            if r[key]>b[key]*(1+relative)+absolute:violations.append([r['id'],key,r[key]-b[key]])
        if r['change_projection']<b['change_projection']-.02:violations.append([r['id'],'lost_teacher_change',r['change_projection']-b['change_projection']])
    for r in value['temporal']:
        for key in ('teacher_relative','effect'):
            if r[key]>tb[r['id']][key]*1.02+.03:violations.append([r['id'],key,r[key]-tb[r['id']][key]])
    delta={k:value['train'][k]-base['train'][k] for k in base['train']}
    return dict(screen_pass=not violations,violations=violations,train_delta=delta,
        worst_temporal_delta=max([r['teacher_relative']-tb[r['id']]['teacher_relative'] for r in value['temporal']]))


def oracle(root,run):
    folder=run/'channel_trials';folder.mkdir(exist_ok=False)
    state=torch.load(root/'runs'/REPAIR,map_location='cpu',weights_only=True);assert file_hash(root/'runs'/REPAIR)==REPAIR_SHA
    base=load(root,'repair');assets,flow=data(run);baseline=measure(base,assets,flow)
    prof=profile(base,as_tensor(assets[0][1])[None].cuda());cpu=base.cpu();del base;torch.cuda.empty_cache()
    write_json(folder/'baseline.json',dict(measurements=baseline,profile=prof,groups=['d1','d3','d5'],
        reason='Do not remove whole statistics/skip paths. Test ALL individual channels in the three decoder-only dependency groups; encoder/skip/attention widths are protected in this bounded trial.',
        calibration='Seven distinct training scenes and three real adjacent 30fps pairs. Short screening only; every combination remeasured on all56 assets before recovery.',
        source_sha256=file_hash(__file__),parent_sha256=REPAIR_SHA))
    records=[];begin=time.perf_counter()
    for group in ['d1','d3','d5']:
        width=len(state['indices'][group]);relative={g:list(range(len(v))) for g,v in state['indices'].items()}
        relative[group]=list(range(1,width));absolute={g:[state['indices'][g][i] for i in values] for g,values in relative.items()}
        net=build_group_pruned(root,absolute).cuda().eval()
        for ch in range(width):
            relative[group]=[i for i in range(width) if i!=ch];transfer(net,cpu,relative)
            v=measure(net,assets,flow);c=compare(v,baseline)
            result=dict(group=group,local_channel=ch,parent_channel=state['indices'][group][ch],**c,measurements=v)
            records.append(result);write_json(folder/'single_channels.json',dict(status='RUNNING',records=records))
            print('CHANNEL',group,ch,'pass',c['screen_pass'],'violations',len(c['violations']),round(time.perf_counter()-begin,1),flush=True)
        del net;torch.cuda.empty_cache()
    write_json(folder/'single_channels.json',dict(status='COMPLETE',records=records,seconds=time.perf_counter()-begin));print('CHANNEL_SCREEN_COMPLETE',flush=True)


def select(root,run):
    folder=run/'channel_trials';raw=json.loads((folder/'single_channels.json').read_text());assert raw['status']=='COMPLETE'
    state=torch.load(root/'runs'/REPAIR,map_location='cpu',weights_only=True);base=load(root,'repair')
    assets,flow=data(run,True);baseline=measure(base,assets,flow);cpu=base.cpu()
    # Rank only the strict per-scene survivors by detail error; no MAC score,
    # and no claims that independently safe deletions combine safely.
    eligible=sorted([r for r in raw['records'] if r['screen_pass']],key=lambda r:(r['train_delta']['excess_detail'],r['train_delta']['fine_error']))
    kept={g:list(range(len(v))) for g,v in state['indices'].items()};accepted=[];records=[]
    for i,row in enumerate(eligible[:8]):
        proposal=copy.deepcopy(kept);proposal[row['group']].remove(row['local_channel'])
        absolute={g:[state['indices'][g][j] for j in idx] for g,idx in proposal.items()}
        net=build_group_pruned(root,absolute).cuda().eval();transfer(net,cpu,proposal)
        out=folder/f'combination_{i:02d}';out.mkdir();values=measure(net,assets,flow,out);checked=compare(values,baseline)
        records.append(dict(removed=accepted+[(row['group'],row['local_channel'])],**checked));write_json(out/'measurements.json',dict(comparison=checked,measurements=values))
        if checked['screen_pass']:
            kept=proposal;accepted.append((row['group'],row['local_channel']))
            p=folder/'candidate_fp32.pt';torch.save(dict(indices=absolute,generator={k:v.cpu() for k,v in net.state_dict().items()},parent_sha256=REPAIR_SHA,removed=accepted,production=False),p)
            pr=profile(net,as_tensor(assets[0][1])[None].cuda());write_json(folder/'candidate.json',dict(profile=pr,sha256=file_hash(p),removed=accepted,comparison=checked,images=str(out)))
        print('COMBINATION',i,checked['screen_pass'],len(accepted),flush=True);del net;torch.cuda.empty_cache()
    write_json(folder/'selection.json',dict(status='COMPLETE',eligible=len(eligible),accepted=accepted,attempts=records,scope='All56 diagnostic assets and33 adjacent pairs; reused regression scenes, no blind acceptance'))
    print('FUNCTIONAL_SELECTION_COMPLETE',len(accepted),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['oracle','select']);args=ap.parse_args()
    root=configure_workspace();contract_guard();set_reproducible(2026100402);torch.set_num_threads(3);cv2.setNumThreads(2)
    globals()[args.action](root,root/'runs'/RUN)


if __name__=='__main__':main()
