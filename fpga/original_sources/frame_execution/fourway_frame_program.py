"""Compile live row dependencies for the continuous fourway RTL driver.

No time estimates: every transfer and compute command must execute in RTL.
Rows retain the same finite 12 x 1024 x 64 arena until the final consumer.
Nearest is an address view. Last PW and residual add share one output stream.
"""
from __future__ import annotations
import argparse,json,sys
from collections import Counter
from pathlib import Path
from compile_frame import Arena,ROOT,HERE,sha
from model_contract import compile_contract,load_contract
import hardware_schedule_model as hw
from coefficient_layout import build_layout


def graph(plan,group,tile,repeat):
    byid={n['id']:n for n in plan['nodes']}
    if group['name']=='residual' and repeat==-1:
        local={'source':dict(id='source',kind='source',inputs=[],parents=[],output_shape_hwc=[180,320,32])}
        for node in plan['nodes']:
            if not node['id'].startswith(('res1_','res2_','res3_')):continue
            n=dict(node);n['parents']=[p if p.startswith(('res1_','res2_','res3_')) else 'source' for p in n['inputs']]
            if n['kind']=='residual_add':
                pw=local.pop(n['parents'][0]);n.update(kind='pw_fused',parameter_node=pw['id'],
                    parents=[pw['parents'][0],n['parents'][1]],input_shape_hwc=pw['input_shape_hwc'],stride=1,input_channels=pw['input_channels'])
            local[n['id']]=n
        final='res3_add';ranges={final:(tile['commit_interval']['x0'],tile['commit_interval']['x1'])}
        for name in reversed(local):
            if name=='source':continue
            a,b=ranges[name];n=local[name]
            if n['kind']=='dw3x3':a,b=max(0,a-1),min(319,b+1)
            for parent in n['parents']:
                lo,hi=ranges.get(parent,(a,b));ranges[parent]=(min(lo,a),max(hi,b))
        assert set(local)==set(ranges)
        return local,ranges,final
    rename=lambda x:x.replace('resX',f'res{repeat}')
    steps={rename(s['operation']):s for s in tile['steps']}
    local={name:dict(byid[name]) for name in steps}
    first=next(iter(local.values()))
    ranges={name:(s['output_interval']['x0'],s['output_interval']['x1']) for name,s in steps.items()}
    source_shape=first['input_shape_hwc']
    local={'source':dict(id='source',kind='source',inputs=[],output_shape_hwc=source_shape),**local}
    ranges['source']=(tile['source_interval']['x0'],tile['source_interval']['x1'])
    for n in local.values():n['parents']=[p if p in local else 'source' for p in n['inputs']]
    final=list(steps)[-1]
    if repeat:
        add=f'res{repeat}_add';pw=local.pop(final)
        n=dict(byid[add]);n.update(kind='pw_fused',parents=[pw['parents'][0],'source'],parameter_node=final,
                                  input_shape_hwc=pw['input_shape_hwc'],stride=1,input_channels=pw['input_channels'])
        local[add]=n;ranges[add]=ranges.pop(final);final=add
    return local,ranges,final


def row_tasks(local,final,block,aligned=False,prefetch_rows=0,compact_lifetime=False):
    done={name:-1 for name in local};tasks=[]
    def physical(name,y):
        n=local[name]
        return physical(n['parents'][0],y//2) if n['kind']=='nearest_x2' else (name,y)
    def ensure(name,hi):
        n=local[name]
        if n['kind']=='source':hi+=prefetch_rows
        if aligned and n['kind'] not in ('source','nearest_x2'):
            quantum=2 if n.get('stride',1)==2 else 4
            hi=((hi+quantum)//quantum)*quantum-1
        hi=min(hi,n['output_shape_hwc'][0]-1);lo=done[name]+1
        if lo>hi:return
        quantum=(2 if n.get('stride',1)==2 else 4) if compact_lifetime and n['kind'] not in ('source','nearest_x2') else hi-lo+1
        for begin in range(lo,hi+1,quantum):
            end=min(hi,begin+quantum-1);dep=[]
            for parent in n['parents']:
                ph=local[parent]['output_shape_hwc'][0]
                if n['kind']=='nearest_x2':a,b=begin//2,end//2
                elif n['kind'] in ('conv3x3','dw3x3'):a,b=max(0,begin*n['stride']-1),min(ph-1,end*n['stride']+1)
                else:a,b=begin,end
                ensure(parent,b);dep.extend(physical(parent,y) for y in range(a,b+1))
            if n['kind']!='nearest_x2':
                tasks.append(dict(id=len(tasks),node=name,lo=begin,hi=end,dependencies=sorted(set(dep)),outputs=[(name,y) for y in range(begin,end+1)]))
            done[name]=end
    for lo in range(0,local[final]['output_shape_hwc'][0],block):
        hi=min(local[final]['output_shape_hwc'][0]-1,lo+block-1);ensure(final,hi)
        tasks.append(dict(id=len(tasks),node='commit',lo=lo,hi=hi,dependencies=[(final,y) for y in range(lo,hi+1)],outputs=[]))
    assert all(done[name]==n['output_shape_hwc'][0]-1 for name,n in local.items())
    return tasks


def pack_lifetimes(tasks,last,local,ranges,plane_offsets):
    """Offline interval placement; returned bases still pass the live allocator.

    No data relocation. Two rows may share addresses only if their complete
    producer/consumer lifetime intervals are disjoint (inclusive endpoints).
    """
    records={}
    for task in tasks:
        for key in task['outputs']:
            name,y=key;width=ranges[name][1]-ranges[name][0]+1
            records[key]=dict(born=task['id'],last=last[key],plane=(y+plane_offsets[name])%3,
                              length=((width+3)//4)*((local[name]['output_shape_hwc'][2]+7)//8))
    neighbors={key:set() for key in records};active=[set() for _ in range(3)]
    for key,r in sorted(records.items(),key=lambda x:x[1]['born']):
        plane=r['plane'];active[plane]={k for k in active[plane] if records[k]['last']>=r['born']}
        neighbors[key].update(active[plane])
        for k in active[plane]:neighbors[k].add(key)
        active[plane].add(key)
    strategies=[lambda k:(-records[k]['length'],records[k]['born'],k),
                lambda k:(-(records[k]['last']-records[k]['born']),-records[k]['length'],records[k]['born'],k),
                lambda k:(records[k]['born'],k)]
    for strategy in strategies:
        placed={}
        for key in sorted(records,key=strategy):
            r=records[key];length=r['length'];at=0
            for lo,hi in sorted((placed[k],placed[k]+records[k]['length']) for k in neighbors[key] if k in placed):
                if lo-at>=length:break
                at=max(at,hi)
            if at+length>1024:break
            placed[key]=at
        else:return placed
    raise MemoryError(('offline interval placement exceeds a physical bank',key,at,length))


def compile_tile(plan,group,tile,repeat,block,aligned=False,plane_shift=False,prefetch_rows=0,retain_commit=False,compact_lifetime=False,plane_pattern=None):
    local,ranges,final=graph(plan,group,tile,repeat)
    tasks=row_tasks(local,final,block,aligned,prefetch_rows,compact_lifetime)
    plane_offsets={'source':0}
    for name,node in local.items():
        if name=='source':continue
        parent=node['parents'][0]
        plane_offsets[name]=(plane_offsets[parent]+int(node['kind']!='nearest_x2'))%3 if plane_shift else 0
    if plane_pattern is not None:
        assert group['name']=='residual' and len(plane_pattern)==4
        base=0;position=0
        for name in local:
            if name=='source':continue
            plane_offsets[name]=(base+plane_pattern[position])%3
            position=(position+1)%4
            if position==0:base=plane_offsets[name]
    last={tuple(k):t['id'] for t in tasks for k in t['dependencies']}
    if retain_commit:
        commits=[t for t in tasks if t['node']=='commit']
        for old,new in zip(commits,commits[1:]):
            for key in old['dependencies']:last[tuple(key)]=new['id']
    placements=pack_lifetimes(tasks,last,local,ranges,plane_offsets) if compact_lifetime else {}
    arena=Arena(1024);commands=[];born=set();used=set();words=Counter();peak=0
    def desc(key):
        plane,base,width,groups=arena.rows[key]
        return dict(plane=plane,base=base,width=width,groups=groups,origin=ranges[key[0]][0],y=key[1])
    def physical(name,y):
        n=local[name]
        y=max(0,min(n['output_shape_hwc'][0]-1,y))
        return physical(n['parents'][0],y//2) if n['kind']=='nearest_x2' else (name,y)
    for task in tasks:
        assert all(tuple(k) in arena.rows for k in task['dependencies'])
        for name,y in task['outputs']:
            assert (name,y) not in born;born.add((name,y))
            arena.allocate((name,y),ranges[name][1]-ranges[name][0]+1,local[name]['output_shape_hwc'][2],plane_offsets[name],placements.get((name,y)))
        for plane in arena.used:
            sorted_ranges=sorted(plane)
            assert all(a[1]<=b[0] for a,b in zip(sorted_ranges,sorted_ranges[1:]))
        name=task['node'];n=local.get(name)
        if name in ('source','commit'):
            keys=task['outputs'] if name=='source' else task['dependencies']
            for key in keys:
                d=desc(tuple(key));c=local[key[0]]['output_shape_hwc'][2]
                words[name]+=d['width']*d['groups']
                commands.append(dict(kind='load' if name=='source' else 'store',row=d,channels=c,global_width=local[key[0]]['output_shape_hwc'][1],final=final))
        else:
            width=ranges[name][1]-ranges[name][0]+1;stride=n.get('stride',1)
            max_rows=2 if stride==2 else 4;max_width=160 if stride==2 else 320
            for y in range(task['lo'],task['hi']+1,max_rows):
                rows=min(max_rows,task['hi']-y+1)
                for x in range(0,width,max_width):
                    w=min(max_width,width-x);assert x%4==0
                    parent=n['parents'][0];nn=local[parent]['kind']=='nearest_x2'
                    if n['kind'] in ('dw3x3','conv3x3'):
                        keys=[physical(parent,y*stride-1+i) for i in range((rows-1)*stride+3)]
                    else:keys=[physical(parent,y+i) for i in range(rows)]
                    inputs=[desc(k) for k in keys]
                    outputs=[desc((name,y+i)) for i in range(rows)]
                    for d in outputs:d['base']+=(x//4)*d['groups'];d['width']=w
                    skip=[desc(physical(n['parents'][1],y+i)) for i in range(rows)] if n['kind']=='pw_fused' else []
                    ox=ranges[name][0]+x
                    if skip:
                        for d in skip:d['delta']=ox-d['origin'];assert d['delta']>=0
                    command=dict(kind='compute',node=name,parameter_node=n.get('parameter_node',name),operator=n['kind'],pixels=w,rows=rows,
                                 output_x=ox*stride,source_width=local[parent]['output_shape_hwc'][1],nn=nn,
                                 stride2=stride==2,inputs=inputs,outputs=outputs,skip=skip,
                                 input_groups=max(1,(n['input_shape_hwc'][2]+7)//8),output_groups=outputs[0]['groups'],
                                 mask=(1<<min(8,n['output_shape_hwc'][2]))-1)
                    # The logical nearest tensor is wider than its physical parent.
                    for d in inputs:assert 0<=d['base']<1024 and 0<=d['plane']<3
                    commands.append(command)
                    words[name]+=w*rows*outputs[0]['groups']
        peak=max(peak,arena.peak_words*8)
        for key in task['dependencies']:
            key=tuple(key);used.add(key)
        for key in list(arena.rows):
            if last[key]==task['id']:arena.release(key)
    assert not arena.rows and born==used
    return dict(group=group['name'],repeat=repeat,tile=tile['tile_index'],block_rows=block,
                finite_high_water=arena.high,peak_live_padded_bytes=peak,word_counts=dict(words),commands=commands)


def build(block=4,adaptive=False,optimized=False,pipeline=False,compact_lifetime=False):
    plan=compile_contract(load_contract());manifest=hw.build_tile_manifest(320);tiles=[]
    rejected=[];selected={}
    for group in manifest['groups']:
        repeat=-1 if group['name']=='residual' else 0
        if optimized:
            # Select widths only after real per-plane allocation; never silently
            # degrade the four-row contract to one/two rows to fit memory.
            candidates=(256,160,128,80,64) if group['name']=='encoder' else (256,160,128,80,64) if group['name']=='decoder' else ((80,72,64,40,32) if compact_lifetime else (64,40,32))
            lead=4 if pipeline else 0;retain=pipeline and group['name']!='residual'
            pattern=(2,0,1,0) if compact_lifetime and group['name']=='residual' else None
            if pattern and pipeline:lead=2
            for width in candidates:
                try:
                    if group['name']=='residual':
                        candidate_group=dict(group,tiles=[dict(tile_index=i,commit_interval=dict(x0=x,x1=min(x+width,320)-1)) for i,x in enumerate(range(0,320,width))])
                    else:candidate_group=next(g for g in hw.build_tile_manifest(width)['groups'] if g['name']==group['name'])
                    aligned=group['name']!='residual'
                    part=[compile_tile(plan,candidate_group,tile,repeat,4,aligned=aligned,plane_shift=True,prefetch_rows=lead,retain_commit=retain,compact_lifetime=compact_lifetime,plane_pattern=pattern) for tile in candidate_group['tiles']]
                    tiles.extend(part);selected[group['name']]=dict(block=4,output_tile_width=width,aligned=aligned,plane_shift=True,prefetch_rows=lead,retain_commit=retain,compact_lifetime=compact_lifetime,plane_pattern=pattern);break
                except MemoryError as error:
                    rejected.append(dict(group=group['name'],width=width,error=str(error)))
            else:raise MemoryError(('no aligned four-row allocation',group['name']))
            continue
        for candidate in ([4,2,1] if adaptive else [block]):
            try:
                part=[compile_tile(plan,group,tile,repeat,candidate) for tile in group['tiles']]
                tiles.extend(part);selected[group['name']]=candidate;break
            except MemoryError as error:
                rejected.append(dict(group=group['name'],block=candidate,error=str(error)))
                if not adaptive or candidate==1:raise
    kinds=Counter(c['kind'] for t in tiles for c in t['commands'])
    return dict(status='FINITE_LIVE_FRAME_PROGRAM_NOT_TIMING',block_rows=block,tiles=tiles,commands=dict(kinds),
                selected_block_rows=selected,rejected_capacity=rejected,
                measured_frame_ms=None,rtl_executed=False,full_frame_30fps=False,
                evidence={str(p):sha(p) for p in (Path(__file__),HERE/'compile_frame.py',ROOT/'network_ab/model_b_contract.json',ROOT/'network_ab/hardware_schedule_model.py')})


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--block',type=int,default=4);p.add_argument('--adaptive',action='store_true');p.add_argument('--optimized',action='store_true');p.add_argument('--pipeline',action='store_true');p.add_argument('--compact-lifetime',action='store_true');a=p.parse_args()
    assert not a.output.exists();r=build(a.block,a.adaptive,a.optimized,a.pipeline,a.compact_lifetime);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,separators=(',',':')),encoding='utf-8')
    print(json.dumps(dict(status=r['status'],block=r['selected_block_rows'],rejected=r['rejected_capacity'],commands=r['commands'],peak=max(t['peak_live_padded_bytes'] for t in r['tiles']),high=[max(t['finite_high_water'][i] for t in r['tiles']) for i in range(3)])))


if __name__=='__main__':main()
