"""Necessary live-memory and arithmetic bounds; not a timing/allocator PASS."""
import argparse, collections, json, time
from pathlib import Path
from fourway_frame_program import graph, row_tasks, hw
from model_contract import compile_contract, load_contract
from run_units import ROOT, sha

def case(plan, group, width):
    if group['name']=='residual':
        group=dict(group,tiles=[dict(tile_index=i,commit_interval=dict(x0=x,x1=min(x+width,320)-1)) for i,x in enumerate(range(0,320,width))])
    max_live=0; commands=0; vectors=0; load_bytes=0; store_bytes=0; node_work=collections.Counter()
    for tile in group['tiles']:
        local,ranges,final=graph(plan,group,tile,-1 if group['name']=='residual' else 0)
        tasks=row_tasks(local,final,4,group['name']!='residual',2 if group['name']=='residual' else 4,True)
        last={tuple(k):t['id'] for t in tasks for k in t['dependencies']}
        if group['name']!='residual':
            commits=[t for t in tasks if t['node']=='commit']
            for old,new in zip(commits,commits[1:]):
                for key in old['dependencies']:last[tuple(key)]=new['id']
        events=collections.Counter()
        for t in tasks:
            name=t['node']
            if name=='commit':
                w=tile['commit_interval']['x1']-tile['commit_interval']['x0']+1
                store_bytes+=w*(t['hi']-t['lo']+1)*local[final]['output_shape_hwc'][2]
            elif name=='source':
                w=ranges[name][1]-ranges[name][0]+1
                load_bytes+=w*(t['hi']-t['lo']+1)*local[name]['output_shape_hwc'][2]
            else:
                n=local[name];w=ranges[name][1]-ranges[name][0]+1;out_groups=(n['output_shape_hwc'][2]+7)//8
                blocks=(t['hi']-t['lo']+4)//4
                terms=27 if n['kind']=='conv3x3' else 9 if n['kind']=='dw3x3' else n['input_channels']
                phases=(terms+8)//9 if n['kind']=='conv3x3' else (terms+7)//8 if n['kind'] in ('pw1x1','pw_fused') else 1
                work=blocks*out_groups*w*max(phases,3 if n['kind']=='pw_fused' else 1)
                commands+=blocks;vectors+=blocks*out_groups*w;node_work[name]+=work
            for name,y in t['outputs']:
                w=ranges[name][1]-ranges[name][0]+1;g=(local[name]['output_shape_hwc'][2]+7)//8
                phases=set()
                for child in local.values():
                    if child.get('parents') and child['parents'][0]==name:
                        if child.get('stride',1)==2:phases.add((ranges[name][0]+1)%2)
                        elif child['kind'] in ('pw1x1','pw_fused') and child['output_shape_hwc'][2]==3:phases.add(ranges[name][0]%2)
                assert len(phases)<=1
                size=((w+next(iter(phases),0)+1)//2)*g*16
                events[t['id']]+=size;events[last[name,y]+1]-=size
        live=0
        for _,delta in sorted(events.items()):live+=delta;max_live=max(max_live,live)
        assert live==0
    return dict(group=group['name'],width=width,tiles=len(group['tiles']),compute_commands=commands,
                max_live_bytes_necessary=max_live,necessary_capacity_pass=max_live<=98304,
                arithmetic_cycle_lower_bound=sum(node_work.values()),node_work=dict(node_work),
                output_vectors=vectors,load_bytes=load_bytes,store_bytes=store_bytes,
                per_bank_placement_proven=False,actual_command_timeline_proven=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    start=time.monotonic();plan=compile_contract(load_contract());cases=[]
    for name,widths in [('encoder',(256,320,512,640,1280)),('residual',(80,96,128,160,320)),('decoder',(256,320,512,640,1280))]:
        for w in widths:
            try:group=next(g for g in hw.build_tile_manifest(320 if name=='residual' else w)['groups'] if g['name']==name)
            except ValueError as error:
                cases.append(dict(group=name,width=w,rejected=str(error)));continue
            c=case(plan,group,w);cases.append(c);print(json.dumps(c),flush=True)
    a.output.write_text(json.dumps(dict(status='NECESSARY_BOUND_ONLY_NOT_RTL_OR_ALLOCATION_PASS',cases=cases,seconds=time.monotonic()-start,script_sha256=sha(__file__),contract_sha256=sha(ROOT/'network_ab/model_b_contract.json')),indent=2)+'\n')
if __name__=='__main__':main()
