"""Compile actual residual DAG row lifetimes into the finite12-bank arena.
This produces executable row descriptors and proves allocation/dependencies,
NOT a hardware execution time or bandwidth measurement. No contract changes.
"""
import argparse,json,hashlib,sys
from collections import defaultdict,Counter
from pathlib import Path
from compile_frame import Arena,ROOT,HERE,sha
from model_contract import compile_contract,load_contract
def tasks_for(plan,start,end,x0,x1,block):
 byid={n['id']:n for n in plan['nodes']}
 local={k:dict(n) for k,n in byid.items() if any(k.startswith(f'res{r}_') for r in range(start,end+1))}
 first=next(iter(local.values()));shape=first['input_shape_hwc']
 local={'source':dict(id='source',kind='source',inputs=[],output_shape_hwc=shape),**local}
 for n in local.values():n['parents']=[p if p in local else 'source' for p in n['inputs']]
 final=f'res{end}_add';ranges={final:(x0,x1)}
 for name in reversed(local):
  if name not in ranges:continue
  n=local[name];a,b=ranges[name]
  if n['kind']=='dw3x3':a,b=a-1,b+1
  for parent in n['parents']:
   w=local[parent]['output_shape_hwc'][1];lo,hi=max(0,a),min(w-1,b)
   if parent in ranges:lo,hi=min(lo,ranges[parent][0]),max(hi,ranges[parent][1])
   ranges[parent]=(lo,hi)
 assert set(ranges)==set(local)
 done={k:-1 for k in local};tasks=[]
 def ensure(name,hi):
  n=local[name];hi=min(hi,n['output_shape_hwc'][0]-1);lo=done[name]+1
  if lo>hi:return
  dependencies=[]
  for parent in n['parents']:
   h=local[parent]['output_shape_hwc'][0];a,b=(max(0,lo-1),min(h-1,hi+1)) if n['kind']=='dw3x3' else (lo,hi)
   ensure(parent,b);dependencies.extend((parent,r) for r in range(a,b+1))
  tasks.append(dict(id=len(tasks),node=name,kind=n['kind'],lo=lo,hi=hi,dependencies=sorted(set(dependencies)),outputs=[(name,r) for r in range(lo,hi+1)]))
  done[name]=hi
 h=local[final]['output_shape_hwc'][0]
 for lo in range(0,h,block):
  hi=min(h-1,lo+block-1);ensure(final,hi)
  tasks.append(dict(id=len(tasks),node='commit',kind='commit',lo=lo,hi=hi,dependencies=[(final,r) for r in range(lo,hi+1)],outputs=[]))
 assert all(done[k]==n['output_shape_hwc'][0]-1 for k,n in local.items())
 return local,ranges,tasks
def one(plan,start,end,width,block,depth):
 width_full=320;traces=[];peak=[0]*3;peak_live=0;traffic=Counter();jobs=Counter();n_tasks=0
 for x0 in range(0,width_full,width):
  local,ranges,tasks=tasks_for(plan,start,end,x0,min(width_full-1,x0+width-1),block)
  last={tuple(k):t['id'] for t in tasks for k in t['dependencies']}
  arena=Arena(depth);tile=[];born=set();used=set()
  for t in tasks:
   assert all(tuple(k) in arena.rows for k in t['dependencies'])
   before={str(k):arena.rows[tuple(k)] for k in t['dependencies']}
   try:
    for name,y in t['outputs']:
     assert (name,y) not in born;born.add((name,y))
     w=ranges[name][1]-ranges[name][0]+1;c=local[name]['output_shape_hwc'][2]
     arena.allocate((name,y),w,c)
   except MemoryError as e:return dict(start=start,end=end,width=width,block=block,depth=depth,status='REJECTED_FINITE_CAPACITY',failure=str(e))
   # Independent address bounds and disjoint allocations within every plane.
   for plane in arena.used:
    pairs=sorted(plane)
    assert all(a[1]<=b[0] for a,b in zip(pairs,pairs[1:]))
   for k in t['outputs']:
    plane,base,w,groups=arena.rows[k]
    for x in (0,w-1):
     for g in (0,groups-1):
      bank,address=arena.address(k,x,g);assert bank//4==plane and base<=address<depth
   outputs={str(k):arena.rows[k] for k in t['outputs']}
   if t['kind'] in ('source','commit'):
    keys=t['outputs'] if t['kind']=='source' else t['dependencies']
    traffic[t['kind']]+=sum(((ranges[k[0]][1]-ranges[k[0]][0]+1)*32+15)//16*16 for k in keys)
   else:jobs[t['kind']]+=(ranges[t['node']][1]-ranges[t['node']][0]+1)*(t['hi']-t['lo']+1)*4
   tile.append(dict(**t,read_descriptors=before,write_descriptors=outputs))
   for k in t['dependencies']:
    k=tuple(k);used.add(k)
    if last[k]==t['id']:arena.release(k)
   n_tasks+=1
  assert not arena.rows and born==used
  peak=[max(a,b) for a,b in zip(peak,arena.high)];peak_live=max(peak_live,arena.peak_words*8)
  traces.append(dict(x0=x0,ranges=ranges,tasks=tile))
 return dict(start=start,end=end,width=width,block=block,depth=depth,status='FINITE_ROW_ALLOCATION_AND_DEPENDENCIES_PASS',
  high_water_per_plane=peak,peak_live_padded_bytes=peak_live,traffic=dict(traffic),oc8_jobs=dict(jobs),tasks=n_tasks,
  physical_arena_ram10=84*((depth+1023)//1024),trace=traces)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();a.root.mkdir(parents=True,exist_ok=False)
 plan=compile_contract(load_contract());cases=[]
 for fused in (1,3):
  for width in (320,160,80):
   for block in (1,2,4):
    r=one(plan,1,fused,width,block,1024);trace=r.pop('trace',None)
    if trace is not None:
     f=a.root/f'blocks{fused}-width{width}-rows{block}.json';f.write_text(json.dumps(trace,separators=(',',':')))
     r.update(trace_path=str(f.resolve()),trace_sha256=sha(f))
    cases.append(r)
    print(json.dumps(r),flush=True)
 result=dict(status='RESIDUAL_FUSION_FINITE_STORAGE_SCREEN_ONLY',cases=cases,
  measured_frame_ms=None,performance_pass=False,hardware_controller=False,ports_and_overlap_proven=False,
  evidence={str(p):sha(p) for p in (Path(__file__),HERE/'compile_frame.py',ROOT/'network_ab/model_b_contract.json',ROOT/'network_ab/model_contract.py')})
 (a.root/'result.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
