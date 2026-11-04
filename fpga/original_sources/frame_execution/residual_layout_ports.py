"""Joint residual bank placement and in-place alignment lifetime check.
No new physical ports: three row planes x four columns x two shared R/W ports.
"""
import argparse,json
from collections import Counter
from pathlib import Path
from residual_fusion import tasks_for
from compile_frame import Arena,ROOT,HERE,sha
from model_contract import compile_contract,load_contract
class ColoredArena(Arena):
 def __init__(self,depth,colors):
  super().__init__(depth);self.colors=colors
 def allocate(self,key,width,channels):
  plane=(key[1]+self.colors[key[0]])%3;length=(width+3)//4*((channels+7)//8);at=0
  for lo,hi,_ in sorted(self.used[plane]):
   if lo-at>=length:break
   at=max(at,hi)
  if at+length>self.depth:raise MemoryError((key,plane,length,at,self.depth))
  self.used[plane].append((at,at+length,key));self.rows[key]=(plane,at,width,(channels+7)//8)
  self.high[plane]=max(self.high[plane],at+length)
  self.peak_words=max(self.peak_words,4*sum(hi-lo for rows in self.used for lo,hi,_ in rows))
def run(plan,width,block):
 peaks=[0]*3;peak_live=0;inplace=0;packets=0;naive_conflicts=0;traffic=Counter();jobs=Counter();traces=[]
 for x0 in range(0,320,width):
  local,ranges,tasks=tasks_for(plan,1,3,x0,min(319,x0+width-1),block)
  colors={'source':0}
  for name,n in local.items():
   if name=='source':continue
   parents=n['parents']
   if n['kind']=='residual_add':
    used={colors[k] for k in parents};assert len(used)==2
    colors[name]=next(c for c in range(3) if c not in used)
   else:colors[name]=(colors[parents[0]]+1)%3
  last={tuple(k):t['id'] for t in tasks for k in t['dependencies']};arena=ColoredArena(1024,colors);trace=[]
  for t in tasks:
   assert all(tuple(k) in arena.rows for k in t['dependencies'])
   try:
    for name,y in t['outputs']:
     lo,hi=ranges[name];arena.allocate((name,y),hi-lo+1,32)
   except MemoryError as e:return dict(status='REJECTED_FINITE_CAPACITY',width=width,block=block,detail=str(e))
   for intervals in arena.used:
    ordered=sorted(intervals);assert all(a[1]<=b[0] for a,b in zip(ordered,ordered[1:]))
   if t['kind'] in ('source','commit'):
    keys=t['outputs'] if t['kind']=='source' else t['dependencies']
    traffic[t['kind']]+=sum(((ranges[k[0]][1]-ranges[k[0]][0]+1)*32+15)//16*16 for k in keys)
   else:
    a,b=ranges[t['node']];jobs[t['kind']]+=(b-a+1)*(t['hi']-t['lo']+1)*4
   if t['kind']=='residual_add':
    n=local[t['node']];main,skip=n['parents']
    for y in range(t['lo'],t['hi']+1):
     # Each of the two possible scaled branches can be rewritten only after
     # all convolution consumers have already executed; add is its last use.
     for name in (main,skip):
      assert last[(name,y)]==t['id'],('unsafe_inplace',name,y,t['id'],last[(name,y)])
      inplace+=1
     rows=[arena.rows[(name,y)] for name in (main,skip,t['node'])]
     assert len({r[0] for r in rows})==3
     out_lo,out_hi=ranges[t['node']]
     for px in range(out_lo,out_hi+1,4):
      for g in range(4):
       counts=Counter()
       for c in range(min(4,out_hi-px+1)):
        for name in (main,skip,t['node']):
         # Each tensor's halo can change x%4. Count real independent bank
         # addresses rather than assuming global pixel column means same bank.
         lx=px+c-ranges[name][0];bank,address=arena.address((name,y),lx,g)
         assert address<1024;counts[bank]+=1
       assert max(counts.values())<=1
       packets+=1;naive_conflicts+=1
    trace.append(dict(task=t['id'],node=t['node'],lo=t['lo'],hi=t['hi'],
     read_descriptors={str(k):arena.rows[tuple(k)] for k in t['dependencies']},
     write_descriptors={str(k):arena.rows[k] for k in t['outputs']}))
   for k in t['dependencies']:
    k=tuple(k)
    if last[k]==t['id']:arena.release(k)
  assert not arena.rows
  peaks=[max(x,y) for x,y in zip(peaks,arena.high)];peak_live=max(peak_live,arena.peak_words*8)
  traces.append(dict(x0=x0,colors=colors,ranges=ranges,residual_tasks=trace))
 return dict(status='COLORED_RESIDUAL_LIFETIME_AND_PORT_ALLOCATION_PASS',width=width,block=block,depth=1024,
  high_water_per_plane=peaks,peak_live_bytes=peak_live,traffic=dict(traffic),oc8_jobs=dict(jobs),
  branch_row_last_use_checks=inplace,actual_add_packets=packets,add_max_words_per_bank_per_packet=1,
  physical_arena_ram10=84,trace=traces,read_data_and_write_ack_rtl=False,performance_pass=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();a.root.mkdir(parents=True,exist_ok=False)
 plan=compile_contract(load_contract());cases=[]
 for width in (160,80,64,40):
  for block in (1,2,4):
   r=run(plan,width,block);trace=r.pop('trace',None)
   if trace is not None:
    path=a.root/f'width{width}-rows{block}.json';path.write_text(json.dumps(trace,separators=(',',':')))
    r.update(trace_path=str(path.resolve()),trace_sha256=sha(path))
   cases.append(r);print(json.dumps(r),flush=True)
 result=dict(status='RESIDUAL_SHARED_ARENA_JOINT_DESIGN_CHECK',cases=cases,hardware_integrated=False,full_frame_30fps=False,
  evidence={str(f):sha(f) for f in (Path(__file__),HERE/'residual_fusion.py',HERE/'compile_frame.py',ROOT/'network_ab/model_b_contract.json')})
 (a.root/'result.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
