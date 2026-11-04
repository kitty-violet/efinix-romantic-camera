"""Bounded row-pair allocation search, using full graph inclusive lifetimes.

Every row reserves the actual consumer alignment half and padded tail.
No format change, narrower tile, extra RAM port or performance result is assumed.
"""
import argparse,collections,itertools,json,random,time
from pathlib import Path
from fourway_frame_program import graph,row_tasks,hw
from model_contract import compile_contract,load_contract
from screen_row_banked_arena import place
from run_units import ROOT,sha

def problem(plan,group,tile):
 local,ranges,final=graph(plan,group,tile,-1 if group['name']=='residual' else 0)
 tasks=row_tasks(local,final,4,group['name']!='residual',2 if group['name']=='residual' else 4,True)
 last={tuple(k):t['id'] for t in tasks for k in t['dependencies']}
 if group['name']!='residual':
  commits=[t for t in tasks if t['node']=='commit']
  for old,new in zip(commits,commits[1:]):
   for key in old['dependencies']:last[tuple(key)]=new['id']
 records={}
 for t in tasks:
  for name,y in t['outputs']:
   width=ranges[name][1]-ranges[name][0]+1;groups=(local[name]['output_shape_hwc'][2]+7)//8
   # Stride2 requests global odd/even pairs, RGB head even/odd pairs.
   # All other consumers read a single half, so leading padding is unnecessary.
   alignments=set()
   for child in local.values():
    if child['parents'] and child['parents'][0]==name:
     if child.get('stride',1)==2:alignments.add((ranges[name][0]+1)%2)
     elif child['kind'] in ('pw1x1','pw_fused') and child['output_shape_hwc'][2]==3:alignments.add(ranges[name][0]%2)
   assert len(alignments)<=1,('incompatible pair alignment',name,alignments)
   phase=next(iter(alignments),0)
   records[name,y]=dict(born=t['id'],last=last[name,y],length=((width+phase+1)//2)*groups,bank=y%6,phase=phase)
 return local,tasks,records

def signatures(records,names,nt):
 # Repeated vertical schedules have the same live-length signature; evaluate
 # each unique signature once. Removal is after last consumer, never before.
 events=collections.defaultdict(list);byname={n:i for i,n in enumerate(names)}
 for (name,y),r in records.items():
  index=byname[name]*6+y%6
  events[r['born']].append((index,r['length']))
  events[r['last']+1].append((index,-r['length']))
 live=[0]*(len(names)*6);seen=set()
 for t in range(nt+1):
  for i,delta in events[t]:live[i]+=delta
  seen.add(tuple(live))
 return [[(i//6,i%6,value) for i,value in enumerate(s) if value] for s in seen]

def score(sigs,offset,limit=10**9):
 worst=0
 for sig in sigs:
  used=[0]*6
  for i,b,value in sig:used[(b+offset[i])%6]+=value
  worst=max(worst,max(used))
  if worst>limit:return worst
 return worst

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 started=time.monotonic();plan=compile_contract(load_contract());cases=[]
 for gname,width in [('encoder',256),('residual',80),('decoder',256)]:
  group=next(g for g in hw.build_tile_manifest(width)['groups'] if g['name']==gname)
  if gname=='residual':group=dict(group,tiles=[dict(tile_index=i,commit_interval=dict(x0=x,x1=min(x+width,320)-1)) for i,x in enumerate(range(0,320,width))])
  problems=[problem(plan,group,t) for t in group['tiles']];names=list(problems[0][0]);sigs=[]
  for _,tasks,records in problems:sigs+=signatures(records,names,len(tasks))
  # Search all repeating residual four-layer offsets first, then independent
  # offsets with deterministic coordinate descent. No claim of global optimum.
  candidates={};best=10**9
  def add(offset):
   nonlocal best
   s=score(sigs,offset,best+128)
   if s<=best+128:candidates[tuple(offset)]=s
   best=min(best,s)
   return s
  for step in (0,1,2,3,4,5):add([(i*step)%6 for i in range(len(names))])
  if gname=='residual':
   for pat in itertools.product(range(6),repeat=4):
    offset=[0];base=0
    for i in range(1,len(names)):
     offset.append((base+pat[(i-1)%4])%6)
     if i%4==0:base=offset[-1]
    add(offset)
  rng=random.Random(4219)
  for _ in range(40):
   offset=[0]+[rng.randrange(6) for n in names[1:]]
   for repeat in range(3):
    changed=False
    for i in range(1,len(names)):
     old=offset[i];options=[]
     for v in range(6):
      offset[i]=v;options.append((score(sigs,offset),v))
     _,offset[i]=min(options,key=lambda item:(item[0],item[1]!=old,item[1]));changed|=offset[i]!=old
    if not changed:break
   add(offset)
  chosen=None
  for offsets,live_peak in sorted(candidates.items(),key=lambda kv:kv[1])[:48]:
   highs=[]
   for _,tasks,records in problems:
    for (name,y),rec in records.items():rec['bank']=(y+offsets[names.index(name)])%6
    high,fits=place(records,1024);highs.append(high)
   cost=max(max(h) for h in highs)
   if chosen is None or cost<chosen['max_address_end']:
    chosen=dict(offsets=dict(zip(names,offsets)),max_address_end=cost,high_water_per_tile=highs,live_pair_bound=live_peak,capacity_pass=cost<=1024)
   if cost<=1024:break
  cases.append(dict(group=gname,tile_width=width,signature_count=len(sigs),searched=len(candidates),**chosen))
  print(cases[-1],flush=True)
 result=dict(status='ROW_PAIR_PADDED_LIFETIME_SCREEN_NOT_RTL_TIMING',cases=cases,
  all_capacity_pass=all(c['capacity_pass'] for c in cases),bytes=98304,ram10=84,
  consumer_required_leading_alignment_and_tail_padding=True,tile_widths_unchanged=True,
  performance_pass=False,requires_new_descriptor_abi=True,contract_sha256=sha(ROOT/'network_ab/model_b_contract.json'),script_sha256=sha(__file__),seconds=time.monotonic()-started)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
