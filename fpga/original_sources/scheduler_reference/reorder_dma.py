"""Private program-only experiment: reorder independent DMA, never remove hazards.

Keep the complete compute order, exact SRAM RAW/WAR/WAW edges and tile fences.
Tile fences also retain the original inter-tile/inter-layer DDR ordering.
No hardware, coefficient, tensor, or allocation changes. Cost model is ONLY
used to choose a candidate; actual production RTL must validate any speedup.
"""
from pathlib import Path
import collections,heapq,json,time,sys
from compile_schedule import R,accesses
from coefficients import sha

def graph(commands):
 n=len(commands);pred=[set() for _ in commands];succ=[set() for _ in commands]
 writer=[None]*12288;readers=[set() for _ in writer];previous_compute=None;previous_rgb=None
 tile_first={};tile_last={}
 for i,c in enumerate(commands):
  t=c['tile'];tile_first.setdefault(t,i);tile_last[t]=i
  rr,ww,pad=accesses(c);rs={u for u,_ in rr};ws={u for u,_ in ww}|pad
  p=pred[i]
  for u in rs|ws:
   if writer[u] is not None:p.add(writer[u])
  for u in ws:p.update(readers[u])
  for u in ws:writer[u]=i;readers[u].clear()
  for u in rs-ws:readers[u].add(i)
  if c['kind']=='compute':
   if previous_compute is not None:p.add(previous_compute)
   previous_compute=i
  if c['kind']=='store' and c['channels']==3:
   if previous_rgb is not None:p.add(previous_rgb)
   previous_rgb=i
 # Full fences: end sentinel is the final command, which must wait for every
 # command in its tile. All next tile commands wait for previous tile end.
 for t,first in tile_first.items():
  last=tile_last[t]
  pred[last].update(range(first,last))
  if t:
   for i in range(first,last+1):pred[i].add(tile_last[t-1])
 for i,p in enumerate(pred):
  assert all(j<i for j in p)
  for j in p:succ[j].add(i)
 rank=[n]*n;ordinal=0
 for i,c in enumerate(commands):
  if c['kind']=='compute':rank[i]=ordinal;ordinal+=1
 for i in range(n-1,-1,-1):
  if succ[i]:rank[i]=min(rank[i],min(rank[j]for j in succ[i]))
 return pred,succ,rank

def cost(c):
 if c['kind']=='compute':
  avg={'stem':795,'down_dw':449,'down_pw':511,'res1_a_dw':525,'res1_a_pw':1812,
       'res1_b_dw':519,'res1_add':1791,'up1_low_dw':355,'up1_low_pw':286,'up1_high':283,'rgb':280}
  return avg[c['node']]
 return 35+(c['row']['width']*c['channels']+7)//8

def run(commands,pred,succ,rank,reorder):
 left=[len(p)for p in pred];ready=[[],[]];running=[None,None];streams=[[],[]];order=[]
 def push(i):
  e=int(commands[i]['kind']!='compute');key=(rank[i],i)if reorder and e else(i,i)
  heapq.heappush(ready[e],(*key,i))
 for i,n in enumerate(left):
  if not n:push(i)
 tick=0;completed=0;totals=[0]*4
 while completed<len(commands):
  for e in (0,1):
   if running[e] and running[e][0]==tick:
    _,i=running[e];running[e]=None;completed+=1
    for j in succ[i]:
     left[j]-=1
     if not left[j]:push(j)
  for e in (0,1):
   if running[e] is None and ready[e]:
    _,_,i=heapq.heappop(ready[e]);running[e]=(tick+cost(commands[i]),i);streams[e].append(i);order.append(i)
  if completed==len(commands):break
  assert any(running),('DEADLOCK',completed)
  nt=min(a[0]for a in running if a);totals[int(running[1] is not None)+2*int(running[0] is not None)]+=nt-tick;tick=nt
 assert streams[0]==[i for i,c in enumerate(commands)if c['kind']=='compute']
 return order,streams,dict(model_ticks=tick,overlap_bins=totals)

def check(commands):
 owner=[None]*12288;count=0
 for c in commands:
  rr,ww,pad=accesses(c)
  for u,v in rr:assert owner[u]==(c['tile'],*v),('STALE',c['command'],u,owner[u],v);count+=1
  for u in pad:owner[u]=('padding',c['command'])
  for u,v in ww:owner[u]=(c['tile'],*v)
 return count

def main():
 st=time.monotonic();src=R/'data/schedule2';dst=R/'data'/('schedule'+(sys.argv[1]if len(sys.argv)>1 else'4'));dst.mkdir(exist_ok=False)
 raw=(src/'program.json').read_bytes();proof=json.loads((src/'result.json').read_text());assert sha(raw)==proof['program_sha256']
 commands=json.loads(raw);pred,succ,rank=graph(commands)
 # The reference machine has a strictly in-order DMA stream. A ready-only
 # queue would accidentally skip a blocked older DMA and overestimate it.
 bp=[set(p)for p in pred];bs=[set(s)for s in succ];prev=None
 for i,c in enumerate(commands):
  if c['kind']=='compute':continue
  if prev is not None:bp[i].add(prev);bs[prev].add(i)
  prev=i
 _,old_streams,before=run(commands,bp,bs,rank,False)
 order,streams,after=run(commands,pred,succ,rank,True)
 new=[]
 for index,i in enumerate(order):
  c=dict(commands[i]);c['original_command']=i;c['command']=index;new.append(c)
 rgb=lambda rows:[(c['tile'],c['row']['y'],c['row']['origin'])for c in rows if c['kind']=='store'and c['channels']==3]
 assert rgb(new)==rgb(commands),'Production issuer requires ordered RGB publication'
 semantic_reads=check(new);data=json.dumps(new,separators=(',',':')).encode();(dst/'program.json').write_bytes(data)
 proof.update(status='DMA_REORDER_STATIC_HAZARDS_PASS_NOT_RTL_RELEASE',program_sha256=sha(data),parent_schedule_sha256=sha(raw),semantic_reads=semantic_reads,
  hazard_edges=sum(map(len,pred)),before_model=before,after_model=after,dma_order_changed=sum(a!=b for a,b in zip(old_streams[1],streams[1])),
  seconds=time.monotonic()-st,hardware_access=False,throughput_claim=False,
  scope='Exact SRAM dependencies and compute order retained. Full tile fences retain DDR ordering. Timing model is only a selection heuristic; no measured speedup claimed.')
 (dst/'result.json').write_text(json.dumps(proof,indent=2),'utf8');print(json.dumps(proof))
if __name__=='__main__':main()
