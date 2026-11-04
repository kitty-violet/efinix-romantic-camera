"""Bounded layout experiment: improve address packing, never relocate live rows."""
import argparse,json,random,time
from pathlib import Path
from screen_row_pair_layout import problem
from model_contract import compile_contract,load_contract
from fourway_frame_program import hw
from run_units import ROOT,sha

def pack(records,limit,order):
 placed={};high=[0]*6
 for key in order:
  r=records[key];at=0
  intervals=sorted((placed[p],placed[p]+records[p]['length']) for p in r['neighbors'] if p in placed)
  for lo,hi in intervals:
   if lo-at>=r['length']:break
   at=max(at,hi)
  if at+r['length']>limit:return None
  placed[key]=at;high[r['bank']]=max(high[r['bank']],at+r['length'])
 for k,r in records.items():
  for p in r['neighbors']:
   assert placed[k]+r['length']<=placed[p] or placed[p]+records[p]['length']<=placed[k]
 return high,placed

def neighbors(records,offsets):
 active=[set() for _ in range(6)]
 for (n,y),r in records.items():r['bank']=(y+offsets[n])%6;r['neighbors']=set()
 for key,r in sorted(records.items(),key=lambda kv:kv[1]['born']):
  b=r['bank'];active[b]={k for k in active[b] if records[k]['last']>=r['born']}
  r['neighbors'].update(active[b])
  for k in active[b]:records[k]['neighbors'].add(key)
  active[b].add(key)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--screen',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 start=time.monotonic();screen=json.loads(a.screen.read_text());offsets=next(c['offsets'] for c in screen['cases'] if c['group']=='residual')
 group=next(g for g in hw.build_tile_manifest(80)['groups'] if g['name']=='residual');group=dict(group,tiles=[dict(tile_index=i,commit_interval=dict(x0=x,x1=min(x+80,320)-1)) for i,x in enumerate(range(0,320,80))])
 plan=compile_contract(load_contract());results=[];rng=random.Random(14917)
 for tile in group['tiles']:
  local,tasks,records=problem(plan,group,tile);neighbors(records,offsets);found=None
  for trial in range(240):
   # Randomized priorities explore static placement only. Every returned
   # assignment is independently checked over inclusive producer/consumer life.
   jitter={k:rng.random() for k in records}
   mode=trial%8
   def priority(k):
    r=records[k];degree=len(r['neighbors']);length=r['length'];life=r['last']-r['born']
    if mode==0:return (-length,r['born']+jitter[k]*40)
    if mode==1:return (-degree,-length,jitter[k])
    if mode==2:return (-length*degree*(1+jitter[k]*.25),r['born'])
    if mode==3:return (-life,-length,jitter[k])
    if mode==4:return (r['born']+jitter[k]*80,-length)
    if mode==5:return (-length*(1+jitter[k]*.10),r['born'])
    if mode==6:return (-degree*(1+jitter[k]*.50),-length,r['born'])
    return (-length*(1+jitter[k]),r['born'])
   found=pack(records,1024,sorted(records,key=priority))
   if found:break
   if time.monotonic()-start>180:break
  result=dict(tile=tile['tile_index'],trials=trial+1,pass_capacity=found is not None)
  if found:
   high,placed=found;result['high_water']=high
   result['allocations']=[dict(node=k[0],row=k[1],base=placed[k],**{f:r[f] for f in ('bank','length','born','last')}) for k,r in records.items()]
  results.append(result);print({k:v for k,v in result.items() if k!='allocations'},flush=True)
  if not found:break
 out=dict(status='STATIC_ROW_PAIR_PLACEMENT_PASS' if len(results)==4 and all(r['pass_capacity'] for r in results) else 'STATIC_ROW_PAIR_PLACEMENT_NOT_FOUND',offsets=offsets,tiles=results,seconds=time.monotonic()-start,
  no_live_relocation=True,all_padding_charged=True,no_rtl_or_timing_claim=True,script_sha256=sha(__file__),screen_sha256=sha(a.screen))
 a.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
