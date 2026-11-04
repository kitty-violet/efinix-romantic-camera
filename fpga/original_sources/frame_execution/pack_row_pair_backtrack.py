"""Exact bounded search over static allocations; validates all inclusive lifetimes."""
import argparse,json,time
from pathlib import Path
from screen_row_pair_layout import problem
from model_contract import compile_contract,load_contract
from fourway_frame_program import hw
from run_units import sha

def solve(records,bank,limit=1024):
 keys=sorted((k for k,r in records.items() if r['bank']==bank),key=lambda k:(records[k]['born'],-records[k]['length']))
 placed={};nodes=0;started=time.monotonic();dead=set()
 # Explicit search stack: global production lifetimes exceed Windows' native
 # Python recursion stack even with sys.setrecursionlimit raised.
 stack=[];index=0;active=[];passed=False
 while True:
  if index==len(keys):passed=True;break
  if nodes>200000 or time.monotonic()-started>20:break
  k=keys[index];r=records[k]
  if len(stack)==index:
   nodes+=1;active=[p for p in active if records[p]['last']>=r['born']]
   state=(index,tuple(sorted((p,placed[p]) for p in active)));options=[]
   if state not in dead:
    gaps=[];at=0
    for lo,hi in sorted((placed[p],placed[p]+records[p]['length']) for p in active)+[(limit,limit)]:
     if lo-at>=r['length']:gaps.append((lo-at,at,lo-r['length']))
     at=hi
    for _,lo,hi in sorted(gaps):options.extend(dict.fromkeys((lo,hi)))
   stack.append([active,state,options,0])
  frame=stack[index]
  if frame[3]<len(frame[2]):
   placed[k]=frame[2][frame[3]];frame[3]+=1
   active=frame[0]+[k];index+=1
  else:
   placed.pop(k,None);dead.add(frame[1]);stack.pop()
   if index==0:break
   index-=1
 if passed:
  for i,k in enumerate(keys):
   r=records[k];assert 0<=placed[k]<=limit-r['length']
   for p in keys[i+1:]:
    q=records[p]
    if q['born']>r['last']:break
    if q['last']>=r['born']:assert placed[k]+r['length']<=placed[p] or placed[p]+q['length']<=placed[k]
 return dict(bank=bank,passed=passed,nodes=nodes,seconds=time.monotonic()-started,
  allocations=[dict(node=k[0],row=k[1],base=placed[k],**records[k]) for k in keys] if passed else [])

def main():
 p=argparse.ArgumentParser();p.add_argument('--screen',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 screen=json.loads(a.screen.read_text());offsets=next(c['offsets'] for c in screen['cases'] if c['group']=='residual')
 group=next(g for g in hw.build_tile_manifest(80)['groups'] if g['name']=='residual');group=dict(group,tiles=[dict(tile_index=i,commit_interval=dict(x0=x,x1=min(x+80,320)-1)) for i,x in enumerate(range(0,320,80))])
 plan=compile_contract(load_contract());results=[]
 # Interior tile is the hardest. Boundary tiles are checked independently.
 for tile in [group['tiles'][i] for i in (1,0,2,3)]:
  local,tasks,records=problem(plan,group,tile)
  for (n,y),r in records.items():r['bank']=(y+offsets[n])%6
  rows=[]
  for bank in range(6):
   r=solve(records,bank);rows.append(r);print(dict(tile=tile['tile_index'],**{k:v for k,v in r.items() if k!='allocations'}),flush=True)
   if not r['passed']:break
  results.append(dict(tile=tile['tile_index'],banks=rows))
  if len(rows)!=6 or not all(r['passed'] for r in rows):break
 passed=len(results)==4 and all(len(t['banks'])==6 and all(r['passed'] for r in t['banks']) for t in results)
 result=dict(status='ROW_PAIR_EXACT_PLACEMENT_PASS' if passed else 'ROW_PAIR_EXACT_SEARCH_EXHAUSTED',offsets=offsets,tiles=results,script_sha256=sha(__file__),screen_sha256=sha(a.screen),performance_pass=False)
 a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
