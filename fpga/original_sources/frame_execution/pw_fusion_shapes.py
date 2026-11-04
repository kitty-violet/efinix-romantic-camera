"""Frozen real fusion PW commands; no substitution of a32-pixel ideal cost."""
import collections,json
from pathlib import Path
from run_units import HERE,ROOT,sha
from residual_fusion import tasks_for
from model_contract import compile_contract,load_contract
from validate_pw_column import vectors as make_vectors
def enumerate_shapes():
 plan=compile_contract(load_contract());counts=collections.Counter();commands=[]
 for x0 in range(0,320,80):
  local,ranges,tasks=tasks_for(plan,1,3,x0,min(319,x0+79),4)
  for t in tasks:
   if t['kind']!='pw1x1':continue
   ox,xend=ranges[t['node']];w=xend-ox+1
   for y in range(t['lo'],t['hi']+1,4):
    h=min(4,t['hi']-y+1);shape=(w,h,4,4,255,ox,3+y%3)
    counts[shape]+=1;commands.append(dict(tile=x0,task=t['id'],node=t['node'],y=y,shape=shape))
 return sorted(counts),counts,commands
def vectors(root):
 shapes,counts,commands=enumerate_shapes();result=make_vectors(root,shapes=shapes)
 result['counts']=[counts[s] for s in shapes];result['real_commands']=len(commands)
 (root/'fusion_commands.json').write_text(json.dumps(dict(shapes=shapes,counts=result['counts'],commands=commands,isolated_cost_only=True),indent=2))
 return result
