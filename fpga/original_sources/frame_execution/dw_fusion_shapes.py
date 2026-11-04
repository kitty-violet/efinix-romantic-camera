"""Enumerate every residual-fusion DW command, splitting hardware row limit4.
Representatives preserve width, x position, vertical boundary/clamp and row color.
Only isolated engine costs: live allocator placement and whole-frame overlap OPEN.
"""
import argparse,collections,json
from pathlib import Path
from run_units import HERE,ROOT,sha
from residual_fusion import tasks_for
from model_contract import compile_contract,load_contract
from validate_dw_block import vectors as make_vectors
def enumerate_shapes():
 plan=compile_contract(load_contract());counts=collections.Counter();commands=[]
 for x0 in range(0,320,80):
  local,ranges,tasks=tasks_for(plan,1,3,x0,min(319,x0+79),4)
  for t in tasks:
   if t['kind']!='dw3x3':continue
   ox,xend=ranges[t['node']];w=xend-ox+1
   for y in range(t['lo'],t['hi']+1,4):
    h=min(4,t['hi']-y+1)
    # Last possible 4-row command starts176; oy175 or less has no lower clamp.
    # Interior choices keep actual row-color modulo3. Upper clamp needs oy0.
    representative_y=y if y==0 or y+h>=179 else 3+y%3
    shape=(w,h,4,ox,representative_y);counts[shape]+=1
    commands.append(dict(tile=x0,task=t['id'],node=t['node'],y=y,shape=shape))
 return sorted(counts),counts,commands
def vectors(root):
 shapes,counts,commands=enumerate_shapes();result=make_vectors(root,shapes=shapes)
 result['counts']=[counts[s] for s in shapes];result['real_commands']=len(commands)
 record=dict(shapes=shapes,counts=result['counts'],commands=commands,isolated_cost_only=True,
  sources={str(f):sha(f) for f in (Path(__file__),HERE/'residual_fusion.py',ROOT/'network_ab/model_b_contract.json')})
 (root/'fusion_commands.json').write_text(json.dumps(record,indent=2))
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 shapes,counts,commands=enumerate_shapes()
 r=dict(status='FUSED_DW_COMMAND_SHAPES',shapes=[dict(shape=s,count=counts[s]) for s in shapes],commands=len(commands),
  rows=sum(s[1]*counts[s] for s in shapes),measured_frame_ms=None,full_frame_30fps=False)
 if a.native:
  import re
  m=json.loads((a.native/'inputs.json').read_text());result=json.loads((a.native/'result.json').read_text());assert m['oracle']['shapes']==[list(s) for s in shapes]
  measured={};detail=[]
  for line in result['runs'][0]['rows']:
   d={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)}
   if not d['random']:
    s=shapes[d['case']];assert (d['width'],d['rows'],d['groups'])==s[:3]
    measured[s]=d['cycles'];detail.append(dict(shape=s,count=counts[s],isolated_cycles=d['cycles'],weighted_cycles=counts[s]*d['cycles']))
  assert len(measured)==len(shapes);r['measured_isolated_command_weighted_cycles']=sum(counts[s]*measured[s] for s in shapes)
  r['measured_isolated_command_weighted_ms']=r['measured_isolated_command_weighted_cycles']/150000
  r['detail']=detail;r['native_sha256']=sha(a.native/'result.json')
 a.output.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ('shapes','detail')}),flush=True)
if __name__=='__main__':main()
