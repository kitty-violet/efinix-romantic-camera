"""Joint finite arena capacity and multirow batch-tail search. Design screening only."""
import argparse,json
from pathlib import Path
from collections import Counter
from run_units import HERE,ROOT,sha
from residual_layout_ports import run as placement
from residual_fusion import tasks_for
from model_contract import compile_contract,load_contract
def cost(plan,width,block):
 c=Counter()
 for x0 in range(0,320,width):
  _,ranges,tasks=tasks_for(plan,1,3,x0,min(319,x0+width-1),block)
  for t in tasks:
   kind=t['kind']
   if kind not in ('dw3x3','pw1x1','residual_add'):continue
   w=ranges[t['node']][1]-ranges[t['node']][0]+1;h=t['hi']-t['lo']+1
   if kind=='residual_add':
    # Existing one-row adapter cost, exact observed formula only for supported
    # widths; other widths are clearly an unmeasured screening projection.
    packets=(w+3)//4*4;c['residual_row_projection']+=(2*packets+packets//8+43)*h
   else:
    terms=9 if kind=='dw3x3' else 32
    c[kind+'_packed_issue']+=((w*h+31)//32*terms+(8 if kind=='dw3x3' else 0))*4
    c[kind+'_row_issue']+=h*((w+31)//32*terms+(8 if kind=='dw3x3' else 0))*4
 return dict(c)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();a.root.mkdir(parents=True,exist_ok=False)
 plan=compile_contract(load_contract());cases=[]
 for width in (80,64,48,40,32):
  for block in (4,6,8,12,16):
   placed=placement(plan,width,block);placed.pop('trace',None)
   row=dict(width=width,block=block,placement=placed)
   if placed['status']=='COLORED_RESIDUAL_LIFETIME_AND_PORT_ALLOCATION_PASS':
    c=cost(plan,width,block);row['cost']=c
    row['residual_region_projected_cycles']=c['dw3x3_packed_issue']+c['pw1x1_packed_issue']+c['residual_row_projection']
   cases.append(row);print(json.dumps({k:v for k,v in row.items() if k!='placement'}|{'capacity':placed['status']},ensure_ascii=False),flush=True)
 feasible=[c for c in cases if 'cost' in c];ranked=sorted(feasible,key=lambda c:c['residual_region_projected_cycles'])
 result=dict(status='JOINT_CAPACITY_BATCH_PACKING_SCREEN_NOT_RTL_PASS',cases=cases,ranked_candidates=ranked,
  excludes_not_zero=['real memory return latency and write contention','PW fill/drain and weight service','DDR and system competition','gesture and final publish'],
  measured_frame_ms=None,full_frame_30fps=False,
  evidence={str(f):sha(f) for f in (Path(__file__),HERE/'residual_layout_ports.py',HERE/'residual_fusion.py',ROOT/'network_ab/model_b_contract.json')})
 (a.root/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()

