"""Charge real fused task widths/tails; compare row and4-row batch packing.
Uses the validated lifetime allocator/tasks, not the old full-frame screening model.
No new RTL performance claim; RAM return/write overlap and DDR remain unproven.
"""
import argparse,json,math
from collections import Counter
from pathlib import Path
from run_units import HERE,ROOT,sha
from residual_fusion import tasks_for
from model_contract import compile_contract,load_contract
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 plan=compile_contract(load_contract());totals=Counter();shape_counts=Counter();rows=[]
 for x0 in range(0,320,80):
  local,ranges,tasks=tasks_for(plan,1,3,x0,min(319,x0+79),4)
  for t in tasks:
   kind=t['kind']
   if kind not in ('dw3x3','pw1x1'):continue
   w=ranges[t['node']][1]-ranges[t['node']][0]+1;h=t['hi']-t['lo']+1
   g=4;terms=9 if kind=='dw3x3' else 32
   # DW phase generator has8 extra start/drain phase slots, per command/G.
   phase_tail=8 if kind=='dw3x3' else 0
   single=h*((w+31)//32*terms+phase_tail)*g
   packed=((w*h+31)//32*terms+phase_tail)*g
   totals[kind+'_single_row']+=single;totals[kind+'_four_row_packed']+=packed
   totals[kind+'_ideal_no_batch_tail']+=math.ceil(w*h*g*terms/32)
   totals['output_rows']+=h;shape_counts[(kind,w,h)]+=1
   rows.append(dict(task=t['id'],x0=x0,node=t['node'],kind=kind,width=w,height=h,single_row_issue_cycles=single,
     packed_issue_cycles=packed,saved_issue_cycles=single-packed))
 lower=json.loads((ROOT/'output/toolchain/frame_arithmetic_lower_bound_2026-09-30.json').read_text())
 old=sum(n['ideal_issue_cycles'] for n in lower['nodes'] if n['node'].startswith(('res1_conv','res2_conv','res3_conv')))
 assert old==1771200
 # Same real measured residual row charge; conservative serial-row adapter.
 add_counts={80:720,82:360,84:720,88:360};measured={80:213,82:221,84:221,88:230}
 residual_charge=sum(add_counts[w]*measured[w] for w in add_counts)
 rest=lower['ideal_main_network_and_residual_cycles']-old-lower['by_kind_cycles']['residual_add']
 single=rest+residual_charge+totals['dw3x3_single_row']+totals['pw1x1_single_row']
 packed=rest+residual_charge+totals['dw3x3_four_row_packed']+totals['pw1x1_four_row_packed']
 result=dict(status='FROZEN_FUSED_TASK_TAIL_PACKING_DESIGN_AUDIT_NOT_FULL_FRAME',totals=dict(totals),
  shapes=[dict(kind=k[0],width=k[1],rows=k[2],tasks=v) for k,v in sorted(shape_counts.items())],
  other_nodes_still_ideal_cycles=rest,real_residual_isolated_row_cycles=residual_charge,
  single_row_partial_lower_cycles=single,single_row_partial_lower_ms=single/150000,
  four_row_partial_lower_cycles=packed,four_row_partial_lower_ms=packed/150000,
  avoidable_issue_cycles=single-packed,avoidable_issue_ms=(single-packed)/150000,
  remaining_single_row_cycles=5000000-single,remaining_packed_cycles=5000000-packed,
  uncounted_not_zero=['PW fill/drain and RAM return stalls','DW actual physical bank read conflicts/returns and write contention',
   'non-residual tiles/starts/tails','DDR/camera/display/CPU','gesture','final B and publication'],
  measured_frame_ms=None,full_frame_30fps=False,packed_production_rtl=False,rows=rows,
  evidence={str(f):sha(f) for f in (Path(__file__),HERE/'residual_fusion.py',ROOT/'network_ab/model_b_contract.json',
   ROOT/'tmp/frame-execution/residual-row-mux-profile1/result.json')})
 a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k not in ('rows','evidence','shapes')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

