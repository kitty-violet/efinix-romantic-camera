"""Transparent lower bound, NOT an implemented full-frame timing estimate."""
import argparse,json,math,sys
from collections import Counter
from pathlib import Path
from run_units import HERE,ROOT,sha
sys.path.insert(0,str(ROOT/'network_ab'))
from model_contract import load_contract,compile_contract
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 contract=ROOT/'network_ab/model_b_contract.json'
 assert sha(contract)=='5750c4668d75077746ec5d4accf8773ab437540cf933c202f153a8797c21dd01'
 plan=compile_contract(load_contract());rows=[];bykind=Counter();byterms=Counter();mac=0;gaps=0
 for n in plan['nodes']:
  h,w,c=n['output_shape_hwc'];groups=(c+7)//8;jobs=h*w*groups
  if n['kind']=='nearest_x2':terms=0;cycles=0;batch=0
  else:
   terms=9 if n['kind']=='residual_add' else n['terms']
   assert terms in (8,9,16,27,32)
   batch=(jobs+31)//32;cycles=batch*terms;gaps+=batch
  # RQ capacity never exceeded; nine-term tasks use32/9, not4/clock.
  assert not terms or cycles*4>=jobs
  if 'macs' in n:mac+=n['macs']
  bykind[n['kind']]+=cycles;byterms[str(terms)]+=cycles
  rows.append(dict(node=n['id'],kind=n['kind'],shape=n['output_shape_hwc'],oc8_jobs=jobs if terms else 0,
   terms=terms,ideal_issue_cycles=cycles,ideal_issue_ms_at_150mhz=cycles/150000))
 assert len(rows)==29 and mac==plan['macs_per_frame']==966758400
 total=sum(bykind.values());assert total==4114800
 result=dict(status='ARITHMETIC_SLOT_LOWER_BOUND_ONLY',frequency_hz=150000000,frame_deadline_cycles=5000000,
  ideal_main_network_and_residual_cycles=total,ideal_ms=total/150000,
  maximum_remaining_cycles_for_all_unhidden_work=5000000-total,maximum_remaining_ms=(5000000-total)/150000,
  hypothetical_one_bubble_per_dot_batch_cycles=gaps,hypothetical_ms_with_one_bubble=(total+gaps)/150000,
  by_kind_cycles=dict(bykind),by_terms_cycles=dict(byterms),nodes=rows,
  uncounted_not_zero=['finite tile halo and tails','phase startup/drain and slot stalls','layer header/weight switching',
   'real shared RAM read conflicts, read latency and result writes','NN consumer scheduling',
   'DDR requests, responses, efficiency, refresh and competing camera/display/CPU','residual operand supply/table switches',
   'gesture inference and control','bad line/tile/frame recovery','final write B and frame publication'],
  assumptions=['same32-front/four-bank pool shared by PW/DW/stem/residual, never duplicate-booked',
   'all fronts supplied on every arithmetic term clock; results never stall',
   'no tile halo or finite-row launch/drain costs in this lower bound',
   'NN costs no MAC only; real address and memory costs remain uncounted',
   '150MHz is a design clock, not new final-netlist STA proof'],
  measured_frame_ms=None,full_frame_30fps=False,final_resource_fit=False,
  evidence={str(f):sha(f) for f in (contract,ROOT/'network_ab/model_contract.py',Path(__file__))})
 a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:result[k] for k in ('status','ideal_ms','maximum_remaining_ms','hypothetical_ms_with_one_bubble','measured_frame_ms')}))
if __name__=='__main__':main()
