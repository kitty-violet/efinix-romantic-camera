"""Freeze actual allocator residual costs without inventing a frame time."""
import json,re
from pathlib import Path
from run_units import ROOT,sha

def main():
 roots=[ROOT/'tmp/frame-execution'/n for n in ('res-actual-native0','res-actual-native1')]
 costs=[];commands=[];evidence={}
 for part,root in enumerate(roots):
  result=json.loads((root/'result.json').read_text())
  manifest=json.loads((root/'inputs.json').read_text())
  assert result['status'] in ('LOCAL_RESIDUAL_COLUMN_NATIVE_SOURCE_PASS','LOCAL_ACTUAL_RESIDUAL_PARTITION_PASS')
  assert sha(root/'inputs.json')==result['inputs_sha256']
  oracle=manifest['oracle'];assert oracle['cases']==540
  seen={}
  for row in result['rows']:
   d={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',row)}
   if not d['random']:
    if d['case'] in seen:assert seen[d['case']]==d
    seen[d['case']]=d
  assert set(seen)==set(range(540))
  for i,d in seen.items():
   assert d['reads']==d['w']*d['rows']*d['groups']*2 and d['writes']==d['acks']==d['reads']//2
   costs.append(d['cycles']);commands.append(dict(part=part,local_case=i,**d))
  for name in ('result.json','inputs.json','simulation.log'):
   f=root/name
   if f.exists():evidence[str(f)]=sha(f)
 assert len(costs)==1080 and sum(costs)==384480
 outer_file=ROOT/'tmp/frame-execution/compute-outer1/result.json'
 outer=json.loads(outer_file.read_text());assert outer['status']=='LOCAL_OUTER_BLOCK_PARTITION_SCREENING_PASS'
 outer_cycles=sum(outer['weighted_cycles'].values());assert outer_cycles==1457768
 dw,pw,still_floor=537680,1557516,874800
 result=dict(status='LOCAL_REAL_ALLOCATOR_RESIDUAL_COST_PASS_NOT_FRAME',actual_commands=1080,
  residual_cycles=sum(costs),residual_ms=sum(costs)/150000,normal_and_random_numeric_scalar_checks=34452480,
  components_cycles=dict(core_dw=dw,core_pw=pw,residual=sum(costs),outer_real_partition=outer_cycles,stem_and_stride2_arithmetic_floor=still_floor),
  conditional_sum_cycles=dw+pw+sum(costs)+outer_cycles+still_floor,
  conditional_sum_ms=(dw+pw+sum(costs)+outer_cycles+still_floor)/150000,
  measured_frame_ms=None,full_frame_30fps=False,final_resource_fit=False,
  uncharged_not_zero=['stem/stride2 real supply overhead above arithmetic floor','real header prefetch and command transitions',
   'one live cross-layer allocator/scheduler including halo/rematerialization and source row lifetime',
   'DDR input/output completion, stalls and camera/display/CPU competition','gesture and effects with successful frame ownership publication'],
  commands=commands,evidence={**evidence,str(outer_file):sha(outer_file),str(Path(__file__)):sha(__file__)})
 output=ROOT/'output/toolchain/shared_residual_actual_cost_2026-09-30.json'
 assert not output.exists()
 output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k not in ('commands','evidence','uncharged_not_zero')},ensure_ascii=False))
if __name__=='__main__':main()
