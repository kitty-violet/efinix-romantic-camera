"""Reject the measured filtered-stride2 partition, retaining all missing costs."""
import json
from pathlib import Path
from run_units import ROOT,sha
def main():
 f=ROOT/'tmp/frame-execution/compute-front1/result.json'
 front=json.loads(f.read_text());assert front['status']=='LOCAL_FRONT_PARTITION_SCREENING_PASS'
 r=ROOT/'output/toolchain/shared_residual_actual_cost_2026-09-30.json'
 old=json.loads(r.read_text())
 parts=dict(old['components_cycles']);del parts['stem_and_stride2_arithmetic_floor'];parts.update(front['weighted_cycles'])
 total=sum(parts.values());assert total==5175749
 result=dict(status='FILTERED_STRIDE2_PARTITION_THROUGHPUT_REJECTED',components_cycles=parts,
  partition_cycles=total,partition_ms=total/150000,deadline_cycles=5000000,over_budget_cycles=total-5000000,
  over_budget_ms=(total-5000000)/150000,measured_frame_ms=None,full_frame_30fps=False,
  reason='Normal column feed filters odd rows/columns, leaving most MAC owners idle. Numeric correctness is PASS, this throughput route is FAIL.',
  uncharged_not_zero=old['uncharged_not_zero'][1:],evidence={str(p):sha(p) for p in (Path(__file__),f,r)})
 out=ROOT/'output/toolchain/filtered_stride2_throughput_rejection_2026-09-30.json';assert not out.exists()
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
