"""Current shared-core cost boundary; never identifies a weighted sum as a frame."""
import json
from pathlib import Path
from run_units import ROOT,sha
paths={k:ROOT/p for k,p in {
 'shared':'tmp/frame-execution/compute-nn-fusion3/result.json',
 'map':'tmp/frame-execution/compute-nn-map3/result.json',
 'residual':'output/toolchain/residual_real_row_cost_audit_2026-09-30.json',
 'floor':'output/toolchain/frame_arithmetic_lower_bound_2026-09-30.json'}.items()}
j={k:json.loads(p.read_text()) for k,p in paths.items()}
assert j['shared']['status']=='LOCAL_SHARED_CLUSTER_FUSION_REPRESENTATIVES_PASS'
dw,pw=j['shared']['weighted_cycles']['dw'],j['shared']['weighted_cycles']['pw'];res=j['residual']['isolated_residual_cycles']
core=dw+pw+res
other=sum(n['ideal_issue_cycles'] for n in j['floor']['nodes'] if not n['node'].startswith(('res1_','res2_','res3_')))
assert (dw,pw,res,other)==(537680,1557516,474840,2149200)
r=j['map']['resources'];assert (r['DSP24'],r['DSP48'],r['RAM10'])==(256,32,24)
combined=r['LUT4']+r['ADD']+r['SRL8']
result=dict(status='SHARED_RESIDENT_DW_PW_PLUS_ISOLATED_RESIDUAL_NOT_FULL_FRAME',
 core_cycles=dict(dw=dw,pw=pw,residual=res),core_isolated_cycles=core,core_isolated_ms=core/150000,
 other_nodes_arithmetic_floor_cycles=other,conditional_sum_cycles=core+other,conditional_sum_ms=(core+other)/150000,
 uncharged_budget_cycles=5000000-core-other,uncharged_budget_ms=(5000000-core-other)/150000,
 shared_cluster_resources=r,shared_cluster_combination_lower_bound=combined,
 physical_dsp_blocks=r['DSP24']//2+r['DSP48'],external_arena_ram=84,external_resident_ram=20,
 measured_frame_ms=None,full_frame_30fps=False,final_resource_fit=False,
 remaining=['Actual residual integration shares the same pool and finite memory, not a second arithmetic pool.',
 'Header prefetch/quantization parameters and all real graph mode transitions.',
 'Stem27 and stride2 real supply; complete non-core scheduling including NN address views.',
 'Live allocator/row lifetime, actual upstream/downstream dependencies and dirty-block suppression.',
 'DDR DMA/refresh/bandwidth/competition, final B response and frame ownership/publication.',
 'Gesture inference/control, evil effect, complete baseline physical gates.'],
 evidence={str(p):sha(p) for p in [Path(__file__),*paths.values()]})
out=ROOT/'output/toolchain/column_core_shared_cost_boundary_2026-09-30.json';assert not out.exists()
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('remaining','evidence')}),flush=True)
