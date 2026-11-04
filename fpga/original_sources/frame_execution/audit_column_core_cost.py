"""Account measured isolated residual-core commands without inventing a frame time."""
import json
from pathlib import Path
from run_units import ROOT,HERE,sha
files={k:ROOT/p for k,p in {
 'dw':'output/toolchain/dw_fused_bounded_command_cost_2026-09-30.json',
 'pw':'tmp/frame-execution/pw-column-fusion2/result.json',
 'residual':'output/toolchain/residual_real_row_cost_audit_2026-09-30.json',
 'floor':'output/toolchain/frame_arithmetic_lower_bound_2026-09-30.json'}.items()}
j={k:json.loads(p.read_text()) for k,p in files.items()}
dw=j['dw']['measured_isolated_command_weighted_cycles'];pw=j['pw']['weighted_isolated_cycles'];add=j['residual']['isolated_residual_cycles']
assert (dw,pw,add)==(585536,1552036,474840)
covered=dw+pw+add;other=sum(n['ideal_issue_cycles'] for n in j['floor']['nodes'] if not n['node'].startswith(('res1_','res2_','res3_')))
assert other==2149200
result=dict(status='MEASURED_ISOLATED_RESIDUAL_CORE_COMMAND_COST_NOT_FULL_FRAME',
 components_cycles=dict(dw=dw,pw=pw,residual=add),covered_core_isolated_cycles=covered,covered_core_isolated_ms=covered/150000,
 other_nodes_still_arithmetic_floor_cycles=other,conditional_sum_cycles=covered+other,conditional_sum_ms=(covered+other)/150000,
 remaining_cycles_for_all_uncharged_work=5000000-covered-other,remaining_ms=(5000000-covered-other)/150000,
 measured_frame_ms=None,full_frame_30fps=False,final_resource_fit=False,
 uncharged_not_zero=['all normal inter-layer mode/parameter/header transitions',
 'stem/stride2/NN-dependent high-resolution DW and non-core PW real supply/tails/halo',
 'actual allocation bases, shared-memory mode multiplexing and row lifetime interactions',
 'DDR fill/commit, burst/refresh stalls and camera/display/Sapphire competition',
 'gesture inference/control and final successful DDR B/frame publication'],
 next_constraint='Use the already verified resident five-bank parameter image; do not copy128 weight words at every PW command.',
 evidence={str(p):sha(p) for p in [Path(__file__),*files.values()]})
out=ROOT/'output/toolchain/column_core_measured_cost_boundary_2026-09-30.json';assert not out.exists()
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ('evidence','uncharged_not_zero')}),flush=True)
