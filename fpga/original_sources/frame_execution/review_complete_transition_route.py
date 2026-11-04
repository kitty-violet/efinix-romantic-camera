"""Record inspected native placement tradeoffs before a single bounded route."""
import json
from pathlib import Path
from run_units import ROOT,sha
from complete_control_transition import audit_delta

def main():
 b=ROOT/'tmp/frame-execution';board=b/'connected-board-complete-transition2';physical=b/'complete-transition-place2'
 prior=b/'accepted-tail-place1';short=b/'complete-transition-mapped2';serial=b/'complete-transition-mapped2-serial/new'
 audit_delta(board);p=physical/'placement_review_inputs.json';d=json.loads(p.read_text());c=d['current'];old=json.loads((prior/p.name).read_text())['current']
 assert c['placement_legal'] and c['resources']=={'XLRs':{'used':54200,'capacity':60800},'Memory Blocks':{'used':228,'capacity':256},'DSP Blocks':{'used':160,'capacity':160}}
 assert c['optimization_setup_reserve_ns']==0 and c['canonical_setup_estimate_ns']==.514
 assert len(c['setup_clock_relationships'])==22 and min(x['slack_ns'] for x in c['setup_clock_relationships'])==.514
 assert [x['slack_ns'] for x in c['setup_clock_relationships'] if x['launch']==x['capture']=='process_clk_150']==[.661]
 assert min(x['slack_ns'] for x in c['hold_clock_relationships'])==-.901
 for n in ('ti60f225_oob.sdc','fixed_dsp.placeloc','ti60f225_oob.peri.xml'):assert sha(physical/n)==sha(prior/n)
 change=dict(weighted_hpwl=100*(c['weighted_hpwl_per_net']/old['weighted_hpwl_per_net']-1),
   **{k:100*(v/old['congestion_peaks'][k]-1) for k,v in c['congestion_peaks'].items()})
 evidence=[p,prior/p.name,physical/'physical_inputs.json',physical/'cnn_ram_actual_locations.json',
  physical/'outflow/ti60f225_oob.place.timing.rpt',board/'result.json',board/'inputs.json',board/'delta_audit.json',
  board/'local_decode_map_audit.json',board/'complete_transition_map_audit.json',
  b/'complete-transition-boundary2/result.json',b/'complete-transition-boundary2/inputs.json',b/'complete-transition-boundary2/boolean_result.json',
  b/'complete-transition-short2/result.json',b/'complete-transition-short2/inputs.json',
  b/'complete-transition-equivalence2/result.json',b/'complete-transition-equivalence2/inputs.json',
  b/'complete-transition-ofp3-rtl2/result.json',b/'complete-transition-ofp3-rtl2/inputs.json',
  short/'startup/result.json',short/'observe/result.json',serial/'result.json',serial/'inputs.json',
  serial/'progress_counter_checks.json',serial/'paired_reply_result.json',serial.parent/'evidence_collation.json',
  b/'complete-transition-evidence2/result.json',b/'complete-transition-evidence2/pre_route_implementation.json',
  b/'complete-transition-evidence2/parent_release_binding.json',Path(__file__)]
 assert json.loads((b/'complete-transition-evidence2/result.json').read_text())['status']=='COMPLETE_TRANSITION_SHORTS_PLACED_AWAIT_REVIEW'
 record=dict(decision='PROCEED_TO_BOUNDED_ROUTE',physical=str(physical),placement_review_sha256=sha(p),
  rationale='Localized next-state implementation preserves all source except controller state assignments. Same-map startup, APB/mailbox/error/idle/reset and observation gates passed. Resource54200/60800 XLR,228/256 RAM,160/160 DSP; ordinary logic free. Same SDC, periphery and401 DSP/RAM anchors as the selected passing parent. Against that parent, weighted wire+1.646%,4x4+0.273%,1/16+1.732%,1/4 essentially unchanged; this modest increase is compensated by all22 positive setup estimates, global minimum+.514 vs+.343 and150MHz+.661 vs+.438. Reset fanout685/595 vs680/592 is explicitly retained in review; largest control850 vs854. No claim this guarantees routing. One bounded run will test legality and original setup/hold.',
  parent_changes_pct=change,
  remaining_risks=['Estimated hold minimum-.901ns must be repaired by real route; final WHS>=0 required.',
   'Local congestion/wire slightly worse and system reset fanouts slightly larger; final route may still fail.',
   'Upstream AXI adapter retains one COMB4. Local next-state25 LUTs and APB transition are isolated; hardware cause remains unproven.',
   'DDR calibration and complete video/A startup remain failed/unproven on hardware; no download authorized by this review.',
   '160 DSP has no free sites; CNN-only accepted-A replay54.036ms is unchanged and does not satisfy30fps.'],
  budget_seconds=1500,route_attempts_authorized=1,
  budget_reason='Same passing parent routed in294.484s;1500s tool/1545s host ceiling permits measured variation without unlimited retries.',
  required_evidence={str(f):sha(f) for f in evidence},hardware_access=False,board_release=False)
 out=physical/'route_review.json';assert not out.exists();out.write_text(json.dumps(record,indent=2));print(record['decision'],json.dumps(change))
if __name__=='__main__':main()
