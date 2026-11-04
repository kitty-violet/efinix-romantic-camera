"""Bind the inspected reduced-footprint layout and all current short evidence."""
import json
from pathlib import Path
from run_units import ROOT,sha
from accepted_tail_board import audit_delta

def main():
 base=ROOT/'tmp/frame-execution';physical=base/'accepted-tail-place1';board=base/'connected-board-accepted-tail2'
 audit_delta(board);p=physical/'placement_review_inputs.json';data=json.loads(p.read_text());c=data['current']
 assert c['resources']['XLRs']['used']==54139 and c['resources']['Memory Blocks']['used']==228 and c['resources']['DSP Blocks']['used']==160
 assert c['placement_legal'] and c['optimization_setup_reserve_ns']==0 and c['canonical_setup_estimate_ns']==0.343
 rows=c['setup_clock_relationships'];assert len(rows)==22 and min(r['slack_ns'] for r in rows)>0
 assert [r['slack_ns'] for r in rows if r['launch']==r['capture']=='process_clk_150']==[0.438]
 assert data['changes_pct']['weighted_hpwl'] < -4.5
 assert all(data['changes_pct'][k]<0 for k in ('4x4','1/16','1/4'))
 shorts=base/'accepted-tail-mapped1';serial=base/'accepted-tail-mapped1-serial/new';full=base/'accepted-tail-frame-full1'
 evidence=[p,physical/'physical_inputs.json',physical/'cnn_ram_actual_locations.json',physical/'outflow/ti60f225_oob.place.timing.rpt',
  board/'result.json',board/'inputs.json',board/'delta_audit.json',board/'local_decode_map_audit.json',board/'feedback_selection_map_audit.json',
  base/'accepted-tail-shorts2/result.json',base/'accepted-tail-shorts2/inputs.json',base/'accepted-tail-control-equivalence1/result.json',
  base/'accepted-tail-ofp3-rtl1/result.json',shorts/'startup/result.json',shorts/'observe/result.json',
  serial/'result.json',serial/'inputs.json',serial/'progress_counter_checks.json',serial/'paired_reply_result.json',
  full/'result.json',full/'inputs.json',base/'minimal-feedback-place2/placement_review_inputs.json',Path(__file__)]
 for f in (full/'result.json',serial/'result.json'):
  r=json.loads(f.read_text());assert r['inputs_sha256']==sha(f.parent/'inputs.json')
 assert json.loads((full/'result.json').read_text())['status']=='REGISTERED_ACCEPTED_A_FULL_PASS'
 assert json.loads((full/'inputs.json').read_text())['board_inputs_sha256']==sha(board/'inputs.json')
 # The first controller subcommand failed before creating its output directory.
 # Actual replacement tests passed; preserve the original orchestration failure.
 recovery=physical/'mapped_recovery.json';assert not recovery.exists()
 recovery.write_text(json.dumps(dict(status='MAPPED_TESTS_COMPLETED_AFTER_DIRECTORY_PREPARATION_FIX',
  original_failure=str(base/'accepted-tail-implementation1/failure.json'),original_failure_sha256=sha(base/'accepted-tail-implementation1/failure.json'),
  source_changed=False,tests={str(f):sha(f) for f in (shorts/'startup/result.json',shorts/'observe/result.json',serial/'result.json')},
  reason='startup driver could not be created because parent output directory was absent; no simulator or hardware ran in that failed subcommand.'),indent=2))
 evidence.append(recovery)
 result=dict(decision='PROCEED_TO_BOUNDED_ROUTE',physical=str(physical),placement_review_sha256=sha(p),
  rationale='Exact accepted-A tail shift specialization removes1433 LUT4,1376 ADD,228 FF and48 SRL8 from exported map. Native C2/seed3 placement uses54139/60800 XLR,228/256 RAM,160/160 DSP;401 hard-column/RAM anchors verified and ordinary logic free. Compared with previous passing implementation, weighted wire is4.551% lower and all three congestion peaks are lower. Against the immediate failed C2 parent, wire falls1.464% and1/4 peak falls4.188%, though local4x4/1/16 peaks rise2.791%/5.702%. Extra2362 free XLR, removal of the measured worst generic multiplication cone, positive22-clock setup estimates, full accepted-A29-node numerical replay and same-map control recovery proofs justify one bounded route. This does not establish STA.',
  remaining_risks=[
   'Compute placement estimate+0.438ns is lower than parent+0.668ns. Other long data/control nets can still fail after routing.',
   'Local peaks rise relative to immediate parent; remaining854/710-load controls and reset networks still require real routing.',
   'Pre-route hold minimum-0.839ns must be repaired by actual routing. Required final original WNS/WHS>=0, no fake exceptions.',
   'A tail is restricted to0/4/8 with checked load rejection; arbitrary B1 weights are not supported by this selected profile.',
   'CNN cycle count remains8105408=54.036ms@150MHz in behavioral-memory replay, not30fps or whole-board latency. Current priority is board bring-up.',
   'Actual board startup, DDR calibration and video remain unproven.'
  ],budget_seconds=1500,route_attempts_authorized=1,
  budget_reason='Previous same-family C2 routes took879-971s; use bounded1500s tool/1545s host with smaller footprint, not unlimited retry.',
  required_evidence={str(f):sha(f) for f in evidence},hardware_access=False,board_release=False)
 dst=physical/'route_review.json';assert not dst.exists();dst.write_text(json.dumps(result,indent=2));print(result['decision'])
if __name__=='__main__':main()
