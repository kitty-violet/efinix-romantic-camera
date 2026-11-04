"""Concrete review of the inspected OFP5 placement, before one bounded route."""
import json
from pathlib import Path
from run_units import ROOT,sha
from startup_cause_board import audit_delta
from finish_startup_cause_release import SHORTS,SERIAL,ADAPTER,MAP_AUDIT,RTL
def main():
 b=ROOT/'tmp/frame-execution';board=b/'connected-board-startup-cause4';physical=b/'startup-cause-place2'
 prior=b/'fault-probe-place1';audit_delta(board)
 p=physical/'placement_review_inputs.json';d=json.loads(p.read_text());c=d['current']
 old=json.loads((prior/p.name).read_text())['current']
 assert c['placement_legal'] and c['optimization_setup_reserve_ns']==0
 assert c['resources']=={'XLRs':{'used':55737,'capacity':60800},'Memory Blocks':{'used':228,'capacity':256},'DSP Blocks':{'used':160,'capacity':160}}
 assert len(c['setup_clock_relationships'])==22 and c['canonical_setup_estimate_ns']==.406
 assert [x['slack_ns'] for x in c['setup_clock_relationships'] if x['launch']==x['capture']=='process_clk_150']==[.526]
 assert min(x['slack_ns'] for x in c['hold_clock_relationships'])==-.814
 assert d['graph']['high_fanout_including_reset'][0]['distinct_sink_cells']==1022
 for n in ('ti60f225_oob.sdc','fixed_dsp.placeloc','ti60f225_oob.peri.xml'):
  assert sha(physical/n)==sha(prior/n),n
 change=dict(weighted_hpwl=100*(c['weighted_hpwl_per_net']/old['weighted_hpwl_per_net']-1),
  **{k:100*(v/old['congestion_peaks'][k]-1) for k,v in c['congestion_peaks'].items()})
 assert json.loads((SHORTS/'result.json').read_text())['status']=='OFP5_SAME_MAP_STARTUP_OBSERVE_SERIAL_ADAPTER_PASS'
 assert json.loads((physical/'pre_route_implementation.json').read_text())['status']=='CONTROL_STATE_MAPPED_RESET_AND_PERIPHERY_PASS'
 evidence=[p,prior/p.name,physical/'physical_inputs.json',physical/'cnn_ram_actual_locations.json',
  physical/'pre_route_implementation.json',physical/'outflow/ti60f225_oob.place.timing.rpt',
  board/'result.json',board/'inputs.json',board/'delta_audit.json',board/'local_decode_map_audit.json',
  board/'complete_transition_map_audit.json',MAP_AUDIT/'result.json',MAP_AUDIT/'boolean_result.json',
  MAP_AUDIT/'new_diagnostic_cells.json',MAP_AUDIT/'held_data_structure.json',
  b/'startup-cause-shorts4/result.json',b/'startup-cause-shorts4/inputs.json',RTL/'result.json',RTL/'inputs.json',
  ADAPTER/'result.json',ADAPTER/'inputs.json',SHORTS/'result.json',SHORTS/'startup/result.json',
  SHORTS/'observe/result.json',SERIAL/'result.json',SERIAL/'inputs.json',
  SERIAL/'progress_counter_checks.json',SERIAL/'paired_reply_result.json',Path(__file__)]
 record=dict(decision='PROCEED_TO_BOUNDED_ROUTE',physical=str(physical),placement_review_sha256=sha(p),
  rationale='Full150MHz video/A implementation retained; same SDC, pins, DSP cascades and113 CNN RAM anchors as passing OFP4, ordinary logic free. Own actual map/control/adapter/reset/serial gates pass. 55737/60800 XLR,228/256 RAM,160/160 DSP. Relative to immediate passing parent, wirelength +0.1205%, congestion peaks4x4 +2.77%,1/16 +1.40%,1/4 +0.37% are worse and explicitly retained as risks. Compensation: all22 estimated setup relations positive, minimum improves+.381 to+.406ns, process+.387 to+.526ns; total wire estimate nearly unchanged and routing-only packing slots2922 to2762. New snapshot control fans1022/980 versus761/724 span most device, while functional CNN largest853 to857 is nearly unchanged. Finite read-only capture has no functional feedback. These support one bounded router attempt with strict final gates, not a prediction of guaranteed success.',
  parent_changes_pct=change,remaining_risks=[
   'Hold estimate minimum-.814ns must be repaired by the router; original WNS and WHS must both pass.',
   'New1026-bit snapshot controls1022/980 sinks increase global load; no acceptance based only on setup estimate or legal coordinates.',
   'Remaining5063XLR,28RAM and0DSP; no additional baseline-function resource budget claimed.',
   'New held DDR observation crosses to150MHz: exact endpoints,3-stage done gate and routed settling<13.332ns remain mandatory.',
   'Physical board root cause, DDR calibration and video operation remain unproven.'],
  budget_seconds=1500,route_attempts_authorized=1,
  budget_reason='One1500s tool/1545s host attempt; passing parent needed461.219s. No unbounded reruns.',
  required_evidence={str(f):sha(f) for f in evidence},hardware_access=False,board_release=False)
 out=physical/'route_review.json';assert not out.exists();out.write_text(json.dumps(record,indent=2))
 print(record['decision'],json.dumps(change))
if __name__=='__main__':main()
