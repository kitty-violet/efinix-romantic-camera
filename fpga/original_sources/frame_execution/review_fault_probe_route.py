"""Inspected full placement review before one bounded route; no hardware."""
import json
from pathlib import Path
from run_units import ROOT,sha
from fault_probe_board import audit_delta
from finish_fault_probe_release import SHORTS,SERIAL,ADAPTER,MAP_AUDIT,RTL
def main():
    b=ROOT/'tmp/frame-execution';board=b/'connected-board-fault-probe1';physical=b/'fault-probe-place1'
    prior=b/'complete-transition-place2';audit_delta(board)
    p=physical/'placement_review_inputs.json';d=json.loads(p.read_text());c=d['current']
    old=json.loads((prior/p.name).read_text())['current']
    assert c['placement_legal'] and c['optimization_setup_reserve_ns']==0
    assert c['resources']=={'XLRs':{'used':54816,'capacity':60800},'Memory Blocks':{'used':228,'capacity':256},'DSP Blocks':{'used':160,'capacity':160}}
    assert len(c['setup_clock_relationships'])==22 and c['canonical_setup_estimate_ns']==.381
    assert [x['slack_ns'] for x in c['setup_clock_relationships'] if x['launch']==x['capture']=='process_clk_150']==[.387]
    assert min(x['slack_ns'] for x in c['hold_clock_relationships'])==-.839
    for n in ('ti60f225_oob.sdc','fixed_dsp.placeloc','ti60f225_oob.peri.xml'):
        assert sha(physical/n)==sha(prior/n),n
    change=dict(weighted_hpwl=100*(c['weighted_hpwl_per_net']/old['weighted_hpwl_per_net']-1),
        **{k:100*(v/old['congestion_peaks'][k]-1) for k,v in c['congestion_peaks'].items()})
    assert json.loads((SHORTS/'result.json').read_text())['status']=='FAULT_PROBE_SAME_MAP_STARTUP_OBSERVE_SERIAL_PASS'
    assert json.loads((physical/'pre_route_implementation.json').read_text())['status']=='CONTROL_STATE_MAPPED_RESET_AND_PERIPHERY_PASS'
    evidence=[p,prior/p.name,physical/'physical_inputs.json',physical/'cnn_ram_actual_locations.json',
        physical/'pre_route_implementation.json',physical/'outflow/ti60f225_oob.place.timing.rpt',
        board/'result.json',board/'inputs.json',board/'delta_audit.json',board/'local_decode_map_audit.json',
        board/'complete_transition_map_audit.json',MAP_AUDIT/'result.json',MAP_AUDIT/'boolean_result.json',
        b/'fault-probe-short2/result.json',b/'fault-probe-short2/inputs.json',RTL/'result.json',RTL/'inputs.json',
        ADAPTER/'result.json',ADAPTER/'inputs.json',SHORTS/'result.json',SHORTS/'startup/result.json',
        SHORTS/'observe/result.json',SERIAL/'result.json',SERIAL/'inputs.json',
        SERIAL/'progress_counter_checks.json',SERIAL/'paired_reply_result.json',Path(__file__)]
    record=dict(decision='PROCEED_TO_BOUNDED_ROUTE',physical=str(physical),placement_review_sha256=sha(p),
        rationale='Observation-only extension; all functional statements and full video/A datapath preserved. Actual map startup/adapter idle/reset, serial and observer gates pass. 54816/60800 XLR,228/256 RAM,160/160 DSP. Same constraints, periphery and DSP/RAM anchors as passing parent; ordinary logic remains free. Versus parent weighted wire and local4x4/1-of16 congestion improve; quarter-device peak increases slightly. All22 estimated setup relations remain positive, but global+.381/process+.387 are lower than parent+.514/+.661. New snapshot capture load761 and related control724 were explicitly reviewed, largest control853 versus850. Reduced total wire and most congestion estimates justify one bounded attempt, without claiming estimated setup/hold or geometric metrics prove routing.',
        parent_changes_pct=change,remaining_risks=[
            'Estimated hold minimum-.839ns requires actual router correction; final original WNS/WHS must both be nonnegative.',
            'New snapshot capture/control high fanout and reduced process setup estimate may hurt final timing; no placement-only release.',
            'Logic exceeds54720 optimization goal by96 but remains5984 below absolute capacity; DSP has no free sites.',
            'This observes adapter error/DDR clocks/reset; hardware fault cause and DDR calibration recovery remain unproven.',
            'Accepted-A datapath throughput unchanged; this is not a30fps result.'],
        budget_seconds=1500,route_attempts_authorized=1,
        budget_reason='Same parent selected route completed with zero overlaps; single1500s tool/1545s host bound, preserve prefix on failure.',
        required_evidence={str(f):sha(f) for f in evidence},hardware_access=False,board_release=False)
    out=physical/'route_review.json';assert not out.exists();out.write_text(json.dumps(record,indent=2))
    print(record['decision'],json.dumps(change))
if __name__=='__main__':main()
