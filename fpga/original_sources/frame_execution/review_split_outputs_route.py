"""Concrete review of the completed independent-output placement, not a sweep."""
import json
from pathlib import Path
from run_units import ROOT,HERE,sha
from split_logic_outputs_board import audit_delta
from finish_split_logic_outputs_release import SHORTS,SERIAL,ADAPTER,MAP_AUDIT,RTL

def main():
    b=ROOT/'tmp/frame-execution';board=b/'connected-board-split-outputs1'
    physical=b/'split-outputs-place3';prior=b/'startup-cause-place2'
    audit_delta(board)
    p=physical/'placement_review_inputs.json';d=json.loads(p.read_text());c=d['current']
    old=json.loads((prior/p.name).read_text())['current']
    assert c['placement_legal'] and c['optimization_setup_reserve_ns']==0
    assert c['resources']=={'XLRs':{'used':56128,'capacity':60800},'Memory Blocks':{'used':228,'capacity':256},'DSP Blocks':{'used':160,'capacity':160}}
    assert c['packing']['COMB4']==c['packing']['COMB4 + FF']==0
    assert len(c['setup_clock_relationships'])==22 and c['canonical_setup_estimate_ns']==.423
    assert min(x['slack_ns'] for x in c['setup_clock_relationships'])==.423
    assert [x['slack_ns'] for x in c['setup_clock_relationships'] if x['launch']==x['capture']=='process_clk_150']==[.640]
    assert min(x['slack_ns'] for x in c['hold_clock_relationships'])==-.901
    assert [x['distinct_sink_cells'] for x in d['graph']['high_fanout_including_reset'][:5]]==[1022,980,897,888,852]
    for n in ('ti60f225_oob.sdc','fixed_dsp.placeloc','ti60f225_oob.peri.xml'):
        assert sha(physical/n)==sha(prior/n),n
    assert json.loads((physical/'cnn_ram_actual_locations.json').read_text())['count']==113
    assert json.loads((SHORTS/'result.json').read_text())['status']=='SPLIT_OUTPUTS_SAME_MAP_SHORT_INTEGRATION_PASS'
    assert json.loads((physical/'pre_route_implementation.json').read_text())['status']=='CONTROL_STATE_MAPPED_RESET_AND_PERIPHERY_PASS'
    change=dict(weighted_hpwl=100*(c['weighted_hpwl_per_net']/old['weighted_hpwl_per_net']-1),
        **{k:100*(v/old['congestion_peaks'][k]-1) for k,v in c['congestion_peaks'].items()})
    evidence=[p,prior/p.name,physical/'physical_inputs.json',physical/'pnr_threads.json',
        physical/'cnn_ram_actual_locations.json',physical/'pre_route_implementation.json',
        physical/'outflow/ti60f225_oob.place.timing.rpt',physical/'result.json',
        board/'result.json',board/'inputs.json',board/'delta_audit.json',
        board/'local_decode_map_audit.json',board/'complete_transition_map_audit.json',
        MAP_AUDIT/'result.json',MAP_AUDIT/'boolean_result.json',MAP_AUDIT/'new_diagnostic_cells.json',
        MAP_AUDIT/'held_data_structure.json',b/'split-outputs-shorts2/result.json',
        b/'split-outputs-shorts2/inputs.json',b/'split-outputs-shorts2/split_wrapper_proof.json',
        RTL/'result.json',RTL/'inputs.json',ADAPTER/'result.json',ADAPTER/'inputs.json',
        SHORTS/'result.json',SHORTS/'startup/result.json',SHORTS/'observe/result.json',
        SERIAL/'result.json',SERIAL/'inputs.json',SERIAL/'progress_counter_checks.json',
        SERIAL/'paired_reply_result.json',Path(__file__),
        HERE/'audit_split_outputs_implementation.py',HERE/'balanced_offline_job.py',
        HERE/'windows_physical_cores_job.py',HERE/'place_split_outputs_eight_threads.py',
        HERE/'place_split_outputs_four_threads.py',
        b/'split-outputs-place3-eight-thread-source.py',b/'split-outputs-place3-eight-thread-source.json',
        b/'split-outputs-place3-eight-thread-wrapper.py',b/'split-outputs-place3-eight-thread-wrapper.json',
        b/'split-outputs-place3-place-driver.py',b/'split-outputs-place3-place-driver.json',
        ROOT/'tmp/board-test/bounded-split-outputs-place3/result.json']
    record=dict(decision='PROCEED_TO_BOUNDED_ROUTE',physical=str(physical),placement_review_sha256=sha(p),
        rationale='Whole-board independent outputs remove the repeated board P=O dependency using official packing controls; full150MHz video/A and protected IP retained. Own map/serial/adapter/reset gates pass. Original SDC/pins/DSP cascades/113 CNN RAM anchors are byte-identical to passing parent; normal logic remains free. 56128/60800XLR (+391),228/256RAM,160/160DSP. Weighted HPWL67.3113 to68.1874 (+1.30%); peak congestion4x4 2.10118 to2.15442 (+2.53%),1/16 1.61016 to1.71936 (+6.78%),1/4 1.32990 to1.37353 (+3.28%) worsen and are retained risks. Compensating evidence: all22 setup estimates remain positive, global minimum .406 to.423ns, process .526 to.640ns; packing routing-only cells2762 to1877; largest snapshot fanouts1022/980 unchanged, additional controls848/840 to897/888, CNN852 versus857 and reset660 versus656 remain comparable. This supports one bounded routing attempt with mandatory original STA/CDC, not guaranteed success or a fixed congestion threshold.',
        parent_changes_pct=change,remaining_risks=[
            'Congestion increases may prevent legal routing; no relaxation of zero-conflict requirement.',
            'Hold estimate -.901ns versus parent-.814ns must be repaired during routing; original WNS/WHS both >=0.',
            'Global snapshot/control networks remain high fanout; no blanket coordinate locking or false paths.',
            '4672 XLR and28 RAM remain, no DSP spare; remaining baseline feature budget not claimed complete.',
            'Exact Gray endpoints and held-data settling below13.332ns, PLL/reset audits remain mandatory.',
            'Actual board P output and DDR calibration remain unverified for this candidate.'],
        budget_seconds=1200,route_attempts_authorized=1,
        budget_reason='User-approved eight physical cores/50%CPU with4GiB memory and6GiB reserve. Route1200s plus bounded canonical STA and CDC within1800s overall; no automatic retries.',
        required_evidence={str(f):sha(f) for f in evidence},hardware_access=False,board_release=False)
    out=physical/'route_review.json';assert not out.exists()
    out.write_text(json.dumps(record,indent=2),encoding='utf8')
    print(record['decision'],json.dumps(change))

if __name__=='__main__':main()
