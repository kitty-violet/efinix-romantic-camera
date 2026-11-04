"""Review actual full placement plus exact Gray endpoint rebinding before route."""
import json
from pathlib import Path
from run_units import ROOT,HERE,sha
from finish_native_split_release import BOARD,SHORTS,SERIAL,MAP_AUDIT,warning_audit
from native_split_board import audit_delta
from rebind_native_placement import audit as rebind_audit,SOURCE
import audit_control_state_implementation as implementation

def main():
    physical=ROOT/'tmp/frame-execution/native-split-bound-place1'
    assert not (physical/'route_review.json').exists()
    audit_delta(BOARD);rebind=rebind_audit(physical);warning=warning_audit(BOARD)
    r=json.loads((SHORTS/'result.json').read_text())
    assert r['status']=='NATIVE_SPLIT_SAME_MAP_SHORT_INTEGRATION_PASS'
    assert r['map_sha256']==sha(BOARD/'outflow/ti60f225_oob.map.v')
    for p,h in r['evidence'].items():assert sha(p)==h
    # The copied old report is explicitly retained; record a current binding.
    implementation.audit_delta=audit_delta
    impl=implementation.audit(BOARD,physical)
    ip=physical/'rebound_implementation.json';assert not ip.exists();ip.write_text(json.dumps(impl,indent=2))
    p=physical/'placement_review_inputs.json';d=json.loads(p.read_text());c=d['current']
    assert c['placement_legal'] and c['optimization_setup_reserve_ns']==0
    assert c['resources']=={'XLRs':{'used':55207,'capacity':60800},'Memory Blocks':{'used':228,'capacity':256},'DSP Blocks':{'used':160,'capacity':160}}
    assert c['packing']['COMB4']==c['packing']['COMB4 + FF']==0
    assert len(c['setup_clock_relationships'])==22 and min(x['slack_ns'] for x in c['setup_clock_relationships'])==.257
    assert json.loads((physical/'cnn_ram_actual_locations.json').read_text())['count']==113
    changes={};paths=[p,ip,physical/'physical_inputs.json',physical/'endpoint_rebinding.json',
        physical/'cnn_ram_actual_locations.json',physical/'result.json',physical/'ti60f225_oob.sdc',
        physical/'outflow/ti60f225_oob.place.timing.rpt',BOARD/'result.json',BOARD/'inputs.json',
        BOARD/'delta_audit.json',SHORTS/'result.json',MAP_AUDIT/'result.json',MAP_AUDIT/'held_data_structure.json',
        SERIAL/'result.json',SERIAL/'progress_counter_checks.json',Path(__file__),HERE/'route_native_split_board.py',
        HERE/'rebind_native_placement.py',HERE/'native_bound_cdc.py']
    for name in ('startup-cause-place2','snapshot-arena-free-place1'):
        folder=ROOT/'tmp/frame-execution'/name;ref=folder/'placement_review_inputs.json'
        old=json.loads(ref.read_text())['current'];paths.append(ref)
        changes[name]=dict(weighted_hpwl_pct=100*(c['weighted_hpwl_per_net']/old['weighted_hpwl_per_net']-1),
            congestion_pct={k:100*(v/old['congestion_peaks'][k]-1) for k,v in c['congestion_peaks'].items()})
        for n in ('fixed_dsp.placeloc','ti60f225_oob.peri.xml'):
            if name=='startup-cause-place2':assert sha(physical/n)==sha(folder/n),n
    evidence={str(f):sha(f) for f in paths};evidence.update(rebind['original']);evidence.update(warning['evidence'])
    record=dict(decision='PROCEED_TO_BOUNDED_ROUTE',physical=str(physical),placement_review_sha256=sha(p),
        rationale='The full150MHz video/A original RTL is retained. Only128LOGIC pairs are split into256independentLUTs. Official fresh VDB and65536four-state checks plus same-map control/adapter/serial/reset all pass. Actual placement uses55207XLR,228RAM,160DSP;128COMB4 removed, routing-only slots2762->1961 versus prior passing original.113RAM and wholeDSP cascades retain anchors; ordinary logic free. HPWL and4x4 congestion improve slightly against passing parent, while1/16 and1/4 worsen and setup reserve decreases; record that risk rather than claim guaranteed route. The same placement is retained, with126of140Gray pin names rebound to actual unique FFs; all ten2.500ns constraints and every other SDC byte remain semantically identical. No placement timing PASS is inferred for rebound constraints; this one route plus strict timing and actual70-bit Gray gates decide acceptance.',
        comparisons=changes,shared_controls_review=d['graph']['high_fanout_including_reset'][:12],
        remaining_risks=['Placement hold minimum remains negative and must be repaired by routing.',
            'Medium/coarse congestion is worse than the previously passing parent; legal zero-overlap route is still required.',
            'Exported disconnected input sensitivity is checked; private vendor CSI warnings remain. DDR calibration and board startup are not proven.',
            'Pre-route STA query is unsupported without routing traces. Actual unique placed FF names were rebound; post-route no-empty constraints and70Gray endpoints are mandatory.',
            '5593XLR and28RAM remain, no DSP spare. Remaining full baseline feature budget is not closed.'],
        budget_seconds=900,route_attempts_authorized=1,
        budget_reason='One900s route; original successful parent used461s.1800s whole-job ceiling includes strict STA/CDC. No blind seed sweep.',
        required_evidence=evidence,hardware_access=False,board_release=False)
    (physical/'route_review.json').write_text(json.dumps(record,indent=2));print(record['decision'],json.dumps(changes))

if __name__=='__main__':main()
