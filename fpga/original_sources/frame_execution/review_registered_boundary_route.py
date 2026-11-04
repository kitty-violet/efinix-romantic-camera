"""Review actual placement, controls and inserted FFs before one bounded route."""
import json,re
from pathlib import Path
from run_units import ROOT,HERE,sha
from registered_boundary_board import BOARD,audit_delta
from audit_snapshot_cells import cells
import audit_control_state_implementation as implementation

def save(p,d):
    assert not p.exists(),p
    p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf8')

def main():
    b=ROOT/'tmp/frame-execution';physical=b/'split-registered-place1'
    evidence=b/'split-registered-evidence1';audit_delta(BOARD)
    r=json.loads((evidence/'result.json').read_text())
    assert r['status']=='REGISTERED_BOUNDARY_SAME_MAP_SHORT_INTEGRATION_PASS'
    assert r['map_sha256']==sha(BOARD/'outflow/ti60f225_oob.map.v')
    for f,h in r['evidence'].items():assert sha(f)==h,f
    impl=physical/'pre_route_implementation.json'
    implementation.audit_delta=audit_delta
    save(impl,implementation.audit(BOARD,physical))
    p=physical/'placement_review_inputs.json';d=json.loads(p.read_text());c=d['current']
    assert c['placement_legal'] and c['optimization_setup_reserve_ns']==0
    assert c['resources']=={'XLRs':{'used':57496,'capacity':60800},
        'Memory Blocks':{'used':228,'capacity':256},'DSP Blocks':{'used':160,'capacity':160}}
    assert c['packing']['COMB4']==c['packing']['COMB4 + FF']==0
    assert len(c['setup_clock_relationships'])==22
    assert min(x['slack_ns'] for x in c['setup_clock_relationships'])==.335
    assert [x['slack_ns'] for x in c['setup_clock_relationships'] if x['launch']==x['capture']=='process_clk_150']==[.625]
    assert json.loads((physical/'cnn_ram_actual_locations.json').read_text())['count']==113
    cc=cells((BOARD/'outflow/ti60f225_oob.map.v').read_text())
    prefix='video_processing_pipeline/pipeline/cnn/core/dut/engine/'
    groups={}
    for suffix,width in (('rq/selected_feedback',288),('mac/source/back_payload',395),
                         ('mac/column_data',384),('od',768),('ox',27),('mac/column',9),('mac/part',2)):
        names=[prefix+suffix+f'[{i}]~FF' for i in range(width)]
        # Use only known exported register groups. Column/part aliases may
        # change names, so record metadata separately from required data FFs.
        if suffix in ('mac/column','mac/part') and not all(n in cc for n in names):continue
        for n in names:
            k,ports,attrs=cc[n];assert k=='EFX_FF' and ports['CLK']=='process_clk_150',(n,k,ports)
        groups[suffix]=dict(count=width,enable_nets=sorted({cc[n][1].get('CE') for n in names}),
            reset_nets=sorted({cc[n][1].get('SR') for n in names}))
    structure=physical/'registered_boundary_map_structure.json'
    save(structure,dict(status='INSERTED_PAYLOAD_REGISTERS_PRESENT_IN_ACTUAL_MAP',groups=groups,
        map_sha256=sha(BOARD/'outflow/ti60f225_oob.map.v'),
        scope='Required feedback/front/back data FFs exist at 150MHz; not a physical timing proof.'))
    comparisons={};evidence_paths=[p,impl,structure,physical/'physical_inputs.json',physical/'pnr_threads.json',
        physical/'cnn_ram_actual_locations.json',physical/'outflow/ti60f225_oob.place.timing.rpt',
        physical/'result.json',BOARD/'result.json',BOARD/'inputs.json',BOARD/'boundary_binding.json',
        evidence/'result.json',b/'split-registered-placement-checks1/map-audit/result.json',
        b/'split-registered-placement-checks1/map-audit/held_data_structure.json',
        b/'split-registered-adapter1/result.json',b/'split-registered-accepted-cluster1/result.json',
        b/'split-registered-edges-shorts4/result.json',b/'registered-source-rate2/result.json',
        Path(__file__),HERE/'registered_boundary_route.py',HERE/'registered_boundary_board.py']
    for ref in ('split-outputs-place3','startup-cause-place2'):
        folder=b/ref;rp=folder/p.name;old=json.loads(rp.read_text())['current'];evidence_paths.append(rp)
        for n in ('ti60f225_oob.sdc','fixed_dsp.placeloc','ti60f225_oob.peri.xml'):
            assert sha(physical/n)==sha(folder/n),(ref,n)
        comparisons[ref]=dict(weighted_hpwl_pct=100*(c['weighted_hpwl_per_net']/old['weighted_hpwl_per_net']-1),
            congestion_pct={k:100*(v/old['congestion_peaks'][k]-1) for k,v in c['congestion_peaks'].items()})
    review=dict(decision='PROCEED_TO_BOUNDED_ROUTE',physical=str(physical),placement_review_sha256=sha(p),
        rationale='Actual zero-COMB4 full150MHz map now has288 feedback FFs,768 RQ output FFs and384+395 source payload FFs. Four-state RQ cycle-equivalence, source random stalls/cancel plus II1, and actual A24-block adjacent numeric tests pass. Same-map startup/adapter/serial/reset gates pass. This change directly divides the long RQ/SRL and RAM-to-MAC paths identified in the66-negative-endpoint parent sample. DSP cascades,113RAM anchors,pins,clocks and originalSDC remain unchanged; ordinary logic remains free. HPWL and fine-scale congestion worsen, so no improvement is claimed from placement alone: one bounded route will measure whether shorter data cones compensate. All22 setup estimates remain positive; no artificial timing exception.',
        comparisons=comparisons,registered_groups=groups,
        shared_controls_review=dict(largest_fanouts=d['graph']['high_fanout_including_reset'][:12],
            finding='Observation1022/980/897/888 unchanged; CNN wsr852->866, reset660->657; new queue push539/537 fanout needs actual routing. Data FFs have existing local CE and no payload reset; no cross-column hand relocation.'),
        remaining_risks=['Congestion4x4/1-16 increased; legal zero-overlap routing is still required.',
            'Placement hold minimum-0.902ns is uncorrected; original routed WNS/WHS must both be nonnegative.',
            'Remaining19 sampled parent paths and newly transferred critical paths can still fail; no seed sweep is scheduled.',
            '3304XLR and28RAM free; no DSP spare. Remaining baseline feature budget is not closed.',
            'Exact Gray nonempty endpoints and bundled-data<13.332ns must pass after route.',
            'Actual board startup and DDR calibration remain unverified.'],
        budget_seconds=1200,route_attempts_authorized=1,
        budget_reason='8physical cores/50%CPU/6GiB tree;1200s route,300s originalSTA,400sCDC;2100s independent total deadline.',
        required_evidence={str(f):sha(f) for f in evidence_paths},hardware_access=False,board_release=False)
    save(physical/'route_review.json',review)
    print(review['decision'],json.dumps(comparisons))

if __name__=='__main__':main()
