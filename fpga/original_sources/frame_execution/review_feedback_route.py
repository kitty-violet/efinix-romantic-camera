"""Record the specific reviewed placement and its bounded route decision."""
import json
from pathlib import Path
from run_units import ROOT,sha

def main():
    base=ROOT/'tmp/frame-execution';physical=base/'feedback-select-place1';board=base/'connected-board-feedback-select1'
    data=json.loads((physical/'placement_review_inputs.json').read_text());current=data['current']
    rows=current['setup_clock_relationships'];assert len(rows)==22 and min(r['slack_ns'] for r in rows)==0.523
    compute=[r for r in rows if r['launch']==r['capture']=='process_clk_150'];assert len(compute)==1 and compute[0]['slack_ns']==0.828
    assert current['resources']['XLRs']['used']==57902
    serial=base/'feedback-select-mapped1-serial/new';shorts=base/'feedback-select-mapped1'
    evidence=[physical/'placement_review_inputs.json',physical/'cached_head_placement_review.json',
        physical/'physical_inputs.json',physical/'cnn_ram_actual_locations.json',
        physical/'outflow/ti60f225_oob.place.timing.rpt',
        board/'local_decode_map_audit.json',board/'cached_head_map_audit.json',board/'feedback_selection_map_audit.json',
        base/'feedback-select-shorts1/result.json',base/'feedback-select-shorts1/inputs.json',
        base/'feedback-select-cluster1/result.json',base/'feedback-select-cluster1/inputs.json',
        base/'feedback-select-control-equivalence1/result.json',base/'feedback-select-ofp3-rtl1/result.json',
        shorts/'startup/result.json',shorts/'observe/result.json',serial/'result.json',serial/'inputs.json',
        serial/'progress_counter_checks.json',serial/'paired_reply_result.json',Path(__file__)]
    for name in ('local_decode_map_audit.json','cached_head_map_audit.json','feedback_selection_map_audit.json'):
        assert json.loads((board/name).read_text())['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
    target=physical/'route_review.json';assert not target.exists()
    result=dict(decision='PROCEED_TO_BOUNDED_ROUTE',physical=str(physical),
        placement_review_sha256=sha(physical/'placement_review_inputs.json'),
        rationale='Legal native C2/seed3 placement;57902/60800 XLR,228/256 RAM,160/160 DSP. Both targeted data dependencies are cut by verified real register boundaries. All 22 setup estimates positive, minimum +0.523ns,150MHz +0.828ns versus original C2 +0.321ns and cache-only -0.103ns. 1/16 congestion improves 4.073% versus original C2; ordinary cells stay free,113 CNN RAM anchors verified. All affected same-map numeric/control/serial/reset tests pass. These concrete improvements justify one bounded route despite wider-region cost, not a guarantee.',
        remaining_risks=[
            'Versus original C2:weighted wirelength +5.280%,4x4 peak +0.655%,1/4 peak +3.415%.',
            'Versus last passing implementation:weighted wirelength +3.309%,1/4 peak +5.056%;4x4 and1/16 slightly improve.',
            '854/710-load control nets remain; unrouted hold estimate -0.902ns requires actual route repair.',
            'Placement estimates do not prove final STA. Real board control and DDR calibration still require supervised tests.'
        ],budget_seconds=1500,route_attempts_authorized=1,
        budget_reason='Original C2 legal route took971s; this candidate has2001 extra packed XLR and5.28% more weighted wirelength. Allow one25min route, independent host limit1545s, no blind retry.',
        source_and_constraints_unchanged=True,required_evidence={str(p):sha(p) for p in evidence},
        hardware_access=False,board_release=False)
    target.write_text(json.dumps(result,indent=2));print(result['decision'],sha(target))

if __name__=='__main__':main()
