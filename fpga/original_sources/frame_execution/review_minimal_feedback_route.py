"""Bind the inspected C2 minimal-feedback placement to one bounded route."""
import json
from pathlib import Path
from run_units import ROOT, sha
from minimal_feedback_board import audit_delta

def main():
    base=ROOT/'tmp/frame-execution'
    physical=base/'minimal-feedback-place2';board=base/'connected-board-minimal-feedback1'
    delta=audit_delta(board)
    review=json.loads((physical/'minimal_feedback_placement_review.json').read_text())
    current=review['current'];rows=current['setup_clock_relationships']
    assert len(rows)==22 and min(r['slack_ns'] for r in rows)==0.302
    compute=[r for r in rows if r['launch']==r['capture']=='process_clk_150']
    assert len(compute)==1 and compute[0]['slack_ns']==0.668
    assert current['resources']['XLRs']['used']==56501
    assert current['resources']['Memory Blocks']['used']==228
    assert current['resources']['DSP Blocks']['used']==160
    assert current['placement_legal'] and current['optimization_setup_reserve_ns']==0
    serial=base/'minimal-feedback-mapped1-serial/new';shorts=base/'minimal-feedback-mapped1'
    evidence=[physical/'placement_review_inputs.json',physical/'minimal_feedback_placement_review.json',
        physical/'physical_inputs.json',physical/'cnn_ram_actual_locations.json',
        physical/'outflow/ti60f225_oob.place.timing.rpt',
        board/'local_decode_map_audit.json',board/'feedback_selection_map_audit.json',board/'delta_audit.json',
        base/'minimal-feedback-shorts1/result.json',base/'minimal-feedback-shorts1/inputs.json',
        base/'minimal-feedback-cluster1/result.json',base/'minimal-feedback-cluster1/inputs.json',
        base/'minimal-feedback-control-equivalence1/result.json',base/'minimal-feedback-ofp3-rtl1/result.json',
        shorts/'startup/result.json',shorts/'observe/result.json',serial/'result.json',serial/'inputs.json',
        serial/'progress_counter_checks.json',serial/'paired_reply_result.json',
        Path(__file__),Path(__file__).with_name('review_minimal_feedback_place.py')]
    for name in ('local_decode_map_audit.json','feedback_selection_map_audit.json'):
        assert json.loads((board/name).read_text())['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
    assert delta==json.loads((board/'delta_audit.json').read_text())
    target=physical/'route_review.json';assert not target.exists()
    result=dict(decision='PROCEED_TO_BOUNDED_ROUTE',physical=str(physical),
        placement_review_sha256=sha(physical/'placement_review_inputs.json'),
        rationale='Legal native C2/seed3,56501/60800 XLR,228/256 RAM,160/160 DSP. No cached heads: original queues are byte-identical. Registered job-kind feedback steering passes same-cycle four-state/numeric/cluster tests and mapped control idle/reset checks. Versus original C2: weighted wire -1.285%,4x4 peak -2.592%,1/16 peak -9.849%; compute placement estimate improves +0.321 to +0.668ns. Versus rejected cached-feedback version:1401 fewer XLR,weighted wire -6.235%,all three congestion peaks improve. All22 setup estimates positive,minimum+0.302ns. Original DSP and113 CNN RAM anchors verified; ordinary logic remains free. These measured global improvements justify one route; they do not prove STA.',
        remaining_risks=[
            'Original-queue variable taps remain; removal from reported top10 is not proof all such paths pass routed STA.',
            'Current worst estimate rgb_y[2] to DMA beat_index[5]/CE has26 levels and+0.668ns; real routing may consume that margin.',
            '1/4 peak remains+1.553% versus original C2,+3.164% versus the old passing implementation;850/710-load nets remain.',
            'Unrouted hold estimate minimum-0.901ns requires real route repair; full canonical STA and functional CDC/Gray/PLL checks remain mandatory.',
            'No real-board control or DDR calibration result exists for this version.'
        ],budget_seconds=1500,route_attempts_authorized=1,
        budget_reason='Original C2 legal route971s,cached-feedback896s. New footprint and wire lower than cached-feedback; one bounded25min route (independent1545s process-tree limit) accommodates repair without blind retries.',
        source_and_constraints_unchanged=True,required_evidence={str(p):sha(p) for p in evidence},
        hardware_access=False,board_release=False)
    target.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(result['decision'],sha(target))

if __name__=='__main__':main()
