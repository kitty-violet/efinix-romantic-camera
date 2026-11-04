"""Whole-chip C2 comparison for the minimal-selector version, no auto route."""
import argparse,json
from pathlib import Path
from run_units import ROOT,sha
from minimal_feedback_board import audit_delta

def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',type=Path,required=True);a=p.parse_args()
    physical=a.physical.resolve();dst=physical/'minimal_feedback_placement_review.json';assert not dst.exists()
    meta=json.loads((physical/'physical_inputs.json').read_text());board=Path(meta['base']);delta=audit_delta(board)
    data=json.loads((physical/'placement_review_inputs.json').read_text());current=data['current']
    old_path=ROOT/'tmp/frame-execution/local-decode-place1/placement_review_inputs.json'
    cached_path=ROOT/'tmp/frame-execution/feedback-select-place1/placement_review_inputs.json'
    old=json.loads(old_path.read_text())['current'];cached=json.loads(cached_path.read_text())['current']
    def comparison(ref):
        return dict(xlr=current['resources']['XLRs']['used']-ref['resources']['XLRs']['used'],
            weighted_hpwl_pct=100*(current['weighted_hpwl_per_net']/ref['weighted_hpwl_per_net']-1),
            **{k+'_pct':100*(current['congestion_peaks'][k]/v-1) for k,v in ref['congestion_peaks'].items()})
    selected=json.loads((board/'feedback_selection_map_audit.json').read_text())
    assert selected['map_sha256']==meta['base_map_sha256']
    graph=data['graph']
    evidence=[physical/'placement_review_inputs.json',physical/'physical_inputs.json',old_path,cached_path,
        board/'feedback_selection_map_audit.json',board/'delta_audit.json',physical/'cnn_ram_actual_locations.json',Path(__file__)]
    result=dict(status='MINIMAL_FEEDBACK_PLACEMENT_REVIEW_NOT_ROUTE_RELEASE',current=current,
        delta_vs_original_c2=comparison(old),delta_vs_cached_feedback=comparison(cached),
        delta_vs_passing_pct=data['changes_pct'],source_delta=delta,
        high_fanout_count=graph['high_fanout_including_reset_count'],
        top_high_fanout=graph['high_fanout_including_reset'][:12],
        evidence={str(f):sha(f) for f in evidence},
        limits=['Placement estimates and congestion do not establish routed timing.',
                'No head-cache benefit may be claimed for this version; original queues are unchanged.',
                'Any route requires an explicit review of short tests and actual global costs.'],
        hardware_access=False,board_release=False)
    dst.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:result[k] for k in ('status','delta_vs_original_c2','delta_vs_cached_feedback','delta_vs_passing_pct','high_fanout_count')}))
    print(json.dumps(current['setup_clock_relationships']))
if __name__=='__main__':main()
