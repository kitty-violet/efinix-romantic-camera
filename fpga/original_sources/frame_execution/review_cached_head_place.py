"""Read actual placement and compare C2 before routing; no automatic release."""
import argparse, json, re
from pathlib import Path
from run_units import ROOT, sha
from fixed_control_ddr_reuse_locations import locations

def review(physical):
    data = json.loads((physical/'placement_review_inputs.json').read_text())
    meta = json.loads((physical/'physical_inputs.json').read_text())
    board = Path(meta['base'])
    heads = json.loads((board/'cached_head_map_audit.json').read_text())
    assert heads['map_sha256'] == meta['base_map_sha256']
    pos = locations(physical/'outflow/ti60f225_oob.out.placeloc')
    fixed = locations(physical/'fixed_dsp.placeloc')
    groups = {i: [] for i in range(3)}
    for signal, cell in heads['head_drivers'].items():
        bus, bit = re.search(r'/(od|ox|ob)\[(\d+)\]$', signal).groups()
        group = int(bit)//{'od':256, 'ox':9, 'ob':1}[bus]
        assert cell in pos and cell not in fixed, cell
        groups[group].append(pos[cell])
    boxes = {}
    for group, points in groups.items():
        assert len(points) == 266
        boxes[group] = dict(bits=len(points), bbox=[min(p[0] for p in points), min(p[1] for p in points),
                                                   max(p[0] for p in points), max(p[1] for p in points)])
    prior = ROOT/'tmp/frame-execution/local-decode-place1/placement_review_inputs.json'
    parent = json.loads(prior.read_text())['current']
    current = data['current']
    delta = dict(weighted_hpwl=100*(current['weighted_hpwl_per_net']/parent['weighted_hpwl_per_net']-1),
                 **{k:100*(current['congestion_peaks'][k]/v-1) for k,v in parent['congestion_peaks'].items()})
    paths = [physical/'placement_review_inputs.json', physical/'physical_inputs.json', prior,
             board/'cached_head_map_audit.json', physical/'fixed_dsp.placeloc',
             physical/'outflow/ti60f225_oob.out.placeloc', Path(__file__)]
    graph = data['graph']
    return dict(status='CACHED_HEAD_PLACEMENT_REVIEW_DATA_NOT_ROUTE_RELEASE', current=current,
                original_c2=parent, delta_vs_original_c2_pct=delta, delta_vs_passing_pct=data['changes_pct'],
                head_positions=boxes, ordinary_head_cells_fixed=False,
                high_fanout_count=graph['high_fanout_including_reset_count'],
                top_high_fanout=graph['high_fanout_including_reset'][:12],
                rq_high_fanout=[r for r in graph['omitted_by_old_cost'] if '/rq/' in r['signal']],
                evidence={str(p):sha(p) for p in paths},
                interpretation='Geometric bboxes are placement context, not path delay or routing capacity. Final route and STA required.',
                hardware_access=False, board_release=False)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--physical', type=Path, required=True)
    args=parser.parse_args(); physical=args.physical.resolve()
    output=physical/'cached_head_placement_review.json'; assert not output.exists()
    result=review(physical); output.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('current','original_c2','evidence','rq_high_fanout')}))
