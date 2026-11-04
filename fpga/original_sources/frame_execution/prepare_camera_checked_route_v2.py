"""Review this repaired map's placement and short evidence before routing."""
import json
from run_units import ROOT, HERE, sha
from camera_checked_v2 import BOARD
from camera_checked_board import audit_delta
from finish_camera_checked_release_v2 import camera_evidence, SHORTS


def main():
    base = ROOT / 'tmp/frame-execution'
    place = base / 'camera-checked-place2'
    out = base / 'camera-checked-route-review2.json'
    assert not out.exists()
    audit_delta(BOARD)
    evidence = {}
    def bind(path):
        evidence[str(path)] = sha(path)
        return json.loads(path.read_text())
    def verify(rows):
        for filename, digest in rows.items():
            assert sha(filename) == digest, filename
            evidence[filename] = digest
    camera_evidence(bind, verify)
    short = bind(SHORTS / 'result.json')
    assert short['status'] == 'LOCAL_DRIVER_SAME_MAP_SHORT_INTEGRATION_PASS'
    assert short['map_sha256'] == sha(BOARD/'outflow/ti60f225_oob.map.v')
    verify(short['evidence'])
    reused = bind(SHORTS/'map-audit/reuse.json');verify(reused['evidence'])
    external = bind(BOARD/'camera_external_input_audit.json')
    assert external['status'] == 'ALL_CAMERA_MAPPED_INPUTS_DRIVEN_RAW6_OR_RAW5_ZERO_PASS'
    verify(external['evidence'])
    review = bind(place / 'placement_review_inputs.json')
    current, previous = review['current'], review['reference']
    assert current['placement_legal'] and current['canonical_setup_estimate_ns'] >= 0
    assert all(x['used'] <= x['capacity'] for x in current['resources'].values())
    assert len(current['setup_clock_relationships']) == 23
    assert min(x['slack_ns'] for x in current['setup_clock_relationships']) >= 0
    for key in ('current', 'reference'):verify(review[key]['evidence'])
    verify(review['graph']['evidence'])
    for filename in ('result.json','physical_inputs.json','cnn_ram_actual_locations.json'):
        bind(place/filename)
    bind(BOARD/'result.json')
    for script in ('prepare_camera_checked_route_v2.py','review_camera_checked_place.py',
                   'resume_camera_checked_shorts.py','finish_camera_checked_release_v2.py'):
        evidence[str(HERE/script)] = sha(HERE/script)
    changes = review['changes_pct']
    result = dict(decision='PROCEED_TO_BOUNDED_ROUTE', board=str(BOARD),
        placement=str(place), shorts=str(SHORTS), physical=str(base/'camera-checked-bound2'),
        budget_seconds=1350, route_profile_cli_override=dict(before='CONGESTION_2',after='TIMING_1'),
        rationale=[
            'Current full video/A placement uses 56637/60800 XLR, only 5 above the passing parent; RAM228/256 and DSP160/160 unchanged. New native map/input-driver proof is bound; checked2 alias-defective implementation is excluded.',
            f"Versus passing parent placement: fine congestion {changes['4x4']:+.3f}%, medium {changes['1/16']:+.3f}%, coarse {changes['1/4']:+.3f}%, weighted HPWL {changes['weighted_hpwl']:+.3f}%. Fine peak is worse; no claim of uniform congestion improvement. Medium reduction, near-flat global wirelength and lower packed routing-only overhead support one bounded trial, not a guarantee.",
            'All23 setup relationships remain positive (worst +0.315ns estimate). Ordinary logic remains movable; 113 CNN RAM and 288 DSP primitive anchors checked. High-fanout observation/control/reset and clock relationships reviewed across the complete placed design, not only the camera cone.',
            'Core/adapter/startup/three-reset/serial same-map short regressions, camera10 RTL cases, actual mapped0/4/6/8, 117 host decode tests and 16 capture-enable cases pass. Added RAW6 reads both genuine fault FFs; reserved RAW5 is explicit zero.',
            'Use previously successful C2 placement/TIMING_1 routing recipe once with1350-second tool budget and independent host stop. Exact Gray endpoint name rebinding preserves all timing values. Retain source and placement frozen.'
        ], remaining_risks=[
            'Increased fine congestion and negative pre-route hold estimates require real routing; placement estimates cannot establish WNS/WHS or zero overlap.',
            'New camera95-bit held-data crossing must settle within13.332ns under dedicated routed timing; no missing or false-pathed endpoints count as pass.',
            'Require original constraints WNS>=0/WHS>=0, all resource capacities, DDR70 Gray bits, JTAG CDC, PLL and final bitstream/same-source binding before offline release.',
            'Actual camera chip ID/reset polarity/physical transport, picture and throughput remain supervised board tests. No board access in this offline task.'
        ], reviewed_evidence=evidence, hardware_access=False, board_release=False)
    out.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(out)


if __name__ == '__main__':
    main()
