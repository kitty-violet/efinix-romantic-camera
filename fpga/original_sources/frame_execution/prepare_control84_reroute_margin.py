"""Reuse checked84 placement; tighten setup only for one bounded reroute.

The failed85 capacity estimate and the original84 STA/route stay immutable.
This does not invoke the FPGA, change clocks, or waive original timing gates.
"""
import json
import re
import shutil
from pathlib import Path
from run_units import ROOT, sha
from video_control84_physical import audit, BOARD

B = ROOT / 'tmp/frame-execution'
SOURCE = B / 'video-control84-place1'
DEST = B / 'video-control84-reroute2-margin'
PROOF = B / 'video-control84-margin-proof1/result.json'

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')

def main():
    assert not DEST.exists()
    audit()
    margin = json.loads(PROOF.read_text())
    assert margin['status'] == 'EXTRA_SETUP_MARGIN_SAME_ROUTE_VERIFIED'
    zero, extra = margin['rows']
    assert zero['reserve_ns'] == 0 and extra['reserve_ns'] == .3
    assert zero['setup']['slack_ns'] == -.032
    assert extra['setup']['slack_ns'] == -.332
    assert zero['hold']['slack_ns'] == extra['hold']['slack_ns'] == .026
    for row in margin['rows']:
        for kind in ('setup', 'hold'):
            assert sha(row[kind]['report']) == row[kind]['sha256']
    for name, digest in margin['unchanged_original'].items():
        assert sha(name) == digest
    old = json.loads((SOURCE/'physical_inputs.json').read_text())
    old_review = json.loads((SOURCE/'route-review.json').read_text())
    for name, digest in old_review['required_evidence'].items():
        assert sha(name) == digest, name
    for row in old['inputs']:
        assert sha(row['snapshot']) == row['sha256']
    assert old['base_map_sha256'] == sha(BOARD/'outflow/ti60f225_oob.map.v')
    assert old['optimization_change']['value'] == 'CONGESTION_2'
    assert old['seed_change']['seed'] == 1
    assert json.loads((B/'video-control84-route1/result.json').read_text())['trace_sha256'] == sha(SOURCE/'outflow/ti60f225_oob.troutingtraces')
    original_sdc = SOURCE/'ti60f225_oob.sdc'
    assert sha(original_sdc) == '464ef739d92e3fb1d2b15018863be99636efc1cae42ca581a0526f305c4be262'
    assert not re.search(r'^\s*set_clock_uncertainty\b', original_sdc.read_text(), re.M)
    report = (B/'video-control84-canonical1/outflow/canonical_setup.rpt').read_text()
    slacks = list(map(float, re.findall(r'^Slack\s*:\s*([-\d.]+)', report, re.M)))
    assert len(slacks) == 2000 and sum(x < 0 for x in slacks) == 2
    assert sum(x < .3 for x in slacks) == 142
    # Do not copy stale route artifacts or recursively duplicate old snapshots.
    exclude = shutil.ignore_patterns('before_route', 'route_service.cmd', '*.troutingtraces',
        '*.route.out', '*.route.rpt', '*.route.rpt.xml', '*.timing.rpt', '*.hold_fixing.rpt')
    shutil.copytree(SOURCE, DEST, ignore=exclude)
    geometry = {}
    for suffix in ('.map.v', '.vdb', '.place', '.out.placeloc'):
        rel = 'outflow/ti60f225_oob'+suffix
        assert sha(SOURCE/rel) == sha(DEST/rel)
        geometry[rel] = sha(DEST/rel)
    for rel in ('ti60f225_oob.xml', 'fixed_dsp.placeloc', 'efx_pnr_settings.ini',
                'outflow/ti60f225_oob.interface.csv', 'outflow/ti60f225_oob.pt.sdc'):
        assert sha(SOURCE/rel) == sha(DEST/rel)
        geometry[rel] = sha(DEST/rel)
    sdc = DEST/'ti60f225_oob.sdc'
    addition = '\n# Reserve 0.300ns during optimization; final acceptance is unchanged.\nset_clock_uncertainty -setup -to [get_clocks {*}] 0.300\n'
    sdc.write_text(original_sdc.read_text()+addition, encoding='utf8')
    meta = dict(old)
    meta['inputs'] = []
    for row in old['inputs']:
        path = DEST/Path(row['snapshot']).relative_to(SOURCE)
        if path != sdc:
            assert sha(path) == row['sha256']
        meta['inputs'].append(dict(row, snapshot=str(path), sha256=sha(path)))
    meta['setup_margin'] = dict(extra_setup_ns=.3, extra_hold_ns=0,
        canonical_sdc=str(original_sdc), canonical_sdc_sha256=sha(original_sdc),
        optimization_sdc_sha256=sha(sdc), proof=str(PROOF), proof_sha256=sha(PROOF))
    meta['frozen_placement_reuse'] = dict(source=str(SOURCE),
        source_physical_inputs_sha256=sha(SOURCE/'physical_inputs.json'),
        geometry=geometry, placement_rerun=False, recipe_changed=False, seed_changed=False,
        rationale='84 has a legal route with two32/18ps misses;85 algebraic rewrite exceeded capacity. Preserve the checked global placement and give the existing routing recipe explicit setup pressure. No logic/capacity/frequency/acceptance relaxation.')
    save(DEST/'physical_inputs.json', meta)
    prior = json.loads((SOURCE/'before_route/result.json').read_text())
    assert prior['status'] == 'PLACEMENT_ONLY_FINISHED_NOT_RELEASE'
    assert all(s['exit_code'] == 0 for s in prior['stages'])
    # These stages are inherited, not falsely reported as a new placement run.
    prior['placement_inherited_from'] = str(SOURCE)
    save(DEST/'result.json', prior)
    save(DEST/'stages.json', prior['stages'])
    docs = Path('E:/efinity-2026.1.132-windows-x64/efinity/2026.1/doc/topics')
    receipt = dict(status='UNCHANGED84_FULL_GEOMETRY_REVIEW_REUSED_SETUP_ONLY_PRESSURE',
        source=str(SOURCE), destination=str(DEST), geometry=geometry,
        previous_negative_paths=2, previous_wns_ns=-.032, previous_whs_ns=.026,
        original2000_paths_below_added_margin=142, source_and_placement_unchanged=True,
        all_original_clock_io_exceptions_preserved=True, extra_setup_ns=.3, extra_hold_ns=0,
        resources=dict(xlr=60047, xlr_capacity=60800, ram=245, ram_capacity=256, dsp=160, dsp_capacity=160),
        original_global_review=str(SOURCE/'global-bandwidth-review.json'),
        original_path_review=str(SOURCE/'control84-path-review.json'),
        evidence={str(p):sha(p) for p in [Path(__file__),PROOF,SOURCE/'physical_inputs.json',
            SOURCE/'placement_review_inputs.json',SOURCE/'global-bandwidth-review.json',
            SOURCE/'control84-path-review.json',SOURCE/'route-review.json',
            docs/'constraint-set-clock-uncertainty.html',docs/'concept_constraints_uncertainty.html']},
        hardware_access=False, board_release=False)
    save(DEST/'geometry-reuse-proof.json', receipt)
    save(DEST/'placement_review_inputs.json', dict(status=receipt['status'],
        proof=str(DEST/'geometry-reuse-proof.json'),proof_sha256=sha(DEST/'geometry-reuse-proof.json')))
    files=[DEST/'physical_inputs.json',DEST/'geometry-reuse-proof.json',DEST/'placement_review_inputs.json',PROOF,Path(__file__)]
    required=dict(old_review['required_evidence'])
    required.update({str(p):sha(p) for p in files})
    review=dict(old_review, physical=str(DEST), placement_review_sha256=sha(DEST/'placement_review_inputs.json'),
        required_evidence=required, budget_seconds=2100,
        rationale=old_review['rationale']+[
            'Exact same84 map/vdb/global+detail placement,404hard anchors andpins. It already routed legally in51iterations. Setup pressure only: +0.300ns gives142/2000 near-critical endpoints additional priority; no seed/recipe sweep or local geometry shuffle.',
            'Current same-route proof confirms WNS-.032->-.332 while WHS+.026 stays identical. Original acceptance remains WNS>=0/WHS>=0 with all vendor uncertainty; the optimizer margin is removed only in a private finalSTA copy.'],
        remaining_risks=old_review['remaining_risks']+[
            'Tighter routing can increase congestion or change formerly positive paths; all2000 setup/hold endpoints and current resource/CDC/PLL gates must be rerun. No guarantee of convergence; one2100s attempt with independent2145s host timeout.'],
        hardware_access=False,board_release=False)
    save(DEST/'route-review.json',review)
    print('CONTROL84_FROZEN_PLACEMENT_MARGIN_REROUTE_READY',DEST,flush=True)

if __name__=='__main__':
    main()
