"""Resolve the one explicitly versioned placement result after a successful route.

The route runner snapshots result.json before writing its final stage record.
All other paths/hashes remain exact and unchanged. No hardware access.
"""
import json
from pathlib import Path
from run_units import sha


def placement_result_version(route):
    route = Path(route).resolve()
    rr = json.loads((route/'result.json').read_text())
    ri = json.loads((route/'inputs.json').read_text())
    assert rr['status'] == 'VIDEO_SERVICE_LEGAL_ROUTE_PASS_NOT_RELEASE'
    assert rr['inputs_sha256'] == sha(route/'inputs.json')
    physical = Path(ri['physical']).resolve()
    current = physical/'result.json'
    snapshot = physical/'before_route/result.json'
    old_hash = ri['required_evidence'][str(current)]
    assert sha(snapshot) == old_hash
    old = json.loads(snapshot.read_text())
    new = json.loads(current.read_text())
    route_stages = json.loads((route/'stages.json').read_text())
    assert old['status'] == 'PLACEMENT_ONLY_FINISHED_NOT_RELEASE'
    assert old['stages'] and all(s['exit_code'] == 0 for s in old['stages'])
    assert len(route_stages) == 1
    assert route_stages[0]['status'] == 'finished' and route_stages[0]['exit_code'] == 0
    assert route_stages[0]['log_sha256'] == sha(route/'route.log')
    assert new['status'] == 'PHYSICAL_CAPACITY_PROBE_FINISHED'
    assert new['stages'] == old['stages'] + route_stages
    assert new['strict_sta_pass'] is False and new['board_download_allowed'] is False
    assert new['source_unchanged'] is True
    assert sha(physical/'outflow/ti60f225_oob.troutingtraces') == rr['trace_sha256']
    return dict(logical_path=str(current), placement_snapshot=str(snapshot),
                placement_sha256=old_hash, routed_result_sha256=sha(current),
                route_result_sha256=sha(route/'result.json'),
                route_inputs_sha256=sha(route/'inputs.json'))


def resolve_version(path, expected, version):
    path = Path(path).resolve()
    if str(path) == version['logical_path'] and expected == version['placement_sha256']:
        snapshot = Path(version['placement_snapshot'])
        assert sha(snapshot) == expected
        assert sha(path) == version['routed_result_sha256']
        return snapshot
    return path
