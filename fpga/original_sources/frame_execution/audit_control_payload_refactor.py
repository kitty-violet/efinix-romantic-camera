"""Bind this structural refactor's real short integrations to the whole-board map.

Unchanged block cycle counts are local throughput evidence, not frame latency.
"""
import argparse, json, re
from pathlib import Path
from run_units import ROOT, sha
from audit_connected_system import manifest


def gate(root, native=False):
    result = json.loads((root/'result.json').read_text())
    assert result['status'].endswith('_PASS'), root
    path = root/('native_inputs.json' if native else 'inputs.json')
    assert sha(path) == result['inputs_sha256'], path
    data = json.loads(path.read_text())
    if native:
        assert sha(data['binary']) == data['binary_sha256']
        source = Path(data['snapshot'])/'inputs.json'
        assert sha(source) == data['source_manifest_sha256']
        data = json.loads(source.read_text())
    names = {}
    for row in data['inputs']:
        snapshot = Path(row['snapshot'])
        assert sha(snapshot) == row['sha256'], snapshot
        assert snapshot.name not in names or names[snapshot.name] == row['sha256']
        names[snapshot.name] = row['sha256']
    if 'log_sha256' in result:
        assert sha(root/'simulation.log') == result['log_sha256']
    return names, dict(path=str(root), status=result['status'],
                       result_sha256=sha(root/'result.json'), inputs_sha256=sha(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--board', type=Path, required=True)
    p.add_argument('--reference-board', type=Path, required=True)
    p.add_argument('--cluster', type=Path, required=True)
    p.add_argument('--reference-cluster', type=Path, required=True)
    p.add_argument('--guard', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    assert out.is_relative_to(ROOT/'tmp') and not out.exists()
    _, board = manifest(a.board)
    _, reference = manifest(a.reference_board)
    assert board.keys() == reference.keys()
    br = json.loads((a.board/'result.json').read_text())
    for name, key in [('inputs.json','inputs_sha256'),
                      ('outflow/ti60f225_oob.map.v','map_sha256'),
                      ('outflow/ti60f225_oob.map.rpt','report_sha256')]:
        assert sha(a.board/name) == br[key], name
    assert not br['comb_loop_warning'] and not br['new_undriven_signatures']
    changed = sorted(name for name in board if board[name] != reference[name])
    assert changed, 'Not a changed candidate'
    evidence = []
    for root, native, expected in [(a.cluster, True, board),
                                   (a.guard, False, board),
                                   (a.reference_cluster, True, reference)]:
        names, record = gate(root, native)
        common = names.keys() & expected.keys()
        assert common, root
        for name in common:
            assert names[name] == expected[name], (root, name)
        record['common_board_sources'] = len(common)
        if root == a.guard:
            assert set(changed) <= common, 'Changed source absent from four-state integration'
            record['changed_sources_covered'] = changed
        evidence.append(record)
    def blocks(root):
        return re.findall(r'^BLOCK .*$', (root/'simulation.log').read_text(), re.M)
    before, after = blocks(a.reference_cluster), blocks(a.cluster)
    assert len(before) == len(after) == 24 and before == after, 'Block throughput changed'
    result = dict(status='CONTROL_PAYLOAD_SOURCE_CYCLE_BINDING_PASS',
                  board=str(a.board.resolve()), inputs_sha256=br['inputs_sha256'],
                  map_sha256=br['map_sha256'], source_names=len(board),
                  changed_sources=changed, gates=evidence, equal_block_cycle_rows=24,
                  block_cycle_rows=after, frame_ms=None, physical_timing_pass=False,
                  resource_fit=False, board_download_allowed=False)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('gates','block_cycle_rows')}))


if __name__ == '__main__':
    main()
