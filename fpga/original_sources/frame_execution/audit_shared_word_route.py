"""Bind shared word routing and fixed load wiring to their actual board inputs."""
import argparse, json, re
from pathlib import Path
from run_units import ROOT, sha
from audit_resource_scope import read_board
from audit_control_payload_refactor import gate
from audit_no_audio import verify_no_audio_binding

CHANGED = {'unified_column_dispatch32.sv', 'frame_arena_row_dma.sv'}


def source_gate(root, native=False, dma=False):
    names, record = gate(root, native)
    if dma:
        # This public-RTL unit runner reads/writes text on Windows, preserving
        # tokens but normalizing LF to CRLF. Prove that exact transformation;
        # do not waive arbitrary snapshot differences or use another RAM file.
        data = json.loads((root/'inputs.json').read_text())
        for row in data['inputs']:
            source, snapshot = Path(row['source']), Path(row['snapshot'])
            if source.name not in ('tensor_arena12.sv', 'frame_arena_row_dma.sv') or source.name != snapshot.name:
                continue
            assert sha(source) == row['source_sha256']
            assert source.read_bytes().replace(b'\r\n', b'\n') == snapshot.read_bytes().replace(b'\r\n', b'\n')
            names[snapshot.name] = row['source_sha256']
    return names, record


def verify_shared_word_binding(binding):
    assert binding['status'] == 'SHARED_WORD_ROUTE_SOURCE_CYCLE_BINDING_PASS'
    board, reference = Path(binding['board']), Path(binding['reference_board'])
    before, after = read_board(reference), read_board(board)
    assert before['names'].keys() == after['names'].keys()
    changed = {n for n in before['names'] if before['names'][n] != after['names'][n]}
    assert changed == CHANGED and len(after['names']) == 125
    assert not after['result']['new_undriven_signatures']
    for key in ('inputs_sha256', 'map_sha256'):
        assert binding[key] == after['result'][key]
    prior = binding['inherited_binding']
    assert sha(prior['path']) == prior['sha256']
    inherited = json.loads(Path(prior['path']).read_text())
    verify_no_audio_binding(inherited)
    assert Path(inherited['board']) == reference
    required = {'cluster', 'reference_cluster', 'guard', 'dispatch', 'dma'}
    assert {g['role'] for g in binding['gates']} == required
    roots = {}
    for record in binding['gates']:
        role, root = record['role'], Path(record['path'])
        roots[role] = root
        assert sha(root/'result.json') == record['result_sha256']
        names, checked = source_gate(root, role in ('cluster', 'reference_cluster'), role == 'dma')
        assert checked['inputs_sha256'] == record['inputs_sha256']
        expected = before['names'] if role == 'reference_cluster' else after['names']
        common = names.keys() & expected.keys()
        assert common and all(names[n] == expected[n] for n in common), (role, sorted(common))
        if role == 'guard': assert CHANGED <= common
        if role == 'dispatch': assert 'unified_column_dispatch32.sv' in common
        if role == 'dma':
            assert {'frame_arena_row_dma.sv', 'tensor_arena12.sv'} <= common
            result = json.loads((root/'result.json').read_text())
            assert result['status'] == 'FIXED_LOAD_ROUTE_FOUR_STATE_CYCLE_EQUIVALENCE_PASS'
            assert result['reference_sha256'] == before['names']['frame_arena_row_dma.sv']
            assert result['candidate_sha256'] == after['names']['frame_arena_row_dma.sv']
            assert {r['seed'] for r in result['runs']} == {71, 391, 1027}
            for run in result['runs']:
                assert sha(root/f"simulation_{run['seed']}.log") == run['log_sha256']
                assert run['edges'] > 17000
    def blocks(root):
        return re.findall(r'^BLOCK .*$', (root/'simulation.log').read_text(), re.M)
    old, new = blocks(roots['reference_cluster']), blocks(roots['cluster'])
    assert len(old) == len(new) == 24 and old == new
    vectors = []
    for role in ('cluster', 'reference_cluster'):
        source = json.loads((roots[role]/'frozen/inputs.json').read_text())
        vectors.append(source['vectors'])
        for name, digest in source['vectors'].items():
            assert sha(roots[role]/'frozen/vectors'/name) == digest
    assert vectors[0] == vectors[1]
    return before, after


def main():
    p = argparse.ArgumentParser()
    for name in ('board', 'reference-board', 'cluster', 'reference-cluster', 'guard', 'dispatch', 'dma'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    board, reference = a.board.resolve(), a.reference_board.resolve()
    output = board/'gate_binding.json'
    assert board.is_relative_to(ROOT/'tmp') and not output.exists()
    br = json.loads((board/'result.json').read_text())
    records = []
    for role in ('cluster', 'reference_cluster', 'guard', 'dispatch', 'dma'):
        root = getattr(a, role).resolve()
        _, record = source_gate(root, role in ('cluster', 'reference_cluster'), role == 'dma')
        records.append(dict(role=role, **record))
    binding = dict(status='SHARED_WORD_ROUTE_SOURCE_CYCLE_BINDING_PASS',
        board=str(board), reference_board=str(reference),
        inputs_sha256=br['inputs_sha256'], map_sha256=br['map_sha256'],
        source_names=125, changed_sources=sorted(CHANGED), unchanged_sources=123,
        inherited_binding=dict(path=str(reference/'gate_binding.json'), sha256=sha(reference/'gate_binding.json')),
        gates=records, equal_block_cycle_rows=24, full_frame=False,
        physical_timing_pass=False, resource_fit=False, board_download_allowed=False)
    before, after = verify_shared_word_binding(binding)
    binding['resource_delta'] = {k:v-before['result']['resources'].get(k,0)
                                  for k,v in after['result']['resources'].items()}
    output.write_text(json.dumps(binding, indent=2)+'\n')
    print(json.dumps({k:v for k,v in binding.items() if k!='gates'}), flush=True)


if __name__ == '__main__': main()
