"""Count the actual A descriptor workload, never infer per-layer wall time.

The report is offline: commands, mathematical MACs and logical payload bytes.
No ideal DDR bandwidth or MAC throughput is substituted for measurements.
"""
import argparse
import collections
import json
import zlib
import math
import sys
from pathlib import Path
from run_units import ROOT, sha
sys.path.insert(0, str(ROOT/'network_ab'))
from model_contract import compile_contract, load_contract


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists() and a.output.resolve().is_relative_to(ROOT/'tmp')
    folder = ROOT/'tmp/frame-execution/accepted-a-integration1/descriptors'
    result = json.loads((folder/'result.json').read_text())
    assert result['status'] == 'ACCEPTED_A_RESIDUAL_PROGRAM_BOUND'
    program_path = Path(result['program_path'])/'program.json'
    assert sha(program_path) == result['program_sha256']
    assert sha(folder/'schema.json') == result['schema_sha256']
    schema = json.loads((folder/'schema.json').read_text())
    program = json.loads(program_path.read_text())
    records = {}
    for kind in ('compute', 'dma'):
        f = folder/(kind+'.bin')
        raw = f.read_bytes()
        assert sha(f) == result['files'][f.name]
        assert f'{zlib.crc32(raw):08x}' == result['crc32'][kind]
        size = schema[kind]['record_bytes']
        assert len(raw) == size*schema[kind]['records']
        records[kind] = []
        for offset in range(0, len(raw), size):
            word = int.from_bytes(raw[offset:offset+size], 'little')
            fields = {f['field']: (word >> f['offset']) & ((1 << f['width'])-1)
                      for f in schema[kind]['fields']}
            records[kind].append(fields)
    nodes = collections.defaultdict(lambda: dict(commands=0, output_scalars=0,
                                                mathematical_macs=0, residual_scalars=0))
    seen = set()
    for fields in records['compute']:
        index = fields['program_index']
        assert index not in seen
        seen.add(index)
        row = program[index]
        assert row['kind'] == 'compute'
        for field, key in [('f2', 'pixels'), ('f3', 'rows'), ('f4', 'input_groups'),
                           ('f5', 'output_groups'), ('f28', 'mask')]:
            assert fields[field] == row[key], (index, key)
        op = row['operator']
        assert op in ('conv3x3', 'dw3x3', 'pw1x1', 'pw_fused')
        # These full-frame descriptors cover every output group. The engine
        # visits DW groups internally; one descriptor is not one OC8 group.
        channels = row['output_groups']*row['mask'].bit_count()
        scalars = row['pixels']*row['rows']*channels
        terms = 27 if op == 'conv3x3' else 9 if op == 'dw3x3' else 8*row['input_groups']
        n = nodes[row['node']]
        n.update(operator=op, group=row['group'], parameter_node=row['parameter_node'])
        n['commands'] += 1
        n['output_scalars'] += scalars
        n['mathematical_macs'] += scalars*terms
        if op == 'pw_fused':
            assert not (fields['f6'] & 32) and fields['f24'] == 1 and fields['f25'] == 131071
            n['residual_scalars'] += scalars
    assert seen == {i for i, r in enumerate(program) if r['kind'] == 'compute'}
    contract_path = ROOT/'network_ab/model_b_contract.json'
    assert sha(contract_path) == result['network_contract_sha256']
    plan = compile_contract(load_contract())
    conv = {n['id']: n for n in plan['nodes'] if n.get('macs', 0)}
    assert {n['parameter_node'] for n in nodes.values()} == set(conv)
    for n in nodes.values():
        c = conv[n['parameter_node']]
        n['canonical_output_scalars'] = math.prod(c['output_shape_hwc'])
        n['canonical_macs'] = c['macs']
        assert n['output_scalars'] >= n['canonical_output_scalars']
        assert n['mathematical_macs'] >= n['canonical_macs']
        n['tile_recompute_macs'] = n['mathematical_macs']-c['macs']
    dma = collections.defaultdict(lambda: dict(commands=0, logical_payload_bytes=0))
    compiler = Path(__file__).with_name('compile_row_pair_descriptors.py')
    assert sha(compiler) == result['script_sha256']
    # The bound compiler appends each stream while visiting original commands
    # in increasing index order; check the encoded geometry below as well.
    indices = [i for i, r in enumerate(program) if r['kind'] != 'compute']
    assert len(indices) == len(set(indices)) == len(records['dma'])
    assert set(indices) == {i for i, r in enumerate(program) if r['kind'] != 'compute'}
    for index, fields in zip(indices, records['dma']):
        row = program[index]
        assert (fields['f0'] == 1) == (row['kind'] == 'store')
        assert fields['f2'] == row['row']['width']
        assert fields['f4'] == row['row']['bank'] and fields['f5'] == row['row']['base']
        assert fields['f9'] == row['row']['y'] and fields['f10'] == row['row']['origin']
        assert (3 if fields['f6'] else 8*fields['f3']) == row['channels']
        group = dma[row['group']+'_'+row['kind']]
        group['commands'] += 1
        group['logical_payload_bytes'] += row['row']['width']*row['channels']
    total_macs = sum(n['mathematical_macs'] for n in nodes.values())
    report = dict(status='ACCEPTED_A_DESCRIPTOR_WORKLOAD_AUDIT_NOT_TIMING',
                  compute_records=len(seen), dma_records=len(indices),
                  descriptor_bytes=sum((folder/(k+'.bin')).stat().st_size for k in records),
                  mathematical_macs_including_tile_recomputation=total_macs,
                  canonical_network_macs=plan['macs_per_frame'],
                  nodes=[dict(node=k, **v, mac_share_percent=100*v['mathematical_macs']/total_macs)
                         for k, v in nodes.items()], dma=dict(dma),
                  measured_per_layer_cycles=None, proves_15fps=False,
                  scope='Actual immutable A descriptor dimensions. Fused PW/residual counted once for PW MACs; residual scaling/add separately. Logical DMA payload excludes padding/bus alignment and protocol overhead. MAC share is not wall-time share.',
                  not_counted=['configuration and coefficient fetch', 'resident/RAM bank stalls',
                               'DDR arbitration/refresh/turnaround', 'NN address work',
                               'RQ and residual latency', 'capture/display and repeats'],
                  evidence={str(f): sha(f) for f in (Path(__file__), folder/'result.json',
                            folder/'schema.json', folder/'compute.bin', folder/'dma.bin',
                            program_path, compiler, contract_path,
                            ROOT/'network_ab/model_contract.py')})
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k:report[k] for k in ('status','compute_records','dma_records',
                                         'mathematical_macs_including_tile_recomputation','dma')}))


if __name__ == '__main__':
    main()
