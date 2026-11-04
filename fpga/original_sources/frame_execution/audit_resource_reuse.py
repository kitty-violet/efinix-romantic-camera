"""Read-only structural/resource audit of the frozen real-datapath candidate.

No RTL/netlist rewriting; duplicate fingerprints are not formal equivalence or
placement-aware removal proofs. Different inputs never count as duplicates.
"""
import collections
import hashlib
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CASE = ROOT / 'tmp/frame-execution/byte-rotated-map1'
VECTORS = ROOT / 'tmp/frame-execution/joint-reorder-vectors1'
OUT = ROOT / 'output/toolchain/resource_reuse_audit_2026-10-01.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(net):
    return re.sub(r'\s+', '', net).lstrip('\\')


def lut_key(inputs, mask):
    """Canonical truth table over distinct, actually influential input nets."""
    constants = {"1'b0": 0, "1'b1": 1}
    nets = sorted(set(inputs) - constants.keys())
    table = []
    for address in range(1 << len(nets)):
        values = dict(constants)
        values.update((n, (address >> i) & 1) for i, n in enumerate(nets))
        index = sum(values[n] << i for i, n in enumerate(inputs))
        table.append((mask >> index) & 1)
    live = [i for i in range(len(nets)) if any(
        table[x] != table[x ^ (1 << i)] for x in range(len(table)))]
    reduced = sum(table[sum(((a >> j) & 1) << i for j, i in enumerate(live))] << a
                  for a in range(1 << len(live)))
    return tuple(nets[i] for i in live), reduced


def verify_canonicalizer():
    for mask in (0, 0xffff, 0xaaaa, 0x6996, 0x8000, 0xcafe):
        original = ['a', 'b', 'c', 'd']
        for order in itertools.permutations(range(4)):
            permuted = [original[i] for i in order]
            new_mask = 0
            for index in range(16):
                old_index = sum(((index >> j) & 1) << i for j, i in enumerate(order))
                new_mask |= ((mask >> old_index) & 1) << index
            assert lut_key(original, mask) == lut_key(permuted, new_mask)
    assert lut_key(['a', 'x', 'y', 'z'], 0xaaaa) == (('a',), 2)
    assert lut_key(['a', 'a', "1'b0", "1'b1"], 0x6666) == ((), 0)


def hierarchy():
    rows, stack = [], []
    keys = ['FF', 'SRL8', 'ADD', 'LUT4', 'COMB4', 'RAM', 'DSP_INSTANCES']
    for line in (CASE/'outflow/cnn_frame_datapath.res.csv').read_text().splitlines():
        fields = line.split('\t')
        if len(fields) < 8 or not re.fullmatch(r'\s*\d+\(\d+\)', fields[1]):
            continue
        depth = len(fields[0]) - len(fields[0].lstrip())
        name, module = fields[0].strip().lstrip('+').split(':', 1)
        while stack and stack[-1][0] >= depth:
            stack.pop()
        path = '/'.join([v[1] for v in stack] + [name])
        stack.append((depth, name))
        values = [tuple(map(int, re.findall(r'\d+', s))) for s in fields[1:8]]
        rows.append(dict(path=path, module=module.strip(), depth=depth,
                         inclusive=dict(zip(keys, [v[0] for v in values])),
                         own=dict(zip(keys, [v[1] for v in values]))))
    for key in keys:
        assert sum(r['own'][key] for r in rows) == rows[0]['inclusive'][key], key
    return rows


def logical_fracture_candidates(luts):
    """Disjoint 3-input/<=2-input pairs sharing pins; no placement claim.

    This deliberately omits 2+2 packing and LUT/ADD opportunities. It is neither
    a maximal packing nor a predicted saving from the vendor's mapper.
    """
    small = collections.defaultdict(list)
    for name, nets, output in luts:
        if 0 < len(nets) <= 2:
            small[nets].append((name, output))
    pairs = []
    for name, nets, output in luts:
        if len(nets) != 3:
            continue
        matched = False
        for size in (2, 1):
            for subset in itertools.combinations(nets, size):
                for i, (other, other_output) in enumerate(small[subset]):
                    if other_output not in nets and output not in subset:
                        pairs.append((name, other))
                        small[subset].pop(i)
                        matched = True
                        break
                if matched:
                    break
            if matched:
                break
    assert len(set(itertools.chain.from_iterable(pairs))) == 2*len(pairs)
    return dict(input_support_histogram=dict(collections.Counter(len(x[1]) for x in luts)),
                disjoint_three_plus_small_pairs=len(pairs), examples=pairs[:8],
                actual_packing_measured=False)


def main():
    verify_canonicalizer()
    result = json.loads((CASE/'result.json').read_text())
    netlist = CASE/'outflow/cnn_frame_datapath.map.v'
    assert sha(netlist) == result['map_sha256']
    assert sha(CASE/'inputs.json') == result['inputs_sha256']
    inputs = json.loads((CASE/'inputs.json').read_text())['inputs']
    for source in inputs:
        assert sha(Path(source['snapshot'])) == source['sha256'], source['snapshot']
    counts = collections.Counter()
    luts = []
    groups = {k: collections.defaultdict(list) for k in ('LUT4', 'FF')}
    cell_pattern = re.compile(r'^\s*EFX_([A-Z0-9_]+)\s+(\S+)\s+\((.*?)\)\s*/\*\s*verific(.*?)\*/', re.M | re.S)
    for match in cell_pattern.finditer(netlist.read_text()):
        kind, name, ports_text, attributes_text = match.groups()
        counts[kind] += 1
        if kind not in groups:
            continue
        pins = {k: normalized(v) for k, v in re.findall(r'\.(\w+)\((.*?)\)', ports_text, re.S)}
        attrs = dict(re.findall(r'(\w+)=([^,\s]+)', attributes_text))
        if kind == 'LUT4':
            assert 'O' in pins and set(pins) <= {'I0', 'I1', 'I2', 'I3', 'O'}
            key = lut_key([pins.get('I'+str(i), '__unconnected_'+str(i)) for i in range(4)], int(attrs['LUTMASK'].split("'h")[1], 16))
            assert not any(n.startswith('__unconnected_') for n in key[0]), name
            luts.append((name, key[0], pins['O']))
        else:
            assert 'Q' in pins
            pins.pop('Q')
            key = tuple(sorted(pins.items())), tuple(sorted(attrs.items()))
        groups[kind][key].append(name)
    assert dict(counts) == result['resources'], (counts, result['resources'])
    duplication = {}
    for kind, fingerprints in groups.items():
        repeated = sorted((v for v in fingerprints.values() if len(v) > 1), key=len, reverse=True)
        excess = sum(len(v)-1 for v in repeated)
        duplication[kind] = dict(cells=counts[kind], repeated_groups=len(repeated),
            excess_cells=excess, excess_percent=100*excess/counts[kind], examples=repeated[:8])
    rows = hierarchy()
    top = rows[0]['inclusive']
    for key in ('LUT4', 'FF', 'ADD', 'SRL8'):
        assert top[key] == counts[key]
    assert top['RAM'] == counts['RAM10'] + counts['DPRAM10']
    partition = []
    for row in rows:
        if row['depth'] == 1 and row['path'].split('/')[-1] != 'dut':
            partition.append(dict(name=row['path'].split('/')[-1], **row['inclusive']))
        elif row['depth'] == 2 and row['path'].split('/')[1] == 'dut':
            partition.append(dict(name='/'.join(row['path'].split('/')[1:]), **row['inclusive']))
    for suffix in ('cnn_frame_datapath', 'cnn_frame_datapath/dut'):
        row = next(r for r in rows if r['path'] == suffix)
        partition.append(dict(name=suffix+'/own', **row['own']))
    assert sum(r['LUT4'] for r in partition) == top['LUT4']
    modules = {}
    for module in ('int8_dual_mac_lane_prefix_stream', 'shared_mac_stream_front',
                   'shared_rq_oc8_cluster', 'requantize_i32_mul18_q17_pipeline',
                   'shallow_result_fifo', 'banked_result_fifo'):
        matched = [r for r in rows if r['module'].split('(')[0] == module]
        modules[module] = dict(instances=len(matched),
            inclusive_sum={k:sum(r['inclusive'][k] for r in matched) for k in top},
            own_sum={k:sum(r['own'][k] for r in matched) for k in top},
            paths=[r['path'] for r in matched] if len(matched)<=6 else [])
    manifest = json.loads((VECTORS/'frame_manifest.json').read_text())
    assert sha(VECTORS/'program.hex') == manifest['files']['program.hex']
    dma = collections.defaultdict(lambda: dict(commands=0, address_mod16=collections.Counter(),
        groups=collections.Counter(), columns_mod4=collections.Counter(), payload_bytes=0))
    compute = 0
    for line in (VECTORS/'program.hex').read_text().splitlines():
        record = int(line, 16)
        f = [(record >> (32*i)) & 0xffffffff for i in range(32)]
        if f[0] == 2:
            compute += 1
            continue
        assert f[0] in (0, 1) and f[6] in (0, 1)
        key = ('store' if f[0] else 'load') + ('_rgb' if f[6] else '_feature')
        stat = dma[key]
        stat['commands'] += 1
        stat['address_mod16'][f[7]%16] += 1
        stat['groups'][f[3]] += 1
        stat['columns_mod4'][f[2]%4] += 1
        stat['payload_bytes'] += f[2]*(3 if f[6] else f[3]*8)
    assert compute == 8789 and sum(v['commands'] for v in dma.values()) == 10440
    layout_file = ROOT/'output/toolchain/frame_descriptor_layout_2026-10-01.json'
    layout = json.loads(layout_file.read_text(encoding='utf-8'))
    assert layout['program_sha256'] == sha(VECTORS/'program.hex')
    refs = sum(len(q['pattern'])*q['count'] for q in layout['rowmap_patterns'])
    unique_refs = sum(len(set(q['pattern']))*q['count'] for q in layout['rowmap_patterns'])
    queue_rows = [dict(path=r['path'], **r['inclusive']) for r in rows if r['module'] == 'shallow_result_fifo']
    ram = dict(arena=dict(blocks=84, useful_bits=12*1024*64, physical_bits=84*10240),
        resident=dict(blocks=20, image_bytes=17792, capacity_bits=5*512*64, physical_bits=20*10240),
        mac_results=dict(blocks=16, useful_bits=4*16*80, physical_bits=16*10240))
    assert sum(v['blocks'] for v in ram.values()) == top['RAM']
    physical_dsp = counts['DSP48'] + counts['DSP24']//2
    assert physical_dsp == 160
    assert modules['shared_rq_oc8_cluster']['instances'] == 4
    assert modules['banked_result_fifo']['instances'] == 4
    contract = ROOT/'network_ab/model_b_contract.json'
    assert sha(contract) == '5750c4668d75077746ec5d4accf8773ab437540cf933c202f153a8797c21dd01'
    macs = 966758400
    audit = dict(status='READ_ONLY_FROZEN_NETLIST_RESOURCE_ANALYSIS_PASS_NOT_RESOURCE_FIT',
        case=str(CASE), map_sha256=sha(netlist), program_sha256=sha(VECTORS/'program.hex'),
        resources=dict(counts), physical_dsp=physical_dsp,
        uncombined_lut_add_srl_primitives=counts['LUT4']+counts['ADD']+counts['SRL8'],
        direct_duplicate_fingerprints=duplication, exclusive_lut_partition=partition,
        logical_fracture_candidates=logical_fracture_candidates(luts),
        module_repetition=modules, shallow_queues=queue_rows, ram_capacity=ram,
        dma_program_statistics=dma, compute_commands=compute, contract_sha256=sha(contract),
        row_references=dict(patterns=len(layout['rowmap_patterns']),total=refs,
            unique_per_command_total=unique_refs,repeated_reference_percent=100*(refs-unique_refs)/refs,
            note='Repeated logical rows from boundary/NN views, not automatically extra physical storage.'),
        ideal_mac_only_bounds_ms={str(fronts):macs/(fronts*8*150000) for fronts in (32,28,24)},
        limitations=[
            'Direct LUT fingerprints normalize input permutation/constants/inactive pins; no alias expansion or global Boolean equivalence.',
            'Identical FF pins/attributes do not prove safe physical removal; timing/fanout and initialization must be checked.',
            'Hierarchy attribution is estimated by Efinity after optimization; partition totals are exact for this mapped netlist.',
            'Resource candidate has local functional gates but no same-source full-frame, placement or STA gate.',
            'Program-specific invariants require compiler checks and hardware rejection before narrowing a general interface.',
            'MAC-only bounds exclude transfer, quantization, tails, schedule and production system overhead.',
            'Analysis leaves all functional RTL and immutable evidence unchanged.'])
    OUT.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ('status','resources','direct_duplicate_fingerprints','dma_program_statistics','ideal_mac_only_bounds_ms')},ensure_ascii=False))


if __name__ == '__main__':
    main()
