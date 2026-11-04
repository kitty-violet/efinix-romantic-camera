"""Read-only Boolean duplicate scan; never rewrite a mapped implementation.

Reports identical functions of identical nets, including input permutations.
It cannot declare four-state equivalence or predict placed timing/area savings.
"""
import argparse
import collections
import json
import re
from pathlib import Path
from run_units import sha


def canonical(inputs, mask):
    constants = {"1'b0": 0, "1'h0": 0, "1'b1": 1, "1'h1": 1}
    names = sorted(set(n for n in inputs if n not in constants))
    values = []
    for word in range(1 << len(names)):
        environment = {name: (word >> bit) & 1 for bit, name in enumerate(names)}
        environment.update(constants)
        address = sum(environment[name] << bit for bit, name in enumerate(inputs))
        values.append((mask >> address) & 1)
    # Remove logically irrelevant inputs after constant cofactoring. This is
    # intentionally labelled two-state and not a simulation-model rewrite.
    live = [bit for bit in range(len(names)) if any(
        values[word] != values[word ^ (1 << bit)] for word in range(len(values)))]
    reduced = 0
    for word in range(1 << len(live)):
        old = sum(((word >> i) & 1) << bit for i, bit in enumerate(live))
        reduced |= values[old] << word
    return tuple(names[bit] for bit in live), reduced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    # Truth-table convention sanity: I0 muxes I1/I2; AND and unused pins.
    assert canonical(['a', 'b', "1'b0", "1'b0"], 0x8888) == (('a', 'b'), 8)
    assert canonical(['b', 'a', "1'b0", "1'b0"], 0x8888) == (('a', 'b'), 8)
    assert canonical(['a', 'a', "1'b0", "1'b0"], 0x8888) == (('a',), 2)
    text = args.net.read_text()
    pattern = re.compile(r'^\s+EFX_LUT4\s+(\S+)\s+\((.*?)\)\s*/\*[^*]*?'
                         r'LUTMASK=16\'h([0-9a-fA-F]{4})[^*]*\*/', re.M | re.S)
    nodes = {}
    for match in pattern.finditer(text):
        ports = {name: re.sub(r'\s+', '', value) for name, value in re.findall(r'\.(\w+)\((.*?)\)', match[2], re.S)}
        assert 'O' in ports and set(ports) <= {'I0', 'I1', 'I2', 'I3', 'O'}, match[1]
        assert ports['O'] not in nodes
        # The official EFX_LUT4 public model pulls omitted inputs weak-low.
        nodes[ports['O']] = dict(instance=match[1], inputs=tuple(ports.get(f'I{i}', "1'b0") for i in range(4)), mask=int(match[3], 16))
    assert len(nodes) == len(re.findall(r'^\s+EFX_LUT4\s', text, re.M))
    groups = collections.defaultdict(list)
    for out, node in nodes.items():
        groups[canonical(node['inputs'], node['mask'])].append(out)
    duplicates = [values for values in groups.values() if len(values) > 1]
    repeated = sum(len(group)-1 for group in duplicates)
    # Only combinational LUTs are recursively matched; every FF/RAM/DSP output
    # remains a separate leaf, even if its name resembles another register.
    signatures, active, intern = {}, set(), {}
    def signature(net):
        if net not in nodes:
            return net
        if net in signatures:
            return signatures[net]
        assert net not in active, 'Combinational LUT loop: '+net
        active.add(net)
        node = nodes[net]
        key = canonical([signature(x) for x in node['inputs']], node['mask'])
        if key not in intern:
            intern[key] = f'@lut_function_{len(intern)}'
        signatures[net] = intern[key]
        active.remove(net)
        return signatures[net]
    for net in nodes:
        signature(net)
    result = dict(status='MAPPED_LUT_BOOLEAN_DUPLICATE_AUDIT', netlist=str(args.net.resolve()),
                  netlist_sha256=sha(args.net), source_sha256=sha(Path(__file__)),
                  total_luts=len(nodes), identical_net_function_duplicate_groups=len(duplicates),
                  identical_net_function_excess_cells=repeated,
                  recursive_function_excess_cells=len(nodes)-len(intern),
                  exact_input_duplicate_fraction=repeated/len(nodes),
                  largest_direct_groups=[dict(count=len(group), outputs=group[:12],
                                              cells=[nodes[out]['instance'] for out in group[:12]])
                                         for group in sorted(duplicates, key=len, reverse=True)[:12]],
                  limitations=['Two-state structural audit, not a four-state equivalence proof.',
                               'Equivalent cones may intentionally be replicated for fanout/timing.',
                               'Neither count is guaranteed removable area; no netlist was edited.',
                               'Similar arithmetic on distinct operand nets is not a duplicate.'])
    args.output.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in result.items() if k != 'largest_direct_groups'}))


if __name__ == '__main__':
    main()
