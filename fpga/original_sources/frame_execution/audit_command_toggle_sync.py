"""Check mapped command-toggle synchronizer cells, independently of OFS CDC.

This is a structural check only. It does not check bundled payload timing,
response capture, routed FF separation, MTBF, or board operation.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from audit_snapshot_cells import cells
from audit_local_reset_implementation import norm

DEFAULT = 'cpu/system_hardJtag_debug_logic_jtagBridge/flowCCUnsafeByToggle_1/'


def audit(text, prefix=DEFAULT+'inputArea_target_buffercc/',
          source=DEFAULT+'inputArea_target', clock='process_clk_150'):
    complete = cells(text)
    found = {n: c for n, c in complete.items() if n.startswith(prefix)}
    errors = []
    params = {}
    for name, key, val in re.findall(r'^\s*defparam\s+(\S+)\s*\.(\w+)\s*=\s*([^;]+);', text, re.M):
        params.setdefault(norm(name), {})[key] = val.strip()
    first = [(n, c) for n, c in found.items() if c[0] == 'EFX_FF' and c[1].get('D') == source]
    second = []
    if len(first) == 1:
        # The second FF may be named after the instance's parent-level output
        # alias. Follow the actual wire instead of requiring a name prefix.
        second = [(n, c) for n, c in complete.items()
                  if c[0] == 'EFX_FF' and n != first[0][0] and c[1].get('D') == first[0][1][1].get('Q')]
        found.update(second)
    if len(found) != 2 or any(c[0] != 'EFX_FF' for c in found.values()):
        errors.append('Expected exactly two real destination EFX_FF cells; SRL/LUT replacement is rejected')
    if len(first) != 1 or len(second) != 1:
        errors.append('No unambiguous direct source -> FF1 -> FF2 chain')
    for n, (kind, ports, attrs) in found.items():
        if kind != 'EFX_FF':
            continue
        if ports.get('CLK') != clock or ports.get('CE') != "1'b1" or ports.get('SR') != "1'b0":
            errors.append(n+': incorrect clock/enable/reset')
        if not re.search(r'async_reg="true"', attrs, re.I):
            errors.append(n+': missing async_reg attribute')
        for key in ('CLK_POLARITY', 'CE_POLARITY', 'D_POLARITY'):
            if params.get(n, {}).get(key) != "1'b1":
                errors.append(n+': unexpected '+key)
    return dict(status='COMMAND_TOGGLE_TWO_REAL_FF_PASS' if not errors else 'COMMAND_TOGGLE_STRUCTURE_FAIL',
                errors=errors, cells={n: dict(kind=k, ports=p) for n, (k, p, _) in found.items()},
                source=source, destination_clock=clock,
                scope='Command toggle synchronizer structure only; not full JTAG CDC or board release')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('netlist', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.netlist.read_text())
    result.update(netlist=str(args.netlist), sha256=hashlib.sha256(args.netlist.read_bytes()).hexdigest())
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(result['status'])
    raise SystemExit(bool(result['errors']))
