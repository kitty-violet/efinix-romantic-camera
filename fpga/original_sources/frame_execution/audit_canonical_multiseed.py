"""Rejudge immutable routed STA receipts under the current explicit policy.

Older +0.300ns FAIL receipts remain untouched. This checks their actual reports,
route/input hashes and independent seeds, not their obsolete pass/fail label.
It is a physical evidence gate only; it cannot release hardware programming.
"""
import argparse
import json
import re
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT, sha
from timing_acceptance import timing_passes


def audit(folder):
    folder = folder.resolve()
    receipt = json.loads((folder / 'result.json').read_text())
    manifest = json.loads((folder / 'inputs.json').read_text())
    assert sha(folder / 'inputs.json') == receipt['inputs_sha256']
    assert receipt['original_and_route_unchanged']
    base = Path(manifest['base'])
    pm = json.loads((base / 'physical_inputs.json').read_text())
    assert sha(base / 'physical_inputs.json') == manifest['base_physical_sha256']
    for row in pm['inputs']:
        assert sha(row['snapshot']) == row['sha256'], row['snapshot']
    for name, digest in manifest['original'].items():
        assert sha(name) == digest, name
    for name, digest in receipt['reports'].items():
        assert sha(folder / 'outflow' / name) == digest, name
    for suffix in ('.troutingtraces', '.place', '.vdb', '.map.v'):
        name = 'ti60f225_oob' + suffix
        assert sha(folder / 'outflow' / name) == sha(base / 'outflow' / name)
    assert sha(folder / 'ti60f225_oob.sdc') == manifest['canonical_sdc_sha256']
    physical = json.loads((base / 'result.json').read_text())
    assert any(s['flow'] == 'pnr' and s['exit_code'] == 0 for s in physical['stages'])
    route = (base / 'outflow/ti60f225_oob.route.out').read_text()
    iterations = re.findall(r'Iter\s+\d+:\s*overlap=(\d+)', route)
    assert iterations and int(iterations[-1]) == 0
    slacks = {}
    for kind in ('setup', 'hold'):
        item = receipt['rows'][kind]
        report = Path(item['report'])
        assert report.resolve() == (folder / 'outflow' / ('canonical_' + kind + '.rpt')).resolve()
        assert sha(report) == item['sha256']
        slacks[kind] = min(map(float, re.findall(r'^Slack\s*:\s*([+\-\d.]+)', report.read_text(), re.M)))
        assert slacks[kind] == item['worst_ns']
    checks_text = (folder / 'outflow/canonical_checks.rpt').read_text()
    checks = {k: int(v) for k, v in re.findall(r'checking (\w+) \((\d+)\)', checks_text)}
    for direction in ('input', 'output'):
        checks['no_' + direction + '_delay'] = int(re.search(
            r'There are (\d+) ' + direction + r' ports with no ' + direction + ' delay specified', checks_text)[1])
    assert checks == receipt['check_timing_actual']
    log = (folder / 'sta.log').read_text(errors='replace')
    log += '\n'.join(p.read_text(errors='replace') for p in (folder / 'outflow').glob('*.sta.out'))
    assert not re.search(r'Unable to run|No (?:ports|pins|clocks) matched|Timer cuts a combinational loop|^\s*(?:ERROR|FATAL)\s*:', log, re.M)
    place = (base / 'outflow/ti60f225_oob.place.rpt').read_text()
    resources = {}
    for label in ('XLRs', 'Memory Blocks', 'DSP Blocks'):
        used, limit = map(int, re.search(r'^' + label + r':\s*(\d+)\s*/\s*(\d+)', place, re.M).groups())
        assert used <= limit
        resources[label] = dict(used=used, capacity=limit)
    # Same RTL, pins, map, optimization constraints and physical directives.
    xml = ET.parse(base / 'ti60f225_oob.xml')
    seed_nodes = [n for n in xml.iter() if n.attrib.get('name') == 'seed']
    assert len(seed_nodes) == 1 and int(seed_nodes[0].attrib['value']) == pm['seed_change']['seed']
    seed_nodes[0].set('value', '0')
    # The recipe comment changed when +0.300ns became an optimization target.
    # Ignore full-line comments/blank lines only, never timing commands.
    sdc_commands = '\n'.join(line.strip() for line in (base / 'ti60f225_oob.sdc').read_text().splitlines()
                             if line.strip() and not line.lstrip().startswith('#'))
    sdc_command_sha = hashlib.sha256(sdc_commands.encode('utf-8')).hexdigest()
    identity = dict(map=pm['base_map_sha256'], canonical_sdc=manifest['canonical_sdc_sha256'],
                    project_except_seed_sha256=hashlib.sha256(ET.tostring(xml.getroot())).hexdigest(),
                    optimization_sdc_commands_sha256=sdc_command_sha,
                    optimization=pm['optimization_change']['value'],
                    placement=sha(base / 'fixed_dsp.placeloc'),
                    settings=sha(base / 'efx_pnr_settings.ini'),
                    inputs={str(Path(r['snapshot']).relative_to(base)): r['sha256'] for r in pm['inputs']
                            if Path(r['snapshot']).name not in ('ti60f225_oob.xml', 'ti60f225_oob.sdc')})
    return dict(seed=pm['seed_change']['seed'], physical=str(base), canonical=str(folder),
                original_status=receipt['status'], current_timing_pass=timing_passes(slacks['setup'], slacks['hold'], checks, 0),
                wns_ns=slacks['setup'], whs_ns=slacks['hold'], checks=checks, resources=resources,
                canonical_result_sha256=sha(folder / 'result.json'), physical_inputs_sha256=sha(base / 'physical_inputs.json'),
                optimization_sdc_file_sha256=sha(base / 'ti60f225_oob.sdc'),
                identity=identity)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--roots', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    # Offline batch receipts may live beside their immutable seed evidence.
    # This changes only the report destination, never timing/capacity gates.
    assert (out.is_relative_to(ROOT / 'output') or out.is_relative_to(ROOT / 'tmp/frame-execution')) and not out.exists()
    rows = [audit(r) for r in args.roots]
    assert len(rows) >= 4 and len({r['seed'] for r in rows}) == len(rows), 'Need four distinct seeds'
    assert all(r['identity'] == rows[0]['identity'] for r in rows), 'Configuration differs beyond seed'
    assert all(r['current_timing_pass'] for r in rows), 'Actual setup/hold/checks fail'
    identity = rows[0]['identity']
    for row in rows:
        del row['identity']
    result = dict(status='FOUR_SEED_PHYSICAL_BASIC_TIMING_PASS_NOT_BOARD_RELEASE',
                  policy=dict(min_setup_ns=0, min_hold_ns=0, extra_0p300_deferred=True, logic_limit=60800),
                  rows=rows, identity=identity, historical_receipts_unchanged=True, board_release=False)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'rows')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
