"""Bounded route-only experiment on a hash-locked existing placement.

This restarts routing from placement; it does not resume failed route state.
The short probe stops as soon as the requested iteration limit is exceeded.
"""
import argparse
import json
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from run_units import ROOT, EF, sha
from windows_process_tree import stop_launched_tree


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--mode', choices=('cli', 'xml', 'ini', 'troute', 'inspect', 'extended'), required=True)
    p.add_argument('--iterations', type=int, default=2)
    p.add_argument('--budget', type=int, default=120)
    p.add_argument('--preflight', type=Path)
    a = p.parse_args()
    source, root = a.source.resolve(), a.root.resolve()
    assert source.is_relative_to(ROOT / 'tmp') and root.is_relative_to(ROOT / 'tmp')
    assert not root.exists() and 1 <= a.iterations <= 150 and 10 <= a.budget <= 2100
    if a.iterations > 3 and a.mode not in ('inspect', 'extended'):
        assert a.preflight, 'Long run requires a successful same-mode short cap test'
        proof = json.loads(a.preflight.read_text(encoding='utf-8'))
        assert proof['status'] == 'ITERATION_LIMIT_EFFECTIVE'
        assert proof['mode'] == a.mode and proof['source'] == str(source)
    old = source / 'pnr-seed1'
    prior = json.loads((old / 'result.json').read_text())
    meta = json.loads((old / 'physical_inputs.json').read_text())
    assert any(s['flow'] == 'pnr' and s['exit_code'] != 0 for s in prior['stages'])
    assert meta['setup_margin'] is None
    frozen_names = ('outflow/ti60f225_oob.place', 'outflow/ti60f225_oob.out.placeloc',
                    'outflow/ti60f225_oob.vdb', 'outflow/ti60f225_oob.map.v',
                    'ti60f225_oob.sdc', 'fixed_dsp.placeloc',
                    'outflow/ti60f225_oob.interface.csv')
    frozen = {name: sha(old / name) for name in frozen_names}
    assert frozen['outflow/ti60f225_oob.map.v'] == meta['base_map_sha256']
    for row in meta['inputs']:
        assert sha(row['snapshot']) == row['sha256']
    root.mkdir()
    physical = root / 'pnr-seed1'
    shutil.copytree(old, physical)
    for row in meta['inputs']:
        row['snapshot'] = str(physical / Path(row['snapshot']).relative_to(old))
        assert sha(row['snapshot']) == row['sha256']
    save(physical / 'physical_inputs.json', meta)
    for name in ('result.json', 'stages.json', 'outflow/ti60f225_oob.route.out'):
        path = physical / name
        if path.exists():
            path.rename(path.with_name('source_' + path.name))
    xml = physical / 'ti60f225_oob.xml'
    settings = physical / 'efx_pnr_settings.ini'
    assert 'max_router_iterations' not in xml.read_text()
    assert 'max_router_iterations' not in settings.read_text()
    if a.mode == 'xml':
        body = xml.read_text()
        anchor = '    </efx:place_and_route>'
        assert body.count(anchor) == 1
        body = body.replace(anchor, '        <efx:param name="max_router_iterations" value="'
                            + str(a.iterations) + '" value_type="e_integer" />\n' + anchor)
        xml.write_text(body)
        assert [e.get('value') for e in ET.parse(xml).iter()
                if e.get('name') == 'max_router_iterations'] == [str(a.iterations)]
    elif a.mode == 'ini':
        settings.write_text(settings.read_text() + '\nmax_router_iterations = ' + str(a.iterations) + '\n')
    args = [str(EF / 'bin/efx_pnr.exe'), '--circuit', 'ti60f225_oob', '--family', 'Titanium',
            '--device', 'Ti60F225', '--operating_conditions', 'I3', '--vdb_file',
            'outflow/ti60f225_oob.vdb', '--use_vdb_file', 'on', '--prj', str(xml),
            '--output_dir', 'outflow', '--work_dir', 'work_pnr', '--place_file',
            'outflow/ti60f225_oob.place', '--route_file', 'outflow/ti60f225_oob.route',
            '--settings_file', str(settings), '--sync_file', 'outflow/ti60f225_oob.interface.csv',
            '--generate_prevpr_netlist', 'off', '--sdc_file=ti60f225_oob.sdc', '--verbose', '--route']
    if a.mode == 'cli':
        args += ['--max_router_iterations', str(a.iterations)]
    elif a.mode == 'troute':
        # The installed Titanium router advertises max_iterations internally;
        # the outer VPR-compatible max_router_iterations did not limit it.
        # Prove this transport in a short run before using a higher limit.
        args += ['--troute_options=max_iterations=' + str(a.iterations)]
    elif a.mode == 'inspect':
        args += ['--troute_options=max_iterations=' + str(a.iterations)
                 + ',dump_config,dry_run,print_troute_command']
    elif a.mode == 'extended':
        # Low-limit probes could not establish whether a recipe imposes a
        # minimum. This is an explicitly bounded capability experiment, not
        # a claim that the setting works. Only observing iteration >70 proves it.
        args += ['--max_router_iterations', str(a.iterations),
                 '--troute_options=max_iterations=' + str(a.iterations)]
    cmd = physical / 'route_only.cmd'
    cmd.write_text('@echo off\ncall "' + str(EF / 'bin/setup.bat') + '"\n'
                   'if errorlevel 1 exit /b 1\n' + subprocess.list2cmdline(args) + '\nexit /b %errorlevel%\n')
    info = dict(source=str(source), mode=a.mode, requested_iterations=a.iterations,
                budget_s=a.budget, source_result_sha256=sha(old / 'result.json'),
                frozen=frozen, command=args, script_sha256=sha(__file__),
                xml_sha256=sha(xml), settings_sha256=sha(settings),
                preflight_sha256=sha(a.preflight) if a.preflight else None,
                cap_prevalidated=a.mode != 'extended',
                resumed_failed_route=False, hardware_access=False, board_release=False)
    save(root / 'inputs.json', info)
    shutil.copy2(__file__, root / 'runner.py')
    started = time.monotonic()
    route_path = physical / 'outflow/ti60f225_oob.route.out'
    reason, cleanup = None, None
    with (root / 'route.log').open('x') as log:
        proc = subprocess.Popen(['cmd.exe', '/d', '/c', str(cmd)], cwd=physical,
                                stdout=log, stderr=subprocess.STDOUT)
        save(root / 'progress.json', dict(status='running', pid=proc.pid, **info))
        while proc.poll() is None:
            route = route_path.read_text(errors='replace') if route_path.exists() else ''
            iterations = re.findall(r'Iter\s+(\d+):\s*overlap=(\d+)', route)
            if re.search(r'^\s*ERROR\s*:.*(?:Cannot parse|must be provided|Unknown option)', route, re.M):
                reason = 'ROUTER_OPTION_PARSE_FAILED'
            elif a.iterations <= 3 and iterations and int(iterations[-1][0]) > a.iterations:
                reason = 'REQUESTED_LIMIT_EXCEEDED'
            elif time.monotonic() - started > a.budget:
                reason = 'HOST_BUDGET_EXCEEDED'
            if reason:
                cleanup = stop_launched_tree(proc)
                save(root / 'stop.json', dict(reason=reason, cleanup=cleanup))
                break
            time.sleep(0.5)
    code = proc.poll()
    route = route_path.read_text(errors='replace') if route_path.exists() else ''
    iterations = [(int(i), int(n)) for i, n in re.findall(r'Iter\s+(\d+):\s*overlap=(\d+)', route)]
    for name, digest in frozen.items():
        assert sha(old / name) == digest, ('source mutated', name)
        assert sha(physical / name) == digest, ('placement/netlist/constraint changed', name)
    legal = code == 0 and bool(iterations) and iterations[-1][1] == 0
    limit_effective = not reason and code == 1 and bool(iterations) and iterations[-1][0] == a.iterations
    result = dict(**info, status=('ROUTE_PASS_PENDING_STA' if legal else
                  'ITERATION_LIMIT_EFFECTIVE' if limit_effective else reason or 'ROUTE_FAILED'),
                  exit_code=code, seconds=time.monotonic()-started,
                  iterations=iterations, route_sha256=sha(route_path) if route_path.exists() else None,
                  extended_beyond_70=bool(iterations) and iterations[-1][0] > 70,
                  source_and_copy_frozen_hashes_pass=True)
    save(root / 'result.json', result)
    save(root / 'progress.json', result)
    if legal:
        save(physical / 'result.json', dict(status='PRESERVED_PLACEMENT_ROUTE_PASS_PENDING_STA',
             stages=[*[s for s in prior['stages'] if s['flow'] == 'interface'],
                     dict(flow='pnr', exit_code=0, seconds=result['seconds'], route_only=True)],
             board_release=False))
    print(json.dumps({k:result[k] for k in ('status','mode','exit_code','seconds','iterations')}))


if __name__ == '__main__':
    main()
