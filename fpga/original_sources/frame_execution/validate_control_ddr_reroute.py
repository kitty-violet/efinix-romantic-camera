"""Validate a successful bounded reroute with the original timing/CDC gates."""
import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from run_units import ROOT, HERE, EF, sha
from control_ddr_combined import audit_delta
from run_control_ddr_physical import audit_comb4
from audit_canonical_multiseed import audit
from windows_process_tree import stop_launched_tree


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    assert root.is_relative_to(ROOT / 'tmp')
    result = json.loads((root / 'result.json').read_text())
    assert result['status'] == 'ROUTE_PASS_PENDING_STA' and result['exit_code'] == 0
    assert result['source_and_copy_frozen_hashes_pass']
    assert not (root / 'route_result.json').exists()
    physical = root / 'pnr-seed1'
    route = physical / 'outflow/ti60f225_oob.route.out'
    assert sha(route) == result['route_sha256']
    source = Path(result['source'])
    meta = json.loads((physical / 'physical_inputs.json').read_text())
    board = Path(meta['base'])
    audit_comb4(board)
    for name, digest in result['frozen'].items():
        assert sha(source / 'pnr-seed1' / name) == sha(physical / name) == digest
    source_meta = json.loads((source / 'inputs.json').read_text())
    assert meta['base_map_sha256'] == source_meta['map_sha256']
    assert meta['base_inputs_sha256'] == source_meta['board_inputs_sha256'] == sha(board / 'inputs.json')
    (root / 'result.json').rename(root / 'route_result.json')
    shutil.copytree(source / 'startup', root / 'startup')
    for name in ('command_sync.json', 'short_evidence_reuse.json'):
        shutil.copy2(source / name, root / name)
    reused = json.loads((root / 'short_evidence_reuse.json').read_text())
    for path, digest in reused.items():
        assert sha(path) == digest
    serial = ROOT / 'tmp/frame-execution/control-ddr-combined-physical2-serial'
    serial_input = json.loads((serial / 'new/inputs.json').read_text())
    assert serial_input['source_map_sha256'] == meta['base_map_sha256']
    # Keep the route runner inputs untouched. Release binding also verifies its
    # runner hash, so use that existing key rather than inventing new ancestry.
    inputs = json.loads((root / 'inputs.json').read_text())
    inputs['runner_sha256'] = sha(root / 'runner.py')
    save(root / 'inputs.json', inputs)
    frozen = {str(source / 'pnr-seed1' / n): h for n, h in result['frozen'].items()}
    save(root / 'route_continuation.json', dict(source_batch=str(source), frozen=frozen,
         source_inputs_sha256=sha(source / 'inputs.json'),
         route_result_sha256=sha(root / 'route_result.json'),
         command_sha256=sha(physical / 'route_only.cmd'),
         official_doc=str(EF / 'doc/topics/cli-pnr-opts.html'),
         official_doc_sha256=sha(EF / 'doc/topics/cli-pnr-opts.html'),
         rtl_changed=False, map_changed=False, placement_changed=False, constraints_changed=False,
         resumed_failed_route=False, hardware_access=False))
    shutil.copy2(__file__, root / 'validation_runner.py')
    stages = []

    def run(name, script, args, budget):
        started = time.monotonic()
        with (root / (name + '.log')).open('x') as log:
            proc = subprocess.Popen([sys.executable, str(HERE / script), *map(str, args)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row = dict(name=name, status='running', pid=proc.pid, budget_s=budget)
            stages.append(row)
            save(root / 'stages.json', stages)
            try:
                code = proc.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                save(root / (name + '-timeout.json'), stop_launched_tree(proc))
                raise
        row.update(status='finished', exit_code=code, seconds=time.monotonic()-started)
        save(root / 'stages.json', stages)
        assert code == 0, (name, code)

    try:
        import audit_control_state_implementation as impl
        impl.audit_delta = audit_delta
        save(root / 'implementation.json', impl.audit(board, physical))
        run('canonical', 'canonical_snapshot_sta.py',
            ['--base', physical, '--root', root / 'canonical-seed1'], 270)
        selected = audit(root / 'canonical-seed1')
        save(root / 'selected.json', selected)
        assert selected['current_timing_pass'], (selected['wns_ns'], selected['whs_ns'])
        run('gray', 'audit_command_progress_gray.py',
            ['--base', physical, '--root', root / 'gray-seed1'], 380)
        save(root / 'result.json', dict(status='COMMAND_PROGRESS_PHYSICAL_PASS_PENDING_FUNCTIONAL_CDC_AND_RELEASE',
             selected=selected, serial=str(serial), gray=str(root / 'gray-seed1'),
             route_continuation_sha256=sha(root / 'route_continuation.json'),
             validation_runner_sha256=sha(root / 'validation_runner.py'),
             hardware_access=False, board_release=False))
    except Exception as exc:
        save(root / 'validation_failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), hardware_access=False))
        raise


if __name__ == '__main__':
    main()
