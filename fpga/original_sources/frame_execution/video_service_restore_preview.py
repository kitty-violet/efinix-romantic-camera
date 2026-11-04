"""Restore stopped INPUT_PREVIEW after inspection/ISP, using frozen tools.

No STOP, download, Flash, mode change, ISP update or compute/DMA upload.
Default prints a plan without USB access. Never infer pattern from the ISP mailbox.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from run_units import ROOT, HERE, sha
from video_service_board import bound
from windows_process_tree_safe import stop_launched_tree


def commands(manifest, receipt, root):
    common = ['--manifest', str(manifest), '--program-receipt', str(receipt)]
    board = str(HERE / 'video_service_board.py')
    prep = str(HERE / 'video_service_reprepare_preview.py')
    return [(name, [sys.executable, script, *common, '--report', str(root / (name+'.json')),
                    *options, '--execute', '--supervised']) for name, script, options in (
        ('clear', board, ['--operation', 'diagnose', '--tx-pattern', 'off', '--inspect', 'clear']),
        ('prepare', prep, []),
        ('live', board, ['--operation', 'live']))]


def verify(name, row):
    if row.get('control', {}).get('errors') != 0:
        raise ValueError(name + ': control errors/missing evidence')
    if name == 'clear':
        path = row['diagnostic']['path']
        if row['diagnostic']['word'] != 0x56531000 or path['mode'] != 1 or path['inspect_enabled']:
            raise ValueError('Explicit pattern-off/inspection-clear in preview required')
    elif name == 'prepare':
        if row['status'] != 'INPUT_PREVIEW_PREPARED_NOT_RUNNING_NOT_CNN_RELEASE':
            raise ValueError('Preparation not complete')
        if row['path']['mode'] != 1 or row['path']['inspect_enabled']:
            raise ValueError('Wrong prepared mode')
        if row['after']['phase'] != 5 or row['after']['apb']['08'] or row['after']['apb']['60']:
            raise ValueError('Prepared idle service required')
    else:
        samples = row['register_samples']
        last = samples[-1]
        paths = [p['fields'] for p in row['pages'] if p['page_kind'] == 'path']
        if last['phase'] != 4 or last['apb']['08'] != 3 or last['apb']['60']:
            raise ValueError('LIVE did not remain healthy')
        if not paths or any(p['mode'] != 'INPUT_PREVIEW' or p['inspect_enabled'] for p in paths):
            raise ValueError('Not live original preview')
        if last['apb']['5c'] <= samples[0]['apb']['5c']:
            raise ValueError('No new committed frame or reset/wrap: cannot claim preview')


def sequence(plan, execute, read, save):
    for name, command in plan:
        result = execute(name, command)
        save(name, result)
        if result != 0:
            raise RuntimeError(name + ': stopped at first failed child; no retry')
        row = read(name)
        verify(name, row)
    return 'INPUT_PREVIEW_RESTORED_NEW_COMMIT_NOT_VISUAL_ACCEPTANCE'


def main():
    parser = argparse.ArgumentParser()
    for field in ('manifest', 'program-receipt', 'root'):
        parser.add_argument('--'+field, required=True, type=Path)
    parser.add_argument('--test-receipt', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--supervised', action='store_true')
    args = parser.parse_args()
    manifest, receipt, root = [p.resolve() for p in (args.manifest, args.program_receipt, args.root)]
    bound(manifest, receipt)
    plan = commands(manifest, receipt, root)
    if not args.execute:
        print(json.dumps(dict(hardware_access=False, commands=plan))); return
    if not args.supervised or not root.is_relative_to(ROOT/'tmp/board-test') or root.exists():
        raise ValueError('New supervised output directory required')
    proof = json.loads(args.test_receipt.read_text(encoding='utf-8'))
    if proof['status'] != 'PASS' or proof['helper_sha256'] != sha(__file__) or proof['test_sha256'] != sha(HERE/'test_video_service_restore_preview.py'):
        raise ValueError('Bound regression tests required')
    # Reprepare is an independent helper, so bind its exact tested bytes too.
    if proof['prepare_sha256'] != sha(HERE/'video_service_reprepare_preview.py'):
        raise ValueError('Preparation helper changed')
    root.mkdir()
    report = dict(status='RESTORING', manifest_sha256=sha(manifest), program_receipt_sha256=sha(receipt),
                  helper_sha256=sha(__file__), cnn_started=False, stages=[])
    def save(name=None, code=None):
        if name: report['stages'].append(dict(stage=name, exit_code=code))
        (root/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    def execute(name, command):
        with (root/(name+'.outer.log')).open('x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try: return process.wait(timeout=240)
            except subprocess.TimeoutExpired:
                report['timeout_cleanup'] = stop_launched_tree(process)
                return 124
    save()
    try:
        report['status'] = sequence(plan, execute, lambda name: json.loads((root/(name+'.json')).read_text()), save)
    except Exception as exc:
        report.update(status='FAIL_NO_AUTOMATIC_RETRY', error=repr(exc)); raise
    finally: save()
    print(report['status'])


if __name__ == '__main__': main()
