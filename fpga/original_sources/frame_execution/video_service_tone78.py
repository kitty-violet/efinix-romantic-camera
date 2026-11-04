"""Use frozen prepared ISP worker for exact offline-generated curve78 data.

No FPGA programming, sensor access, Flash writes or auto-restart. The imported
worker retains its STOP/DDR/A guards, bounded register policy, CRC and all-byte
readback. This file adds only input-data binding and an outer host deadline.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from run_units import ROOT, sha
from video_input_isp_control import upload_plan
import video_service_input_isp_prepared as base
import video_input_tone78 as curve
from windows_process_tree_safe import stop_launched_tree


def payload(root):
    root = root.resolve()
    if not root.is_relative_to(ROOT/'tmp/board-test'):
        raise ValueError('Curve inputs must remain under project board-test')
    plan = json.loads((root/'plan.json').read_text(encoding='utf-8'))
    return curve.validate((root/'table.bin').read_bytes(), plan)


def verify_tests(path):
    proof = json.loads(path.read_text(encoding='utf-8'))
    if (proof.get('status') != 'PASS' or proof.get('tests', 0) < 9 or
        proof.get('runner_sha256') != sha(__file__) or
        proof.get('generator_sha256') != sha(curve.__file__) or
        proof.get('worker_sha256') != sha(base.__file__) or
        proof.get('test_sha256') != sha(Path(__file__).with_name('test_video_input_tone78.py'))):
        raise ValueError('Exact curve78/runner/worker tests required')


def run_worker(args):
    # All control logic stays in the unchanged release-bound worker.
    data = payload(args.curve_root)
    base.verified_tests(args.test_receipt)
    verify_tests(args.curve_test_receipt)
    args.data = data
    base.worker(args, upload_plan(data, True))


def main():
    p = argparse.ArgumentParser()
    for name in ('manifest', 'program-receipt', 'curve-root', 'test-receipt', 'curve-test-receipt'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--report', type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--supervised', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    for name in ('manifest', 'program_receipt', 'curve_root', 'test_receipt', 'curve_test_receipt'):
        setattr(a, name, getattr(a, name).resolve())
    data = payload(a.curve_root)
    base.bound(a.manifest, a.program_receipt)
    base.verified_tests(a.test_receipt)
    verify_tests(a.curve_test_receipt)
    if not a.execute:
        print(json.dumps(dict(status='TONE78_OFFLINE_BINDING_PASS_NO_USB', bytes=len(data),
                              crc32=upload_plan(data, True)['crc32']))); return
    if not a.supervised or a.report is None:
        raise ValueError('Onsite supervision and fresh report required')
    a.report = a.report.resolve()
    if not a.report.is_relative_to(ROOT/'tmp/board-test') or a.report.exists():
        raise ValueError('New project board-test report required')
    if a.worker:
        run_worker(a); return
    cmd = [sys.executable, str(Path(__file__).resolve())]
    for name in ('manifest','program_receipt','curve_root','test_receipt','curve_test_receipt','report'):
        cmd += ['--'+name.replace('_','-'), str(getattr(a, name))]
    cmd += ['--execute', '--supervised', '--worker']
    started = time.monotonic()
    with a.report.with_suffix('.log').open('x') as log:
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            rc = child.wait(timeout=630)
            result = dict(exit_code=rc, seconds=time.monotonic()-started)
        except subprocess.TimeoutExpired:
            rc = 124
            result = dict(exit_code=rc, seconds=time.monotonic()-started,
                          cleanup=stop_launched_tree(child))
    result.update(runner_sha256=sha(__file__), generator_sha256=sha(curve.__file__),
                  curve_plan_sha256=sha(a.curve_root/'plan.json'),
                  curve_test_sha256=sha(a.curve_test_receipt))
    a.report.with_suffix('.host.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
