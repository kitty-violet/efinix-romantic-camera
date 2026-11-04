"""Finish accounting for an already running, frozen CSI-to-TMDS measurement.

Does not start, modify or stop a simulator. Independent Python accounting jobs
run concurrently after the simulator has written its final result. Only the
strict publisher may label a report complete. Failed runs retain partial
diagnostics and return failure; they never produce a complete-board report.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
from run_units import ROOT, HERE, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board', type=Path, required=True)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    board, mapped, root, report = [p.resolve() for p in
        (args.board, args.map, args.root, args.report)]
    assert board.is_relative_to(ROOT/'tmp') and mapped.is_relative_to(ROOT/'tmp')
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    assert report.is_relative_to(ROOT/'output') and not report.exists()
    scripts = ['summarize_board_measurement.py', 'summarize_board_traffic.py',
               'summarize_board_command_profile.py', 'publish_board_measurement.py']
    inputs = dict(board=str(board), mapped=str(mapped), report=str(report),
        board_inputs_sha256=sha(board/'inputs.json'),
        map_inputs_sha256=sha(mapped/'inputs.json'), runner_sha256=sha(__file__),
        analyzers={name: sha(HERE/name) for name in scripts},
        simulator_started=False, hardware_modified=False)
    root.mkdir()
    (root/'inputs.json').write_text(json.dumps(inputs, indent=2), encoding='utf8')
    print('WAITING_FOR_FROZEN_BOARD_RESULT '+str(board), flush=True)
    started = time.monotonic()
    while not (board/'run_result.json').exists():
        if time.monotonic()-started > 24*3600:
            raise TimeoutError('No final simulator result; no complete measurement published')
        time.sleep(2)
    run = json.loads((board/'run_result.json').read_text(encoding='utf8'))
    assert run['inputs_sha256'] == inputs['board_inputs_sha256'] == sha(board/'inputs.json')
    assert inputs['map_inputs_sha256'] == sha(mapped/'inputs.json')
    assert run['log_sha256'] == sha(board/'simulation.log')
    for name in scripts:
        assert sha(HERE/name) == inputs['analyzers'][name], name

    def execute(script, *arguments):
        command = [sys.executable, '-B', str(HERE/script), *map(str, arguments)]
        begin = time.monotonic()
        path = root/(Path(script).stem+'.log')
        with path.open('wb') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=300)
        row = dict(script=script, returncode=result.returncode,
                   seconds=time.monotonic()-begin, log_sha256=sha(path))
        print('ACCOUNTING_STAGE '+json.dumps(row), flush=True)
        return row

    # These outputs are independent and consume only the finalized evidence.
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(execute, name, '--root', board) for name in scripts[:3]]
        stages = [job.result() for job in jobs]
    success = (run['status']=='BOARD_DIGITAL_MEASUREMENT_COMPLETE'
               and run['returncode']==0 and not run['failures']
               and run['ddr_model_error_total']==0)
    if success and all(row['returncode']==0 for row in stages):
        stages.append(execute(scripts[3], '--board', board, '--map', mapped, '--output', report))
    published = success and len(stages)==4 and all(row['returncode']==0 for row in stages)
    result = dict(status='UNIFIED_BOARD_MEASUREMENT_PUBLISHED' if published else
                  'BOARD_MEASUREMENT_INCOMPLETE_OR_FAILED', stages=stages,
        wait_seconds=time.monotonic()-started, run_sha256=sha(board/'run_result.json'),
        inputs_sha256=sha(root/'inputs.json'), simulator_status=run['status'],
        complete_board_measurement=published, physical_pass=False,
        report=str(report) if published else None,
        report_sha256=sha(report) if published else None)
    (root/'result.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result), flush=True)
    if not published:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
