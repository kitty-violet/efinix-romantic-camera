"""Preserve useful evidence from a host-budget interruption, never a frame PASS."""
import argparse
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    assert root.is_relative_to(Path('E:/fpga/tmp').resolve())
    if args.finalize:
        archived=json.loads((root/'partial_measurement.json').read_text(encoding='utf8'))
        target=root/'partial_prefix';target.mkdir(exist_ok=False)
        rows=[]
        for name,digest in archived['input_hashes'].items():
            data=(root/name).read_bytes()
            if hashlib.sha256(data).hexdigest()!=digest:
                assert name=='simulation.log', name
                # The intentional process shutdown appended a license-process
                # exit message. Retain the exact measured prefix by its hash.
                h=hashlib.sha256();n=0;matched=False
                for line in data.splitlines(keepends=True):
                    h.update(line);n+=len(line)
                    if h.hexdigest()==digest:
                        data=data[:n];matched=True;break
                assert matched, 'Original log prefix cannot be recovered'
            (target/name).write_bytes(data)
            assert sha(target/name)==digest
            rows.append(dict(path=str(target/name),sha256=digest))
        result=dict(status='PARTIAL_EVIDENCE_PREFIX_HASH_VERIFIED',files=rows,
            complete_board=False,final_log_sha256=sha(root/'simulation.log'),
            shutdown_message_is_intentional='Exiting VSIM license process' in (root/'simulation.log').read_text(encoding='utf8'))
        (root/'partial_archive.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        print(json.dumps(result));return
    hold = json.loads((root / 'host_suspend.json').read_text(encoding='utf8'))
    assert sha(root / 'inputs.json') == hold['inputs_sha256']
    log = (root / 'simulation.log').read_text(encoding='utf8')
    with (root / 'cnn_command_events.csv').open(encoding='utf8', newline='') as f:
        commands = list(csv.DictReader(f))
    names = ('BOARD_CAL_DONE', 'BOARD_CPU_RUN', 'CAMERA_PORT_BEGIN',
             'RAW_FRAME_STORED', 'RAW_FRAME_READ_DONE', 'RGB_FRAME_COMPLETE',
             'CNN_BEGIN', 'CNN_COMMIT', 'SCREEN_COMPLETE')
    events = []
    for line in log.splitlines():
        for name in names:
            if re.search(r'\b' + name + r'\b', line):
                found = re.search(r't_ns=([\d.]+)', line)
                events.append(dict(event=name, t_ns=float(found[1]) if found else None,
                                   line=line))
                break
    assert not any(e['event'] in ('CNN_COMMIT', 'SCREEN_COMPLETE') for e in events), 'Reassess archive classification'
    errors = [line for line in log.splitlines() if re.search(r'\*\* (Fatal|Error)| ERROR: |BOARD_FAIL_FAST', line)]
    metrics = {}
    def stamp(event, marker):
        return next(e['t_ns'] for e in events if e['event'] == event and marker in e['line'])
    camera = stamp('CAMERA_PORT_BEGIN', 'camera=1 ')
    raw = stamp('RAW_FRAME_STORED', 'frame=0 ')
    cnn = stamp('CNN_BEGIN', 'frame=0 ')
    metrics.update(camera_to_raw_stored_ms=(raw-camera)/1e6,
                   raw_stored_to_cnn_begin_ms=(cnn-raw)/1e6,
                   camera_to_cnn_begin_ms=(cnn-camera)/1e6)
    hashes = {name: sha(root/name) for name in ('inputs.json', 'simulation.log',
        'cnn_command_events.csv', 'progress.json', 'host_suspend.json')}
    result = dict(status='PARTIAL_PREFIX_ONLY_HOST_BUDGET_INTERRUPTION',
        timestamp=datetime.now().astimezone().isoformat(), complete_frame=False,
        hardware_failure_observed=bool(errors), errors=errors, metrics=metrics,
        completed_compute_commands=len(commands), compute_commands_per_frame=8789,
        last_completed_command=commands[-1] if commands else None,
        last_bus_profile=next((line for line in reversed(log.splitlines()) if 'BOARD_PROFILE event=periodic' in line), None),
        milestones=events, input_hashes=hashes,
        useful_for=['startup and capture prefix timings', 'DDR traffic/response prefix',
                    'completed CNN command windows', 'host execution speed analysis',
                    'same-source resource analysis using connected-board-native-lock1'],
        not_proven=['complete frame latency', 'frame interval or FPS', 'full CNN output',
                    'complete TMDS output', 'P&R/STA or board behavior'])
    path = root/'partial_measurement.json'
    assert not path.exists(), 'Preserve earlier archive'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    interruption = dict(status='INTERRUPTED_FOR_HOST_RUNTIME_BUDGET',
        reason='User cannot wait for the current pure Questa run and authorizes redesigned co-simulation evaluation.',
        partial_measurement_sha256=sha(path), kernel_pid=hold['kernel_pid'],
        kernel_start=hold['kernel_start'], launcher_pid=hold['launcher_pid'],
        full_measurement=False, observed_rtl_failure=bool(errors))
    assert not (root/'interruption.json').exists()
    (root/'interruption.json').write_text(json.dumps(interruption, indent=2), encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('milestones','input_hashes')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
