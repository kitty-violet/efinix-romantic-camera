"""Reconcile native DDR counters from the same CSI-to-TMDS measurement.

Counters describe overlapping board traffic, not an exclusive per-frame byte
bill. Interval differences are meaningful; lifetime high-water marks are not
subtracted. R_wait counts response backpressure, not controller read latency.
"""
import argparse
import json
import re
from pathlib import Path
from run_units import sha

COUNTERS = ('AW', 'AR', 'W', 'R', 'B', 'written_bytes', 'A_wait', 'W_wait',
            'R_wait', 'upstream_W_wait')
PEAKS = ('W_stall_max', 'raw_fifo_peak')


def analyze(lines):
    points = []
    for line in lines:
        match = re.search(r'\bBOARD_PROFILE\s+(.*)', line)
        if not match:
            continue
        fields = dict(re.findall(r'(\w+)=([^\s,]+)', match[1]))
        point = {key: int(fields[key]) for key in COUNTERS + PEAKS}
        point.update(event=fields['event'], frame=int(fields['frame']),
                     time_ns=float(fields['t_ns']))
        assert point['event'] in ('commit', 'screen_complete', 'periodic')
        if point['event'] == 'periodic':
            assert point['frame'] == -1, 'Periodic sample must not claim a frame'
        assert all(point[key] >= 0 for key in COUNTERS + PEAKS)
        assert point['B'] <= point['AW'], 'Response without accepted write address'
        assert point['AW'] - point['B'] <= 1, 'More than one native write outstanding'
        assert point['written_bytes'] <= 16 * point['W'], 'Invalid write byte count'
        assert point['raw_fifo_peak'] <= 256, 'RAW FIFO overflow'
        assert point['W_stall_max'] <= point['W_wait']
        if points:
            previous = points[-1]
            assert point['time_ns'] >= previous['time_ns'], 'Reversed profile time'
            assert all(point[key] >= previous[key] for key in COUNTERS + PEAKS), 'Counter decreased'
        points.append(point)
    intervals = []
    # A compute-commit interval includes all concurrent camera, CPU, display,
    # instruction and feature-map transactions on the actual DDR interface.
    commits = [point for point in points if point['event'] == 'commit']
    assert len({point['frame'] for point in commits}) == len(commits), 'Duplicate commit'
    for previous, current in zip(commits, commits[1:]):
        ns = current['time_ns'] - previous['time_ns']
        assert ns > 0, 'Zero commit interval'
        delta = {key: current[key] - previous[key] for key in COUNTERS}
        read_bus_bytes = 16 * delta['R']
        write_bus_bytes = 16 * delta['W']
        intervals.append(dict(previous_frame=previous['frame'], frame=current['frame'],
            interval_ms=ns / 1e6, counters=delta,
            read_bus_bytes=read_bus_bytes, write_bus_bytes=write_bus_bytes,
            strobed_write_bytes=delta['written_bytes'],
            bus_payload_MB_per_s=(read_bus_bytes + write_bus_bytes) * 1000 / ns,
            useful_read_plus_strobed_write_MB_per_s=(read_bus_bytes + delta['written_bytes']) * 1000 / ns))
    return dict(points=points, compute_commit_intervals=intervals,
        lifetime_highwater={key: points[-1][key] for key in PEAKS} if points else {},
        note='Native AXI accepted traffic at 128 bits. Read bytes include narrow CPU reads. '
             'Wait counters overlap and must not be added to elapsed time. R_wait means '
             'RVALID && !RREADY, not waiting for RVALID. Peaks are lifetime values. '
             'No inference about physical DDR efficiency or sustained board bandwidth.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    log = root / 'simulation.log'
    run_path = root / 'run_result.json'
    run = json.loads(run_path.read_text())
    assert sha(log) == run['log_sha256'], 'Simulation log changed'
    result = analyze(log.read_text(encoding='utf8').splitlines())
    result.update(status='BOARD_TRAFFIC_COUNTERS_RECONCILED',
        complete_run=run['status'] == 'BOARD_DIGITAL_MEASUREMENT_COMPLETE',
        inputs_sha256=run['inputs_sha256'], log_sha256=sha(log), run_sha256=sha(run_path))
    (root / 'traffic_profile.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
