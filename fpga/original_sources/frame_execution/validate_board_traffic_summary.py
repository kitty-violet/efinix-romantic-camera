"""Directed integrity tests for native DDR traffic accounting, no HDL claims."""
import argparse
import json
from pathlib import Path
from summarize_board_traffic import analyze
from run_units import sha


def point(frame, ns, scale, **overrides):
    values = dict(event='commit', frame=frame, t_ns=ns, AW=3*scale, AR=4*scale,
        W=20*scale, R=30*scale, B=3*scale, written_bytes=200*scale,
        A_wait=12*scale, W_wait=8*scale, R_wait=5*scale,
        upstream_W_wait=17*scale, W_stall_max=4, raw_fifo_peak=19)
    values.update(overrides)
    return '# BOARD_PROFILE ' + ' '.join(f'{key}={value}' for key, value in values.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    first = point(0, 1000000, 1)
    second = point(2, 3000000, 2)
    result = analyze([first, point(-1, 1500000, 1, event='periodic'),
                      point(0, 2000000, 1, event='screen_complete'), second])
    interval = result['compute_commit_intervals'][0]
    assert interval['interval_ms'] == 2
    assert interval['bus_payload_MB_per_s'] == 0.4
    assert interval['useful_read_plus_strobed_write_MB_per_s'] == 0.34
    assert interval['read_bus_bytes'] == 480 and interval['write_bus_bytes'] == 320
    assert result['lifetime_highwater']['W_stall_max'] == 4
    negatives = [
        [first, point(2, 3000000, 2, W=19)],
        [first, point(2, 3000000, 2, B=7)],
        [first, point(2, 3000000, 2, AW=8)],
        [first, point(2, 3000000, 2, written_bytes=641)],
        [first, point(2, 3000000, 2, raw_fifo_peak=257)],
        [first, point(0, 3000000, 2)],
        [first, point(2, 1000000, 2)],
        [first, point(2, 999999, 2)],
        [first, point(2, 3000000, 2, W_stall_max=17)],
        [first, point(0, 2000000, 1, event='periodic')],
    ]
    for lines in negatives:
        try:
            analyze(lines)
        except AssertionError:
            continue
        raise AssertionError('Corrupt profile accepted: ' + str(lines))
    from summarize_board_traffic import __file__ as analyzer
    summary = dict(status='TRAFFIC_SUMMARY_DIRECTED_PASS', positive_cases=1,
        rejected_negative_cases=len(negatives), analyzer_sha256=sha(Path(analyzer)),
        test_sha256=sha(Path(__file__)), simulated_board_frames=0)
    (args.root / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
