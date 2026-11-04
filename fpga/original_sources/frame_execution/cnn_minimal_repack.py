"""Two measured placement controls on the frozen minimal map, never route here.

C1 keeps the 36 complete arena cells. Free-arena keeps C2 and all 401 hard
anchors but releases those 36 ordinary cells. These are independent contrasts
against minimal-place1, not a seed sweep or a new functional design.
"""
import argparse
import json
import runpy
import sys
from pathlib import Path

from run_units import ROOT, HERE, sha
from control_command_repair import once


def driver_code(mode):
    if mode == 'c1':
        from camera_factor_island_flow import driver_code as parent
        source, code = parent()
        code = once(code, 'from camera_factor_balanced_board import BOARD,audit_delta',
                    'from cnn_minimal_board import BOARD,audit_delta')
        code = code.replace('camera_factor_island_locations', 'cnn_run_budget_locations')
    else:
        assert mode == 'free-arena'
        from camera_ratio_before_place import driver_code as parent
        source, code = parent()
        code = once(code, 'from camera_checked_v2 import BOARD\nimport camera_checked_board\ncamera_checked_board.BOARD=BOARD\naudit_delta=camera_checked_board.audit_delta',
                    'from cnn_minimal_board import BOARD,audit_delta')
    code = once(code, 'value="8"', 'value="4"')
    code = once(code, 'official_max_threads=8', 'official_max_threads=4')
    compile(code, __file__ + ':' + mode, 'exec')
    return source, code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('c1', 'free-arena'), required=True)
    parser.add_argument('--check-only', action='store_true')
    args, rest = parser.parse_known_args()
    source, code = driver_code(args.mode)
    if args.check_only:
        print('FROZEN_MINIMAL_PLACEMENT_DRIVER_CHECKED', args.mode)
        return
    board = ROOT / 'tmp/frame-execution/connected-board-cnn-minimal1'
    failed = ROOT / 'tmp/frame-execution/cnn-minimal-route1'
    assert (failed / 'failure.json').exists()
    assert all(s['status'] == 'finished' for s in json.loads((failed / 'stages.json').read_text()))
    assert sha(board / 'outflow/ti60f225_oob.map.v') == '2796ceb91151326977b2dd86f3b1474d582020b0d1030d9bcde487ef71f8a1a2'
    short = ROOT / 'tmp/frame-execution/cnn-minimal-evidence1/result.json'
    assert json.loads(short.read_text())['status'] == 'LOCAL_DRIVER_SAME_MAP_SHORT_INTEGRATION_PASS'
    root = ROOT / ('tmp/frame-execution/cnn-minimal-' + args.mode + '-place1')
    assert not root.exists() and not rest
    driver = root.parent / (root.name + '-place-wrapper.py')
    assert not driver.exists()
    driver.write_text(code, encoding='utf-8')
    driver.with_suffix('.json').write_text(json.dumps(dict(
        source=str(source), source_sha256=sha(source), wrapper_sha256=sha(__file__),
        driver_sha256=sha(driver), mode=args.mode, map_sha256=sha(board / 'outflow/ti60f225_oob.map.v'),
        short_result_sha256=sha(short), original_failure_sha256=sha(failed / 'failure.json'),
        ordinary_anchors=36 if args.mode == 'c1' else 0,
        hard_anchors=401, seed=3, host_threads=4,
        logic_pins_clocks_timing_requirements_unchanged=True,
        route_started=False, hardware_access=False), indent=2), encoding='utf-8')
    sys.argv = [str(driver), '--root', str(root), '--base', str(board),
                '--cnn-ram-reference', str(ROOT / 'tmp/frame-execution/native-local-driver-route1/canonical-seed3'),
                '--pnr-budget', '1800', '--logic-limit', '60800', '--fixed-dsp-layout',
                '--clock-profile-proof', str(ROOT / 'tmp/frame-execution/axi-valid-payload-physical1/clock_crossing_inventory.json'),
                '--optimization', 'CONGESTION_1' if args.mode == 'c1' else 'CONGESTION_2', '--seed', '3']
    runpy.run_path(str(driver), run_name='__main__')


if __name__ == '__main__':
    main()
