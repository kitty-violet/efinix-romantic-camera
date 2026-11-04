"""One placement-only C3 control after the C1 terminal routing failure.

Keep the exact mapped design, fixed RAM/DSP anchors and original constraints.
No automatic route, no new seed, and no board access.
"""
import json
import runpy
import sys
from pathlib import Path
from audit_legacy_cache_recovery import install
install()
from run_units import ROOT, EF, sha
from cnn_minimal_repack import driver_code


def main():
    source, code = driver_code('free-arena')
    compile(code, __file__ + ':placement-only', 'exec')
    if sys.argv[1:] == ['--check-only']:
        print('C3_FREE_ORDINARY_PLACEMENT_ONLY_COMPILE_PASS')
        return
    assert not sys.argv[1:]
    base = ROOT / 'tmp/frame-execution'
    board = base / 'connected-board-cnn-minimal1'
    failure = base / 'cnn-minimal-c1-route1'
    stages = json.loads((failure / 'stages.json').read_text())
    assert all(row['status'] == 'finished' for row in stages)
    assert json.loads((failure / 'failure.json').read_text())['board_release'] is False
    log = base / 'cnn-minimal-c1-bound1/outflow/ti60f225_oob.route.out'
    assert 'not been converging for the last 3 iterations' in log.read_text()
    assert sha(board / 'outflow/ti60f225_oob.map.v') == '2796ceb91151326977b2dd86f3b1474d582020b0d1030d9bcde487ef71f8a1a2'
    short = base / 'cnn-minimal-evidence1/result.json'
    assert json.loads(short.read_text())['status'] == 'LOCAL_DRIVER_SAME_MAP_SHORT_INTEGRATION_PASS'
    root = base / 'cnn-minimal-free-c3-place1'
    driver = base / (root.name + '-place-wrapper.py')
    assert not root.exists() and not driver.exists()
    driver.write_text(code, encoding='utf8')
    files = (Path(__file__), Path(__file__).with_name('cnn_minimal_repack.py'),
             failure / 'failure.json', failure / 'stages.json', log,
             EF / 'doc/topics/tim-pnr.html', short)
    driver.with_suffix('.json').write_text(json.dumps(dict(
        source=str(source), source_sha256=sha(source), wrapper_sha256=sha(__file__),
        driver_sha256=sha(driver), mode='free-ordinary-C3-placement-only',
        map_sha256=sha(board / 'outflow/ti60f225_oob.map.v'),
        evidence={str(p): sha(p) for p in files}, ordinary_anchors=0,
        hard_anchors=401, seed=3, host_threads=4, optimization='CONGESTION_3',
        rationale='C1 placement/C2 route failed by actual nonconvergence; C2 free placement is still routing. Use the idle second four-thread slot for one documented C3 placement-only contrast, preserving the active candidate. Review full congestion/connectivity before any route.',
        logic_pins_clocks_timing_requirements_unchanged=True,
        route_started=False, hardware_access=False), indent=2), encoding='utf8')
    sys.argv = [str(driver), '--root', str(root), '--base', str(board),
                '--cnn-ram-reference', str(base / 'native-local-driver-route1/canonical-seed3'),
                '--pnr-budget', '1800', '--logic-limit', '60800', '--fixed-dsp-layout',
                '--clock-profile-proof', str(base / 'axi-valid-payload-physical1/clock_crossing_inventory.json'),
                '--optimization', 'CONGESTION_3', '--seed', '3']
    runpy.run_path(str(driver), run_name='__main__')


if __name__ == '__main__':
    main()
