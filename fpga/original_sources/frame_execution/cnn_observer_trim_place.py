"""One independent placement-only candidate after an observer-only reduction."""
import json,runpy,sys
from pathlib import Path
from audit_legacy_cache_recovery import install
install()
from run_units import ROOT,sha
from control_command_repair import once
from cnn_minimal_repack import driver_code as original
from cnn_observer_trim_board import BOARD,audit_delta

def main():
    source,code=original('free-arena')
    code=once(code,'from cnn_minimal_board import BOARD,audit_delta','from cnn_observer_trim_board import BOARD,audit_delta')
    if sys.argv[1:]==['--check-only']:
        compile(code,__file__,'exec');print('OBSERVER_TRIM_PLACE_DRIVER_CHECKED');return
    assert sys.argv[1:]==[]
    audit_delta(BOARD)
    root=ROOT/'tmp/frame-execution/cnn-observer-trim-place1';assert not root.exists()
    driver=root.parent/(root.name+'-place-wrapper.py');assert not driver.exists()
    driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        driver_sha256=sha(driver),wrapper_sha256=sha(__file__),
        transform_sha256=sha(ROOT/'tmp/frame-execution/cnn-observer-trim-native1/result.json'),
        ordinary_anchors=0,hard_anchors=401,seed=3,host_threads=4,route_started=False,hardware_access=False),indent=2))
    sys.argv=[str(driver),'--root',str(root),'--base',str(BOARD),
        '--cnn-ram-reference',str(ROOT/'tmp/frame-execution/native-local-driver-route1/canonical-seed3'),
        '--pnr-budget','1800','--logic-limit','60800','--fixed-dsp-layout',
        '--clock-profile-proof',str(ROOT/'tmp/frame-execution/axi-valid-payload-physical1/clock_crossing_inventory.json'),
        '--optimization','CONGESTION_2','--seed','3']
    runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
