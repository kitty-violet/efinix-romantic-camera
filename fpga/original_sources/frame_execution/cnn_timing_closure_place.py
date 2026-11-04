"""Global placement with only hard DSP/RAM anchors, explicit post-place review."""
import json,runpy,sys
from pathlib import Path
from run_units import ROOT,sha
from control_command_repair import once
from camera_ratio_before_place import driver_code as previous

def driver_code():
    source,code=previous()
    code=once(code,'from camera_checked_v2 import BOARD\nimport camera_checked_board\ncamera_checked_board.BOARD=BOARD\naudit_delta=camera_checked_board.audit_delta',
        'from cnn_timing_closure_board import BOARD,audit_delta')
    compile(code,'closure-place','exec');return source,code

def main():
    source,code=driver_code()
    if sys.argv[1:]==['--check-only']:print('CLOSURE_HARD_ANCHORS_ONLY_PLACE_COMPILES');return
    assert sys.argv.pop(1)=='place'
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    driver=root.parent/(root.name+'-place-wrapper.py');assert not driver.exists()
    driver.write_text(code,encoding='utf8');driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),wrapper_sha256=sha(__file__),driver_sha256=sha(driver),hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
