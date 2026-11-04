"""Four-thread full-board placement, retaining only revalidated complete anchors."""
import json,runpy,sys
from pathlib import Path
from run_units import ROOT,sha
from control_command_repair import once
from camera_factor_island_flow import driver_code as parent

def main():
    assert sys.argv.pop(1)=='place'
    i=sys.argv.index('--variant');variant=sys.argv[i+1];del sys.argv[i:i+2]
    assert variant in ('minimal','data64')
    source,code=parent()
    code=once(code,'from camera_factor_balanced_board import BOARD,audit_delta',f'from cnn_{variant}_board import BOARD,audit_delta')
    code=code.replace('camera_factor_island_locations','cnn_run_budget_locations')
    code=once(code,'value="8"','value="4"');code=once(code,'official_max_threads=8','official_max_threads=4')
    compile(code,__file__+':'+variant,'exec')
    if '--check-only' in sys.argv:print('SMALL_PLACE_DRIVER_CHECKED',variant);return
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    driver=root.parent/(root.name+'-place-wrapper.py');assert not driver.exists();driver.write_text(code,encoding='utf-8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),wrapper_sha256=sha(__file__),driver_sha256=sha(driver),variant=variant,hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
