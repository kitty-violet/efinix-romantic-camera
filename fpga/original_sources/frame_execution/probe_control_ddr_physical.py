"""Keep proven full-board physical flow; bind independent debug-only delta."""
import runpy,sys
from pathlib import Path
from run_units import HERE,ROOT,sha
from control_ddr_combined import once

if __name__=='__main__':
    args=sys.argv[1:];root=Path(args[args.index('--root')+1]).resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    source=HERE/'probe_shift_guard_physical.py'
    code=once(source.read_text(),"'from shift_guard_clock_profile import apply_profile'","'from control_ddr_clock_profile import apply_profile'")
    driver=root.parent/(root.name+'-combined-probe.py');assert not driver.exists()
    driver.write_text(code);sys.argv=[str(driver),*args];runpy.run_path(str(driver),run_name='__main__')
