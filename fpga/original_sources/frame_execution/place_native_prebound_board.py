"""Same C2/seed3 native map and anchors; Gray names valid before placement."""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,sha
from control_command_repair import once

def main():
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    source=HERE/'place_registered_boundary_board.py';code=source.read_text()
    code=once(code,'from registered_boundary_board import audit_delta','from native_split_board import audit_delta')
    needle="    driver=root.parent/(root.name+'-place-wrapper.py')"
    insertion="    code=once(code,\"'from control_ddr_clock_profile import apply_profile'\",\"'from native_prebound_constraints import apply_profile'\")\n"
    code=once(code,needle,insertion+needle)
    driver=root.parent/(root.name+'-native-wrapper.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        wrapper_sha256=sha(__file__),driver_sha256=sha(driver),hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
    from native_prebound_constraints import audit
    audit(root)

if __name__=='__main__':main()
