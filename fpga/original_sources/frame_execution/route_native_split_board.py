"""Original strict route sequence; only native lineage/CDC adapter changes."""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,sha
from control_command_repair import once


def main():
    source=HERE/'startup_cause_route.py';code=source.read_text()
    code=once(code,'from startup_cause_board import audit_delta','from native_split_board import audit_delta')
    code=once(code,"HERE/'audit_startup_cause_gray.py'","HERE/'audit_native_split_gray.py'")
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    driver=root.parent/(root.name+'-native-wrapper.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        wrapper_sha256=sha(__file__),driver_sha256=sha(driver),hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')


if __name__=='__main__':main()
