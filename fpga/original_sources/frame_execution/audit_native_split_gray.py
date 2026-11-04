"""Original routed Gray/CDC gates with native-equivalent cell names."""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,sha


def main():
    source=HERE/'audit_startup_cause_gray.py';code=source.read_text()
    code=code.replace('from startup_cause_clock_cdc import inventory','from native_bound_cdc import inventory')
    code=code.replace("here/'startup_cause_clock_cdc.py'","here/'native_bound_cdc.py'")
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    driver=root.parent/(root.name+'-native-wrapper.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        wrapper_sha256=sha(__file__),driver_sha256=sha(driver),hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')


if __name__=='__main__':main()
