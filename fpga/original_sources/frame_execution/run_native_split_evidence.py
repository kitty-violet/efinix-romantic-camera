"""Run the existing same-map gate sequence with native lineage binding."""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,sha


def main():
    source=HERE/'run_split_logic_outputs_evidence.py';code=source.read_text()
    code=code.replace("'audit_split_logic_outputs_map.py'","'audit_native_split_map.py'")
    code=code.replace("'split_logic_outputs_flow.py'","'native_split_flow.py'")
    code=code.replace("status='SPLIT_OUTPUTS_SAME_MAP_SHORT_INTEGRATION_PASS'","status='NATIVE_SPLIT_SAME_MAP_SHORT_INTEGRATION_PASS'")
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    driver=root.parent/(root.name+'-driver.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        wrapper_sha256=sha(__file__),driver_sha256=sha(driver),mapped_checks_unchanged=True),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')


if __name__=='__main__':main()
