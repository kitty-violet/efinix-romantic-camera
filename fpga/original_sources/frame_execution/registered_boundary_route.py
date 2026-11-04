"""Original route/STA/CDC gates, using the exact registered-boundary audit."""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,sha
from control_command_repair import once

def main():
    source=HERE/'route_local_decode_board.py';code=source.read_text()
    changes={
        'from local_control_decode import audit_delta':'from registered_boundary_board import audit_delta',
        'from audit_local_control_decode_map import audit as audit_map':
        'import audit_local_control_decode_map as map_auditor\nmap_auditor.audit_delta=audit_delta\naudit_map=map_auditor.audit',
        "HERE/'audit_command_progress_gray.py'":"HERE/'audit_split_logic_outputs_gray.py'",
        'from inline_official_state_luts import inline':'from inline_fault_probe_luts import inline',
        "HERE/'inline_official_state_luts.py'":"HERE/'inline_fault_probe_luts.py'",
        "            except subprocess.TimeoutExpired:save(root/(name+'-timeout.json'),stop_launched_tree(proc));raise":
        "            except subprocess.TimeoutExpired:\n                save(root/(name+'-timeout.json'),stop_launched_tree(proc))\n                row.update(status='timed_out_process_tree_stopped',seconds=time.monotonic()-start)\n                save(root/'stages.json',stages);raise"}
    for before,after in changes.items():code=once(code,before,after)
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    driver=root.parent/(root.name+'-driver.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        driver_sha256=sha(driver),wrapper_sha256=sha(__file__),hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
