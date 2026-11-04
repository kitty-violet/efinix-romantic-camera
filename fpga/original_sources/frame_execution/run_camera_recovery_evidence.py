"""Existing full-map control tests; public-only serial model uses ModelSim."""
import runpy,sys,json
from pathlib import Path
from run_units import HERE,sha
from control_command_repair import once
source=HERE/'run_split_logic_outputs_evidence.py';code=source.read_text()
code=once(code,"'audit_split_logic_outputs_map.py',['--board'","'camera_recovery_flow.py',['audit','--board'")
code=code.replace("'split_logic_outputs_flow.py',['", "'camera_recovery_flow.py',['flow','")
code=once(code,"'--tck-fast'],300)","'--tck-fast','--modelsim'],300)")
code=once(code,"status='SPLIT_OUTPUTS_SAME_MAP_SHORT_INTEGRATION_PASS'","status='LOCAL_DRIVER_SAME_MAP_SHORT_INTEGRATION_PASS'")
root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();driver=root.parent/(root.name+'-control-shorts.py');assert not driver.exists()
compile(code,str(driver),'exec');driver.write_text(code)
driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),driver_sha256=sha(driver),wrapper_sha256=sha(__file__),hardware_access=False),indent=2))
sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
