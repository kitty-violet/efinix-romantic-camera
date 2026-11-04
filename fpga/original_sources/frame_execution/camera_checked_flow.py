"""Full-board offline gates with the explicit camera-only change proof."""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,sha
from camera_checked_board import BASE,audit_delta
from control_command_repair import once

def generated(source,code,suffix):
 root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();driver=root.parent/(root.name+suffix+'.py')
 assert not driver.exists();compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),wrapper_sha256=sha(__file__),driver_sha256=sha(driver),hardware_access=False),indent=2))
 sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')

def main():
 action=sys.argv.pop(1)
 if action=='flow':
  import startup_cause_flow as flow
  flow.BASE=BASE;flow.audit_delta=audit_delta;flow.main()
 elif action=='audit':
  argv=list(sys.argv);board=Path(argv[argv.index('--board')+1]).resolve();root=Path(argv[argv.index('--root')+1]).resolve()
  import audit_fault_probe_map as prior
  prior.audit_delta=audit_delta
  try:prior.main()
  finally:sys.argv=argv
  from camera_checked_cdc import map_structure
  (root/'held_data_structure.json').write_text(json.dumps(map_structure(board),indent=2))
 elif action=='shorts':
  source=HERE/'run_split_logic_outputs_evidence.py';code=source.read_text()
  code=once(code,"'audit_split_logic_outputs_map.py',['--board'","'camera_checked_flow.py',['audit','--board'")
  code=code.replace("'split_logic_outputs_flow.py',['", "'camera_checked_flow.py',['flow','")
  code=once(code,"status='SPLIT_OUTPUTS_SAME_MAP_SHORT_INTEGRATION_PASS'","status='LOCAL_DRIVER_SAME_MAP_SHORT_INTEGRATION_PASS'")
  generated(source,code,'-camera-short-wrapper')
 elif action=='place':
  source=HERE/'place_registered_boundary_board.py';code=once(source.read_text(),'from registered_boundary_board import audit_delta','from camera_checked_board import audit_delta')
  generated(source,code,'-local-driver-wrapper')
 elif action=='prepare-route':
  import prepare_local_driver_route as previous
  previous.audit_delta=audit_delta;previous.main()
 elif action=='route':
  from route_local_driver_board import driver_code
  source,code=driver_code();code=once(code,'from local_driver_board import audit_delta','from camera_checked_board import audit_delta')
  code=once(code,"HERE/'audit_native_split_gray.py'","HERE/'camera_checked_gray.py'")
  generated(source,code,'-local-driver-route')
 else:raise ValueError(action)
if __name__=='__main__':main()
