"""One same-placement C1 route after the recorded bounded C2 timeout."""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,sha
from control_command_repair import once
def prepare_code():
 from camera_early_ratio_route import prepare_code as parent
 source,code=parent()
 code=once(code,'from camera_checked_board import audit_delta','from cnn_run_budget_board import audit_delta')
 code=once(code,"('-ratio-place-wrapper','-place-driver')","('-place-wrapper','-place-driver')")
 code=once(code,"{'before':'CONGESTION_2','after':'CONGESTION_2'}","{'before':'CONGESTION_2','after':'CONGESTION_1'}")
 return source,code
def route_code():
 from route_local_driver_board import driver_code
 source,code=driver_code();assert code.count('TIMING_1')==2
 code=code.replace('TIMING_1','CONGESTION_1')
 code=once(code,'from local_driver_board import audit_delta','from cnn_run_budget_board import audit_delta')
 code=once(code,'import audit_control_state_implementation as impl','import cnn_run_budget_implementation as impl')
 code=once(code,"HERE/'audit_native_split_gray.py'","HERE/'camera_joint_gray.py'")
 return source,code
def main():
 action=sys.argv.pop(1)
 if action=='--check-only':
  for name,fn in [('prepare',prepare_code),('route',route_code)]:compile(fn()[1],name,'exec')
  print('CNN_BUDGET_C1_PREPARE_ROUTE_COMPILE');return
 if action=='prepare':
  source,code=prepare_code();review=Path(sys.argv[sys.argv.index('--review')+1]).resolve()
  root=Path(json.loads(review.read_text())['physical']);suffix='-prepare-cnn-budget'
 elif action=='route':
  source,code=route_code();root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();suffix='-local-driver-route'
 else:raise ValueError(action)
 driver=root.parent/(root.name+suffix+'.py');assert not driver.exists()
 compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
  generated_sha256=sha(driver),driver_sha256=sha(driver),adapter_sha256=sha(__file__),wrapper_sha256=sha(__file__),hardware_access=False),indent=2))
 sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
