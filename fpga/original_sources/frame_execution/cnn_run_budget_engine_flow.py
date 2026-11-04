"""C2 route of the reviewed new engine-retaining placement, with full gates."""
import json,runpy,sys
from pathlib import Path
from run_units import sha
from control_command_repair import once
from cnn_run_budget_c1_flow import prepare_code as prior_prepare,route_code as prior_route

def prepare_code():
 source,code=prior_prepare()
 code=once(code,"{'before':'CONGESTION_2','after':'CONGESTION_1'}","{'before':'CONGESTION_2','after':'CONGESTION_2'}")
 return source,code

def route_code():
 source,code=prior_route();assert code.count('CONGESTION_1')==2
 return source,code.replace('CONGESTION_1','CONGESTION_2')

def main():
 action=sys.argv.pop(1)
 if action=='--check-only':
  for name,fn in [('prepare',prepare_code),('route',route_code)]:compile(fn()[1],name,'exec')
  print('CNN_ENGINE_PREPARE_ROUTE_COMPILE');return
 if action=='prepare':
  source,code=prepare_code();review=Path(sys.argv[sys.argv.index('--review')+1]).resolve();r=json.loads(review.read_text())
  from cnn_run_budget_engine_locations import verify_placed
  assert verify_placed(Path(r['placement']))['count']==9084
  root=Path(r['physical']);suffix='-prepare-cnn-budget'
 elif action=='route':
  source,code=route_code();root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();suffix='-local-driver-route'
 else:raise ValueError(action)
 driver=root.parent/(root.name+suffix+'.py');assert not driver.exists()
 compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
  generated_sha256=sha(driver),driver_sha256=sha(driver),adapter_sha256=sha(__file__),wrapper_sha256=sha(__file__),hardware_access=False),indent=2))
 sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
