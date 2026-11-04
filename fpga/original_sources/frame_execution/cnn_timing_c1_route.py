"""C1-reviewed placement with C2 routing and unchanged complete release gates."""
import json,runpy,sys
from pathlib import Path
from run_units import sha
from control_command_repair import once
from cnn_timing_closure_route import prepare_code as prior_prepare,route_code as prior_route
def adjust(code):
 return once(code,"{'before':'CONGESTION_2','after':'CONGESTION_2'}","{'before':'CONGESTION_1','after':'CONGESTION_2'}")
def prepare_code():
 s,c=prior_prepare();return s,adjust(c)
def route_code():
 s,c=prior_route();return s,adjust(c)
def main():
 action=sys.argv.pop(1)
 if action=='--check-only':
  for n,f in [('prepare',prepare_code),('route',route_code)]:compile(f()[1],n,'exec')
  print('C1_LAYOUT_C2_ROUTE_COMPLETE_GATES_COMPILE');return
 if action=='prepare':
  source,code=prepare_code();r=Path(sys.argv[sys.argv.index('--review')+1]);root=Path(json.loads(r.read_text())['physical']);suffix='-prepare-cnn-budget'
 else:
  assert action=='route';source,code=route_code();root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();suffix='-local-driver-route'
 driver=root.parent/(root.name+suffix+'.py');assert not driver.exists();compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),driver_sha256=sha(driver),adapter_sha256=sha(__file__),wrapper_sha256=sha(__file__),hardware_access=False),indent=2))
 sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
