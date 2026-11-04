"""One timing-directed routing comparison on the already reviewed placement."""
import json,runpy,sys
from pathlib import Path
from run_units import ROOT,sha
from control_command_repair import once
import cnn_observer_trim_free_route as parent

def prepare_code():
 source,code=parent.prepare_code()
 code=once(code,"{'before':'CONGESTION_2','after':'CONGESTION_2'}","{'before':'CONGESTION_2','after':'TIMING_1'}")
 return source,code

def route_code():
 source,code=parent.route_code()
 code=once(code,"{'before':'CONGESTION_2','after':'CONGESTION_2'}","{'before':'CONGESTION_2','after':'TIMING_1'}")
 code=once(code,'optimization_level=CONGESTION_2','optimization_level=TIMING_1')
 code=once(code,"HERE/'canonical_snapshot_sta.py'","HERE/'canonical_snapshot_sta_isolated.py'")
 code=once(code,"'--root',canonical],ROOT,300)","'--root',canonical],ROOT,450)")
 return source,code

def main():
 action=sys.argv.pop(1)
 if action=='--check-only':
  for fn in (prepare_code,route_code):compile(fn()[1],__file__,'exec')
  print('OBSERVER_TIMING_ROUTE_SAME_MAP_PLACEMENT_SDC_GATES_COMPILE');return
 if action=='prepare':
  source,code=prepare_code();review=Path(sys.argv[sys.argv.index('--review')+1]).resolve();r=json.loads(review.read_text())
  assert r['route_profile_cli_override']==dict(before='CONGESTION_2',after='TIMING_1')
  root=Path(r['physical']);suffix='-prepare-cnn-budget'
 else:
  assert action=='route';source,code=route_code();root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();suffix='-local-driver-route'
 assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 driver=root.parent/(root.name+suffix+'.py');assert not driver.exists();driver.write_text(code,encoding='utf8')
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),driver_sha256=sha(driver),adapter_sha256=sha(__file__),wrapper_sha256=sha(__file__),hardware_access=False),indent=2))
 sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
