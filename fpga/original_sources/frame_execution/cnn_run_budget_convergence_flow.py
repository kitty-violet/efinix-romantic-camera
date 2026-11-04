"""Complete a bounded C2 convergence test on the original reviewed placement.

Prepared only after the three shorter routes have ended. Same current map,
placement and original SDC; no targeted tightening, seed sweep or old route.
Tool2700s, host2745s, at most100 routing iterations, unchanged release gates.
"""
import json,runpy,sys
from pathlib import Path
from run_units import ROOT,sha
from control_command_repair import once
from cnn_run_budget_engine_flow import prepare_code as old_prepare
from cnn_run_budget_c1_flow import route_code as old_route

def prepare_code():
 source,code=old_prepare();code=once(code,"review['budget_seconds']==1350","review['budget_seconds']==2700")
 return source,code

def route_code():
 source,code=old_route();assert code.count('CONGESTION_1')==2
 code=code.replace('CONGESTION_1','CONGESTION_2')
 code=once(code,'60<=a.budget<=1500','60<=a.budget<=2700')
 code=once(code,"review=json.loads(review_path.read_text());assert review['decision']=='PROCEED_TO_BOUNDED_ROUTE'",
  "review=json.loads(review_path.read_text());assert review['decision']=='PROCEED_TO_BOUNDED_ROUTE'\n    assert a.budget==review['budget_seconds']==2700")
 return source,code

def main():
 action=sys.argv.pop(1)
 if action=='--check-only':
  for n,fn in [('prepare',prepare_code),('route',route_code)]:compile(fn()[1],n,'exec')
  print('BOUNDED_C2_2700_ORIGINAL_CONSTRAINT_FLOW_COMPILES');return
 if action=='prepare':
  for name in ('cnn-run-budget-route1','cnn-run-budget-route2','cnn-run-budget-path-route1'):
   batch=ROOT/'tmp/frame-execution'/name
   assert (batch/'failure.json').exists() and (batch/'route-timeout.json').exists(),name
   assert all(s['status']!='running' for s in json.loads((batch/'stages.json').read_text()))
  source,code=prepare_code();review=Path(sys.argv[sys.argv.index('--review')+1]).resolve();root=Path(json.loads(review.read_text())['physical']);suffix='-prepare-cnn-budget'
 else:
  assert action=='route';source,code=route_code();root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();suffix='-local-driver-route'
 driver=root.parent/(root.name+suffix+'.py');assert not driver.exists()
 compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),driver_sha256=sha(driver),adapter_sha256=sha(__file__),wrapper_sha256=sha(__file__),hardware_access=False),indent=2))
 sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
