"""Prioritize the eight measured parent bottlenecks on current free placement.

Only the routing optimization SDC is stricter. Restore exact original bytes
before STA/Gray, including tool failure or host timeout. No route is imported.
"""
import json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once
from cnn_run_budget_engine_flow import prepare_code
from cnn_run_budget_c1_flow import route_code as prior_route
from camera_compact_path_flow import EXTRA,PROBE,targets,verified_probe as parent_probe
from native_local_drivers import graph
from probe_full_native_split_map import params

def verified_targets():
 proof=parent_probe()
 old=ROOT/'tmp/frame-execution/connected-board-camera-factor-compact1/outflow/ti60f225_oob.map.v'
 new=ROOT/'tmp/frame-execution/connected-board-cnn-run-budget1/outflow/ti60f225_oob.map.v'
 a,b=graph(old.read_text()),graph(new.read_text());ap,bp=params(old.read_text()),params(new.read_text())
 rows=[]
 for p in targets:
  row={}
  for end in ('start','end'):
   n,pin=p[end].rsplit('|',1);row[end]=p[end]
   assert n in a and n in b and a[n][0]==b[n][0] and ap[n]==bp[n],n
   field=pin.split('[')[0];assert field in b[n][1],(n,pin)
   if a[n][0]=='EFX_FF':assert a[n][1]['CLK']==b[n][1]['CLK'],n
  rows.append(row)
 return dict(status='CURRENT_TARGET_ENDPOINTS_AND_CLOCKS_CHECKED_STRICTER_NOT_STA',targets=rows,
  parent_probe_sha256=sha(PROBE/'result.json'),old_map_sha256=sha(old),map_sha256=sha(new),
  optimization_target_ns=6.366,original_process_period_ns=6.666,acceptance_unchanged=True)

def route_code():
 source,code=prior_route();assert code.count('CONGESTION_1')==2
 code=code.replace('CONGESTION_1','CONGESTION_2')
 anchor="        row=run('route',['cmd.exe','/d','/c',cmd],physical,a.budget+45)"
 change="""        from cnn_run_budget_path_flow import EXTRA,PROBE,verified_targets
        target_check=verified_targets()
        sdc=physical/'ti60f225_oob.sdc';canonical_bytes=sdc.read_bytes()
        (root/'route_acceptance.sdc').write_bytes(canonical_bytes)
        optimized=sdc.read_text()+EXTRA
        (root/'route_optimization.sdc').write_text(optimized,encoding='utf8')
        receipt=dict(status='OPTIMIZATION_ONLY_RUNNING',canonical_sha256=sha(sdc),optimization_sha256=sha(root/'route_optimization.sdc'),
            probe=str(PROBE/'result.json'),probe_sha256=sha(PROBE/'result.json'),adapter_sha256=sha(HERE/'cnn_run_budget_path_flow.py'),
            full_original_sta_required=True,hardware_clock_changed=False,placement_changed=False,extra=EXTRA,target_check=target_check)
        save(root/'route_optimization_inputs.json',receipt)
        sdc.write_bytes((root/'route_optimization.sdc').read_bytes())
        try:
            row=run('route',['cmd.exe','/d','/c',cmd],physical,a.budget+45)
        finally:
            sdc.write_bytes(canonical_bytes)
            assert sha(sdc)==receipt['canonical_sha256']
            receipt['status']='ORIGINAL_SDC_RESTORED_BEFORE_ACCEPTANCE'
            save(root/'route_optimization_inputs.json',receipt)
"""
 code=once(code,anchor,change.rstrip());compile(code,'cnn-budget-targeted-route','exec');return source,code

def main():
 action=sys.argv.pop(1)
 if action=='--check-only':
  prepare_code();route_code();print(json.dumps(verified_targets()));return
 if action=='prepare':
  source,code=prepare_code();review=Path(sys.argv[sys.argv.index('--review')+1]).resolve();r=json.loads(review.read_text())
  root=Path(r['physical']);suffix='-prepare-cnn-budget'
 else:
  assert action=='route';source,code=route_code();root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();suffix='-local-driver-route'
 driver=root.parent/(root.name+suffix+'.py');assert not driver.exists()
 compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
  generated_sha256=sha(driver),driver_sha256=sha(driver),adapter_sha256=sha(__file__),wrapper_sha256=sha(__file__),hardware_access=False),indent=2))
 sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
