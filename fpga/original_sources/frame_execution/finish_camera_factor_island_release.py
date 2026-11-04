"""Preserve every balanced-map release gate and audit the 36-cell placement."""
import json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once
import finish_camera_factor_balanced_release as prior
from camera_factor_island_locations import verify_placed
BASE,SHORTS,RTL,SERIAL=prior.BASE,prior.SHORTS,prior.RTL,prior.SERIAL
parent_evidence,warning_audit,camera_evidence=prior.parent_evidence,prior.warning_audit,prior.camera_evidence

code=once(prior.code,"wrappers(physical,[('-ratio-place-wrapper',HERE/'camera_factor_balanced_flow.py')]",
 "wrappers(physical,[('-ratio-place-wrapper',HERE/'camera_factor_island_flow.py')]")
scope=dict(prior.__dict__)
exec(compile(code,__file__+':all-prior-physical-gates','exec'),scope)

def extra_evidence(board,batch,physical,serial,bind,verify):
 scope['extra_evidence'](board,batch,physical,serial,bind,verify)
 actual=verify_placed(physical);proof=bind(physical/'arena_island_placement.json');verify(proof['evidence'])
 assert actual['count']==36 and proof['total_assignments']==437
 assert proof['exact_all_ports_parameters_including_clock_ce_reset'] and proof['complete_tiles_and_chains']
 from fixed_cnn_ram_locations import verify_placed as check_ram
 assert check_ram(physical)['count']==113
 for n in ('arena_island_placement.json','arena_island_generator.py','arena_island_actual_locations.json','cnn_ram_placement.json','cnn_ram_actual_locations.json'):
  verify({str(physical/n):sha(physical/n)})
 verify({str(HERE/'camera_factor_island_locations.py'):sha(HERE/'camera_factor_island_locations.py'),str(HERE/'camera_factor_island_flow.py'):sha(HERE/'camera_factor_island_flow.py')})

def package_code():
 source,code=prior.package_code();code=code.replace('finish_camera_factor_balanced_release','finish_camera_factor_island_release')
 compile(code,'island-release-full-gates','exec');return source,code

def main():
 if sys.argv[1:]==['--check-only']:
  package_code();print('ISLAND_COMPLETE_RELEASE_GENERATORS_COMPILE');return
 mode=sys.argv.pop(1)
 if mode=='pll':
  import audit_pll_shift_routed as check
  check.audit_delta=prior.audit_delta;check.main();return
 if mode=='bundle':source,code=package_code();key='--output'
 else:
  assert mode=='finish'
  from finish_local_driver_release import finish_code
  source,code=finish_code();code=code.replace('finish_local_driver_release.py','finish_camera_factor_island_release.py').replace('a.output.resolve()],240)','a.output.resolve()],360)');key='--root'
 args=sys.argv[1:];out=Path(args[args.index(key)+1]).resolve();prior.audit_delta(prior.BOARD);assert not out.exists()
 driver=out.parent/(out.name+'-camera-release.py');assert not driver.exists();compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),adapter_sha256=sha(__file__)),indent=2))
 sys.argv=[str(driver),*args];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
