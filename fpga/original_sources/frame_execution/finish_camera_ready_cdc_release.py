"""Same complete release gates plus the exact retained-done CDC cone proof."""
from pathlib import Path
_source=Path(__file__).with_name('finish_camera_ready_release.py')
_code=_source.read_text(encoding='utf8').replace('finish_camera_ready_release','finish_camera_ready_cdc_release')
_code,_guard,_tail=_code.rpartition("if __name__=='__main__':main()")
assert _guard and not _tail.strip()
exec(compile(_code,str(_source)+':cdc-repaired','exec'),globals())

_prior_extra=extra_evidence
def extra_evidence(board,batch,physical,serial,bind,verify):
 _prior_extra(board,batch,physical,serial,bind,verify)
 gray=batch/'gray-seed3';g=bind(gray/'result.json');r=bind(gray/'reaudit_result.json');i=bind(gray/'reaudit_inputs.json')
 assert r['status']=='COMPLETED_REPORTS_CDC_REAUDIT_PASS' and r['inputs_sha256']==sha(gray/'reaudit_inputs.json')
 assert g['reaudit_sha256']==sha(gray/'reaudit_result.json') and r['inventory_sha256']==sha(gray/'clock_crossing_inventory.json')
 verify(i['evidence']);assert not i['hardware_or_constraint_change'] and not i['sta_rerun']
 for n in ('camera_checked_cdc_v3.py','resume_camera_ready_gray.py','finish_camera_ready_release.py','finish_camera_ready_cdc_release.py'):verify({str(HERE/n):sha(HERE/n)})

_prior_package=package_code
def package_code():
 source,code=_prior_package()
 needle='from camera_checked_cdc_v2 import inventory';assert code.count(needle)==1
 code=code.replace(needle,'from camera_checked_cdc_v3 import inventory')
 compile(code,'ready-exact-cdc-package','exec');return source,code

def finish_code():
 import finish_local_driver_release as old
 source,code=old.finish_code()
 code=code.replace('finish_local_driver_release.py','finish_camera_ready_cdc_release.py')
 # The measured full native audit alone takes about 67 seconds. Keep bounded
 # host budgets appropriate for actual repeated evidence checks, not old small maps.
 old_bound="'--root',root/'pll-paths'],45)";assert code.count(old_bound)==1
 code=code.replace(old_bound,"'--root',root/'pll-paths'],150)")
 old_bundle="'--output',a.output.resolve()],240)";assert code.count(old_bundle)==1
 code=code.replace(old_bundle,"'--output',a.output.resolve()],600)")
 compile(code,'ready-exact-cdc-finish','exec');return source,code

def main():
 if sys.argv[1:]==['--check-only']:
  package_code();finish_code();print('READY_CDC_COMPLETE_RELEASE_GENERATORS_COMPILE');return
 mode=sys.argv.pop(1)
 if mode=='pll':
  import audit_pll_shift_routed as check
  check.audit_delta=audit_delta;check.main();return
 if mode=='bundle':source,code=package_code();key='--output'
 else:
  assert mode=='finish';source,code=finish_code();key='--root'
 args=sys.argv[1:];out=Path(args[args.index(key)+1]).resolve();audit_delta(BOARD)
 driver=out.parent/(out.name+'-camera-release.py');assert not driver.exists()
 driver.write_text(code,encoding='utf8')
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),adapter_sha256=sha(__file__)),indent=2))
 sys.argv=[str(driver),*args];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
