"""Camera-only delta plus unchanged complete-video release gates. Offline only."""
import inspect,json,re,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once
from camera_checked_board import BOARD,PARENT as BASE,audit_delta
from finish_split_logic_outputs_release import archived_placement_evidence
from finish_local_driver_release import wrappers

PARENT=ROOT/'output/deployment/board-full-150-local-drivers-20261012'
SHORTS=ROOT/'tmp/frame-execution/camera-checked-evidence1'
RTL=SHORTS/'rtl';SERIAL=SHORTS.parent/(SHORTS.name+'-serial/new')
MAP_AUDIT=SHORTS/'map-audit';ADAPTER=SHORTS.parent/(SHORTS.name+'-adapter')

def parent_evidence(parent,verify):
 assert Path(parent['board'])==BASE
 verify(parent['required_evidence']);audit_delta(BOARD)
 return dict(status='FULL_VIDEO_A_PARENT_CAMERA_ONLY_CHANGE_BOUND',manifest=str(PARENT/'manifest.json'),sha256=sha(PARENT/'manifest.json'),
  scope='CPU, DDR, CNN and display public cells/connections remain equal. Only dead camera/probe cone removed and 113 camera pad/snapshot inputs replaced. No board PASS inherited.')

def warning_audit(board):
 from finish_local_driver_release import warning_audit as prior_audit
 prior=prior_audit(BASE);audit_delta(board)
 original=(BASE/'outflow/ti60f225_oob.warn.log').read_text()
 f=board/'outflow/ti60f225_oob.warn.log';body=f.read_text()
 assert '[EFX-0654' not in body
 names=lambda s:re.findall(r"\[EFX-0201 WARNING\] Re-wiring to GND non-driven net/port '([^']+)'",s)
 assert names(body)==names(original)==['Open_'+str(n) for n in range(8,18)]
 for marker in ('[EFX-0200 WARNING]','[EFX-0201 WARNING]'):assert body.count(marker)==original.count(marker)
 mapped=json.loads((board/'result.json').read_text())
 assert mapped['protected_boundary']==dict(ports=1201,bits=4128,disconnected_bits=1374,other_connections_equal=True)
 return dict(status='UNCHANGED_OPEN_INPUT_PROOF_AND_CAMERA_EXTERNAL_INPUT_CHECK',evidence={**prior['evidence'],str(f):sha(f),
  str(board/'result.json'):sha(board/'result.json'),str(board/'public_connectivity_audit.json'):sha(board/'public_connectivity_audit.json')},
  private_vendor_warnings_retained=True,new_functional_unconnected_inputs=False)

def camera_evidence(bind,verify):
 unit=ROOT/'tmp/board-test/camera-checked-unit2';u=bind(unit/'result.json')
 assert u['status']=='CHECKED_CAMERA_PIN_TESTS_PASS' and len(u['cases'])==10
 verify(u['inputs'])
 assert all(c['passed'] for c in u['cases'])
 # Preserve the failed aggregate/license attempt; only its successful mode4
 # is reused. The serial run supplies modes0/6/8 with the same exact map.
 for folder,modes in (('camera-checked-mapped-questa2',[4]),('camera-checked-mapped-questa3',[0,6,8])):
  root=ROOT/'tmp/board-test'/folder;r=bind(root/'result.json');i=bind(root/'inputs.json')
  assert r['inputs_sha256']==sha(root/'inputs.json') and r['map_sha256']==sha(BOARD/'outflow/ti60f225_oob.map.v')
  verify({str(root/n):h for n,h in i['files'].items()})
  for mode in modes:
   c=next(x for x in r['cases'] if x['mode']==mode);assert c['passed'] and not c['timed_out']
   verify({str(root/f'mode{mode}.log'):c['log_sha256']})
 observer=ROOT/'tmp/board-test/camera-observer-host1';o=bind(observer/'result.json')
 assert o['status']=='CAMERA_OBSERVER_RTL_MAPPED_GATES_AND_HOST_PASS' and o['held_bits']==95 and o['host_checks']==117
 verify(o['evidence'])
 return dict(rtl_cases=10,actual_mapped_modes=[0,4,6,8],host_checks=117,held_bits=95)

def extra_evidence(board,batch,physical,serial,bind,verify):
 assert board==BOARD and audit_delta(board)==bind(board/'delta_audit.json')
 mapped=bind(board/'result.json');ma=bind(MAP_AUDIT/'result.json')
 assert ma['status']=='FAULT_PROBE_ACTUAL_CONTROL_BOOLEAN_AND_CDC_MAP_PASS' and ma['checks']==17504
 assert ma['map_sha256']==mapped['map_sha256'] and ma['board_inputs_sha256']==sha(board/'inputs.json');verify(ma['evidence'])
 for folder,status in ((SHORTS/'observe','MAPPED_DDR_STATUS_AND_EVENTS_CONTROL_RESET_PASS'),(ADAPTER,'ACTUAL_MAPPED_CONTROL_ADAPTER_IDLE_THREE_RESETS_PASS')):
  r=bind(folder/'result.json');i=bind(folder/'inputs.json')
  assert r['status']==status and r['inputs_sha256']==sha(folder/'inputs.json')
  assert i.get('source_map_sha256',i.get('original_map_sha256'))==mapped['map_sha256']
  verify({str(folder/n):h for n,h in i['files'].items()});verify({str(folder/'simulation.log'):r.get('log_sha256',r.get('simulation_sha256'))})
 import finish_local_decode_release as old
 code=inspect.getsource(old.extra_evidence)
 code='def inherited_physical(board,batch,physical,serial,bind,verify):\n'+code[code.index("    si=bind(serial/'inputs.json')"):]
 code=code.replace('from inline_official_state_luts import inline','from inline_fault_probe_luts import inline').replace("HERE/'inline_official_state_luts.py'","HERE/'inline_fault_probe_luts.py'")
 code=once(code,"len(details['instances'])==23","len(details['instances'])==proof['primitive_instances_removed']")
 for n in ('route','review'):code=once(code,"verify("+n+"['required_evidence'])","verify(archived_placement_evidence("+n+"['required_evidence'],physical,bind))")
 needle="    verify({str(driver):driver_meta['driver_sha256'],driver_meta['source']:driver_meta['source_sha256'],str(wrapper):driver_meta['wrapper_sha256']})"
 code=once(code,needle,"    wrapper=physical.parent/(physical.name+'-place-wrapper.py')\n"+needle)
 scope=dict(old.__dict__,archived_placement_evidence=archived_placement_evidence);exec(compile(code,__file__+':physical','exec'),scope)
 scope['inherited_physical'](board,batch,physical,serial,bind,verify)
 wrappers(physical,[('-place-wrapper',physical.parent/(physical.name+'-local-driver-wrapper.py')),('-local-driver-wrapper',HERE/'camera_checked_flow.py')],bind,verify)
 wrappers(batch,[('-local-driver-route',HERE/'camera_checked_flow.py')],bind,verify)
 inv=bind(batch/'gray-seed3/clock_crossing_inventory.json')
 assert inv['camera_bundled_settling']['endpoints']==95 and inv['camera_bundled_settling']['max_delay_ns']<13.332
 assert inv['camera_structure']['map_sha256']==mapped['map_sha256'];verify(inv['camera_bundled_settling']['reports'])
 complete=bind(SHORTS/'result.json');assert complete['status']=='LOCAL_DRIVER_SAME_MAP_SHORT_INTEGRATION_PASS'
 assert complete['map_sha256']==mapped['map_sha256'];verify(complete['evidence'])
 rebound=bind(physical/'endpoint_rebinding.json');origin=Path(rebound['source'])
 assert bind(physical/'physical_inputs.json')['endpoint_rebinding_sha256']==sha(physical/'endpoint_rebinding.json')
 verify(rebound['original']);verify({str(HERE/'prepare_local_driver_route.py'):rebound['script_sha256'],str(HERE/'rebind_native_placement.py'):rebound['translator_sha256']})
 from rebind_native_placement import translate
 expected,pins=translate((origin/'ti60f225_oob.sdc').read_text(),(origin/'outflow/ti60f225_oob.map.v').read_text(),(origin/'outflow/ti60f225_oob.out.placeloc').read_text())
 assert (physical/'ti60f225_oob.sdc').read_text()==expected and pins==rebound['pins'];verify({str(physical/n):h for n,h in rebound['geometry'].items()})
 camera_evidence(bind,verify)
 for n in ('camera_checked_board.py','audit_camera_checked_map.py','camera_checked_flow.py','camera_checked_cdc.py','camera_checked_gray.py','finish_camera_checked_release.py','read_camera_checked.py','read_camera_checked_snapshot.py'):
  verify({str(HERE/n):sha(HERE/n)})

def package_code():
 import finish_local_driver_release as old
 source,code=old.package_code()
 code=code.replace('from local_driver_board import audit_delta','from camera_checked_board import audit_delta').replace('finish_local_driver_release','finish_camera_checked_release')
 code=code.replace('from native_bound_cdc import inventory','from camera_checked_cdc import inventory')
 code=code.replace('output/deployment/board-full-150-startup-cause-20261011','output/deployment/board-full-150-local-drivers-20261012')
 code=once(code,"verify({str(HERE/'local_driver_board.py'):mi['local_driver_packager_sha256']})","verify({str(HERE/'camera_checked_board.py'):mi['camera_builder_sha256']})")
 code=once(code,"functional_evidence_scope='Original full RTL retained; native remap and current interfaces separately checked, no inherited board PASS'", "functional_evidence_scope='Full parent video/A retained except checked camera configuration and passive observation; current-map tests and physical gates required',camera_checks=camera_evidence(bind,verify),camera_diagnostic_slots=bind(board/'camera_graft.json')['diagnostic_slots']")
 code=code.replace('extra_evidence,parent_evidence,warning_audit','extra_evidence,parent_evidence,warning_audit,camera_evidence')
 start=code.index("    (out/'README.md').write_text(");end=code.index('    manifest=dict(',start)
 code=code[:start]+"    shutil.copy2(HERE/'CAMERA_CHECKED_RELEASE.md',out/'README.md')\n    (out/'read_camera_checked_snapshot.py').write_text('import sys\\nsys.path.insert(0,'+repr(str(HERE))+')\\nfrom read_camera_checked_snapshot import main\\nif __name__==\\\"__main__\\\":main()\\n')\n"+code[end:]
 return source,code

def main():
 mode=sys.argv.pop(1)
 if mode=='pll':
  import audit_pll_shift_routed as check
  check.audit_delta=audit_delta;check.main();return
 if mode=='bundle':source,code=package_code();key='--output'
 else:
  assert mode=='finish'
  import finish_local_driver_release as old
  source,code=old.finish_code();code=code.replace('finish_local_driver_release.py','finish_camera_checked_release.py');key='--root'
 args=sys.argv[1:];out=Path(args[args.index(key)+1]).resolve();audit_delta(BOARD)
 driver=out.parent/(out.name+'-camera-release.py');assert not driver.exists();compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),adapter_sha256=sha(__file__)),indent=2))
 sys.argv=[str(driver),*args];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
