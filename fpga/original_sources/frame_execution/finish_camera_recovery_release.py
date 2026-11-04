"""Full existing release gates plus affected camera recovery proofs; offline."""
import inspect,json,re,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once
from camera_recovery_board import BOARD,PARENT as BASE,audit_delta
from camera_checked_v2 import BOARD as CAMERA_PARENT
import finish_camera_ready_release as ready
import finish_camera_payload_release as payload
from finish_split_logic_outputs_release import archived_placement_evidence
from finish_local_driver_release import wrappers
PARENT=ROOT/'output/deployment/board-full-150-camera-ready-20261013'
SHORTS=ROOT/'tmp/frame-execution/camera-recovery-evidence1'
RTL=SHORTS/'rtl';SERIAL=ROOT/'tmp/frame-execution/camera-recovery-serial-questa1/new'
MAP_AUDIT=SHORTS/'map-audit';ADAPTER=SHORTS.parent/(SHORTS.name+'-adapter')

def parent_evidence(parent,verify):
 assert Path(parent['board'])==BASE;verify(parent['required_evidence']);audit_delta(BOARD)
 return dict(status='FULL_VIDEO_A_PARENT_CAMERA_RECOVERY_DELTA_BOUND',manifest=str(PARENT/'manifest.json'),sha256=sha(PARENT/'manifest.json'),
  scope='Full accepted A/video/DDR/control preserved by outside-cone graph proof. Boundary recovery and passive E page have current mapped tests. No board result inherited.')

def warning_audit(board):
 old=ready.warning_audit(BASE)
 current=board/'outflow/ti60f225_oob.warn.log';previous=BASE/'outflow/ti60f225_oob.warn.log'
 body=current.read_text();before=previous.read_text();assert '[EFX-0654' not in body
 signature=lambda text:re.findall(r"\[EFX-0201 WARNING\] Re-wiring to GND non-driven net/port '([^']+)'",text)
 assert signature(body)==signature(before)
 for marker in ('[EFX-0200 WARNING]','[EFX-0201 WARNING]'):assert body.count(marker)==before.count(marker)
 # No new unexplained dangling nets. A removed boundary cone's internal
 # declaration may be retained, but cannot be consumed by live cells.
 dangling=lambda t:set(re.findall(r'Net (.*?) is dangling',t))
 assert not (dangling(body)-dangling(before)),dangling(body)-dangling(before)
 r=json.loads((board/'result.json').read_text());assert r['protected_boundary']==dict(ports=1201,bits=4128,disconnected_bits=1374,other_connections_equal=True)
 return dict(status='UNCHANGED_PROTECTED_OPEN_INPUTS_NO_NEW_WARNING_SIGNATURE',evidence={**old['evidence'],str(current):sha(current),str(board/'result.json'):sha(board/'result.json'),str(board/'public_connectivity_audit.json'):sha(board/'public_connectivity_audit.json')},new_functional_unconnected_inputs=False)

def payload_evidence(bind,verify):
 # Old map IDs remain explicit. The complete new-map bijection preserves
 # the RAM specialization and ready factors; no numerical re-run is claimed.
 r=ready.payload_evidence(bind,verify)
 m=bind(BOARD/'result.json');verify(m['evidence']);assert m['status']=='CAMERA_RECOVERY_FULL_BOARD_MAP_PASS'
 r.update(recovery_map_sha256=m['map_sha256'],unchanged_parent_proofs_reused=True)
 return r

def camera_evidence(bind,verify):
 r=ready.camera_evidence(bind,verify);payload_evidence(bind,verify)
 mapped=ROOT/'tmp/frame-execution/camera-recovery-mapped2';m=bind(mapped/'result.json')
 assert m['status']=='CAMERA_RECOVERY_ACTUAL_MAP_SHORT_PASS' and len(m['cases'])==20 and m['map_sha256']==sha(BOARD/'outflow/ti60f225_oob.map.v')
 verify({str(mapped/n):h for n,h in m['inputs'].items()});verify({str(HERE/'test_camera_recovery_mapped.py'):m['runner_sha256'],str(mapped/'observer.log'):m['observer_log_sha256']})
 for c in m['cases']:verify({str(mapped/f"a{c['active']}-c{c['case']}.log"):c['log_sha256']})
 for folder in ('camera-boundary-recovery1','camera-fault-observer1','camera-recovery-adjacent1'):
  root=ROOT/'tmp/frame-execution'/folder;v=bind(root/'result.json');assert v['status']=='PASS';verify(v['inputs'])
 for f,status in [('camera-recovery-reader1.json','RECOVERY_READER_FRESHNESS_PAGES_RESET_ERRORS_PASS'),('camera-recovery-decoder1.json','PASS')]:
  v=bind(ROOT/'tmp/board-test'/f);assert v['status']==status;verify(v.get('evidence',v.get('inputs')))
 v=bind(ROOT/'tmp/board-test/camera-recovery-cdc-structure1.json');assert v['map_sha256']==m['map_sha256'] and len(v['held'])==76
 verify({str(HERE/'camera_recovery_cdc.py'):v['auditor_sha256']})
 unused=bind(ROOT/'tmp/board-test/camera-recovery-unused-pin-audit1.json')
 assert unused['status']=='MAPPED_OMITTED_INPUTS_PROVEN_UNUSED' and unused['map_sha256']==m['map_sha256'] and unused['truth_checks']==1024
 verify(unused['evidence'])
 r.update(recovery_mapped_cases=20,stream_held_bits=76,page_mux_truth_checks=828,reader_cases=7,parent_camera_map_proofs_reused=True)
 return r

def payload_prepare_code():
 from camera_early_ratio_route import prepare_code
 return once(prepare_code()[1],'from camera_checked_board import audit_delta','from camera_recovery_board import audit_delta')

code=payload.extra_code().replace('camera_payload_flow','camera_recovery_flow')
code=once(code," camera_evidence(bind,verify)"," camera_evidence(bind,verify)\n inv_new=bind(batch/'gray-seed3/clock_crossing_inventory.json')\n assert inv_new['camera_recovery_settling']['endpoints']==76 and inv_new['camera_recovery_settling']['max_delay_ns']<13.332\n assert inv_new['camera_recovery_structure']['map_sha256']==mapped['map_sha256']\n verify({inv_new['camera_recovery_settling']['report']:inv_new['camera_recovery_settling']['report_sha256']})")
exec(compile(code,__file__+':complete-extra-gates','exec'),globals())

def package_code():
 source,code=ready.package_code()
 code=code.replace('camera_ready_board','camera_recovery_board').replace('camera_ready_implementation','camera_recovery_implementation').replace('finish_camera_ready_release','finish_camera_recovery_release')
 code=code.replace('from camera_checked_cdc_v2 import inventory','from camera_recovery_cdc import inventory')
 code=once(code,'output/deployment/board-full-150-local-drivers-20261012','output/deployment/board-full-150-camera-ready-20261013')
 code=once(code,'from read_camera_checked_snapshot_v2 import main','from read_camera_recovery_snapshot import main')
 code=code.replace('CAMERA_CHECKED_RELEASE_EARLY_RATIO.md','CAMERA_RECOVERY_RELEASE.md')
 code=once(code,"observation_format='OFP5',","observation_format='OFP5',camera_page_schema='C7-config/E-first-fault',")
 code=once(code,"testlog=batch/'host_release_tests.log';assert not testlog.exists()",
  "testlog=batch/('host_release_tests_'+Path(__file__).stem+'.log');assert not testlog.exists()")
 code=once(code,"functional_evidence_scope='Full video/A retained, checked camera configuration, proved local RAM payload selection and exact combinational ready factoring; current-map tests and physical gates required'",
 "functional_evidence_scope='Full video/A retained; camera boundary cycle alignment, re-learning and RAW-reset alignment repaired; first-fault E page added. Parent unaffected proofs plus current mapped/interface/physical gates; no inherited board PASS'")
 compile(code,'recovery-package','exec');return source,code

def main():
 if sys.argv[1:]==['--check-only']:
  _,code=package_code();compile(code,'release-preflight','exec');print('RECOVERY_COMPLETE_RELEASE_GENERATORS_COMPILE');return
 mode=sys.argv.pop(1)
 if mode=='pll':
  import audit_pll_shift_routed as check
  check.audit_delta=audit_delta;check.main();return
 if mode=='bundle':source,code=package_code();key='--output'
 else:
  assert mode=='finish'
  from finish_local_driver_release import finish_code
  source,code=finish_code();code=code.replace('finish_local_driver_release.py','finish_camera_recovery_release.py').replace('a.output.resolve()],240)','a.output.resolve()],360)');key='--root'
 args=sys.argv[1:];out=Path(args[args.index(key)+1]).resolve();audit_delta(BOARD)
 driver=out.parent/(out.name+'-camera-release.py')
 # Preserve a failed packaging attempt and its original source/hash. The
 # output bundle itself must still be absent; only metadata drivers get a
 # fresh attempt name. All functional and physical gates run unchanged.
 assert not out.exists(),out
 attempt=1
 while driver.exists():
  attempt+=1;driver=out.parent/(out.name+f'-camera-release-attempt{attempt}.py')
 compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),adapter_sha256=sha(__file__)),indent=2))
 sys.argv=[str(driver),*args];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
