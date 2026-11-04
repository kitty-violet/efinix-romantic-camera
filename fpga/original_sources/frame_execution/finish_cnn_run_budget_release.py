"""Reuse complete physical/CDC/bitstream gates and bind the finite-run delta."""
import inspect,json,re,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once
from cnn_run_budget_board import BOARD,audit_delta,TRANSFORM
from cnn_run_budget_native import PARENT as BASE
from finish_split_logic_outputs_release import archived_placement_evidence
from finish_local_driver_release import wrappers
import finish_camera_payload_release as prior
PARENT=ROOT/'output/deployment/board-full-150-camera-compact-20261014'
CAMERA_PARENT=prior.CAMERA_PARENT
SHORTS=ROOT/'tmp/frame-execution/cnn-run-budget-evidence1';RTL=SHORTS/'rtl'
SERIAL=SHORTS.parent/(SHORTS.name+'-serial/new');MAP_AUDIT=SHORTS/'map-audit';ADAPTER=SHORTS.parent/(SHORTS.name+'-adapter')

def parent_evidence(parent,verify):
 assert Path(parent['board'])==BASE
 verify(parent['required_evidence']);audit_delta(BOARD)
 return dict(status='FROZEN_FULL_VIDEO_A_PARENT_PLUS_BOUNDED_RUN_DELTA',manifest=str(PARENT/'manifest.json'),sha256=sha(PARENT/'manifest.json'),
  scope='Parent numerical/camera/DDR proofs retain original map identity. Current run-control/interface/physical evidence is separate; no inherited board PASS.')

def payload_evidence(bind,verify):
 mapped=bind(BOARD/'result.json');verify(mapped['evidence'])
 t=bind(TRANSFORM/'result.json');verify(t['inputs']);verify({str(TRANSFORM/'native_source.v'):t['source_sha256']})
 expected={'cnn-run-budget-short3':'CNN_RUN_BUDGET_SHORT_PASS','cnn-owner-admission1':'OWNER_ADMISSION_RACE_REPRODUCED_AND_FIXED',
 'cnn-budget-real-program1':'ACCEPTED_A_AUTO_DW_BUDGET_ISSUE_PASS','cnn-native-budget-short5':'NATIVE_COUNTER_AND_CPU_CYCLE_READBACK_PASS',
 'cnn-progress-page1':'NATIVE_PACKING_SCAN_HOST_DECODE_PASS','cnn-budget-mapped2':'ACTUAL_MAPPED_ADMISSION_AND_PROGRESS_PAGE_PASS'}
 for name,status in expected.items():
  root=ROOT/'tmp/frame-execution'/name;r=bind(root/'result.json');assert r['status']==status
  if 'inputs' in r:verify(r['inputs'])
  if (root/'inputs.json').exists():
   i=bind(root/'inputs.json')
   if 'inputs_sha256' in r:assert sha(root/'inputs.json')==r['inputs_sha256']
   rows=i.get('inputs',i)
   if isinstance(rows,dict):verify(rows)
   else:
    verify({v['snapshot']:v['sha256'] for v in rows});verify({v['source']:v['source_sha256'] for v in rows})
   for v in i.get('descriptor_records',[]):verify({v['source']:v['sha256'],v['hex_snapshot']:v['hex_sha256']})
   if 'vector_manifest_sha256' in i:
    vfile=Path(i['vectors'])/'frame_manifest.json';verify({str(vfile):i['vector_manifest_sha256']})
    vm=bind(vfile);verify({str(Path(i['vectors'])/n):vm['files'][n] for n in ('program.hex','dma_commands.hex')})
  if 'log_sha256' in r:verify({str(root/'simulation.log'):r['log_sha256']})
  if 'simulation_sha256' in r:verify({str(root/'simulation.log'):r['simulation_sha256']})
  for v in r.get('results',[]):verify({str(root/(v['name']+'.log')):v['log_sha256']})
  if 'map_sha256' in r:assert r['map_sha256']==mapped['map_sha256']
 return dict(deadline_cycles=33554432,counter_bits=26,hardware_frame_limit=16,progress_schema='D-run-progress-111')

def camera_evidence(bind,verify):
 parent=bind(PARENT/'manifest.json');verify(parent['required_evidence']);contract=payload_evidence(bind,verify)
 return dict(parent['camera_checks'],historical_camera_map_sha256=sha(BASE/'outflow/ti60f225_oob.map.v'),
  current_map_sha256=sha(BOARD/'outflow/ti60f225_oob.map.v'),current_run_contract=contract,
  parent_camera_proofs_reused_via_exact_unchanged_graph=True)

def warning_audit(board):
 assert board==BOARD;audit_delta(board)
 old=BASE/'outflow/ti60f225_oob.warn.log';new=board/'outflow/ti60f225_oob.warn.log';a,b=old.read_text(),new.read_text()
 assert '[EFX-0654' not in b
 for marker in ('[EFX-0200 WARNING]','[EFX-0201 WARNING]'):assert a.count(marker)==b.count(marker)
 undriven=lambda s:re.findall(r"\[EFX-0201 WARNING\] Re-wiring to GND non-driven net/port '([^']+)'",s)
 from probe_native_open_ports import protected_pins
 from audit_native_roundtrip import tokens
 for root,body in ((BASE,a),(board,b)):
  source=(root/'native_source.v').read_text();samples=tokens(protected_pins(source)['u_ddr3_top/samples'])
  names=undriven(body);assert len(names)==10 and len(samples)==40
  assert sorted(39-samples.index(n) for n in names)==[12,13,14,15,32,33,36,37,38,39]
  assert all(len(re.findall(r'\b'+n+r'\b',source))==1 for n in names)
 dangling=lambda s:set(re.findall(r'Net (.*?) is dangling',s))
 assert dangling(a)==dangling(b)
 m=json.loads((PARENT/'manifest.json').read_text());r=dict(m['native_warning_audit'])
 r.update(status='SAME_PROTECTED_UNUSED_SAMPLE_BITS_AND_WARNING_CLASSES_PASS',new_functional_unconnected_inputs=False,
  evidence={**r['evidence'],str(old):sha(old),str(new):sha(new),str(BOARD/'public_connectivity_audit.json'):sha(BOARD/'public_connectivity_audit.json'),str(Path(__file__)):sha(__file__)})
 return r

def payload_prepare_code():
 from cnn_run_budget_c1_flow import prepare_code
 return prepare_code()[1]

code=prior.extra_code().replace('-ratio-place-wrapper','-place-wrapper').replace('-prepare-payload.py','-prepare-cnn-budget.py')
code=once(code,"wrappers(physical,[('-place-wrapper',HERE/'camera_payload_flow.py')]","wrappers(physical,[('-place-wrapper',HERE/'cnn_run_budget_place.py')]")
code=once(code,"wrappers(batch,[('-local-driver-route',HERE/'camera_payload_flow.py')]","wrappers(batch,[('-local-driver-route',HERE/'cnn_run_budget_c1_flow.py')]")
code=once(code,"str(HERE/'camera_payload_flow.py'):pm['adapter_sha256']","str(HERE/'cnn_run_budget_c1_flow.py'):pm['adapter_sha256']")
exec(compile(code,__file__+':all-inherited-physical-gates','exec'),globals())
base_extra=extra_evidence
def extra_evidence(board,batch,physical,serial,bind,verify):
 assert bind(physical/'route_review.json')['route_profile_cli_override']==dict(before='CONGESTION_2',after='CONGESTION_1')
 base_extra(board,batch,physical,serial,bind,verify)
 from cnn_run_budget_locations import verify_placed
 from fixed_cnn_ram_locations import verify_placed as check_ram
 assert verify_placed(physical)['count']==36 and check_ram(physical)['count']==113
 for n in ('arena_island_placement.json','arena_island_actual_locations.json','cnn_ram_placement.json','cnn_ram_actual_locations.json'):
  r=bind(physical/n)
  if 'evidence' in r:verify(r['evidence'])
 for n in ('cnn_run_budget_locations.py','cnn_run_budget_place.py','cnn_run_budget_flow.py','cnn_run_budget_implementation.py','cnn_run_budget_route_prepare.py','cnn_run_budget_c1_flow.py','read_cnn_run_progress.py','read_cnn_run_snapshot.py'):
  verify({str(HERE/n):sha(HERE/n)})

def package_code():
 source,code=prior.package_code()
 code=code.replace('finish_camera_payload_release','finish_cnn_run_budget_release').replace('camera_payload_board','cnn_run_budget_board').replace('camera_payload_implementation','cnn_run_budget_implementation')
 code=once(code,"PARENT=ROOT/'output/deployment/board-full-150-local-drivers-20261012'","PARENT=ROOT/'output/deployment/board-full-150-camera-compact-20261014'")
 code=once(code,"mapped['auditor_sha256']","mapped['evidence'][str(HERE/'cnn_run_budget_board.py')]")
 code=code.replace('from camera_checked_cdc_v2 import inventory','from camera_joint_cdc import inventory')
 code=code.replace('CAMERA_CHECKED_RELEASE_EARLY_RATIO.md','CNN_RUN_BUDGET_RELEASE.md')
 code=once(code,'from read_camera_checked_snapshot_v2 import main','from read_cnn_run_snapshot import main')
 code=once(code,"manifest=dict(status=","manifest=dict(cnn_run_contract=payload_evidence(bind,verify),status=")
 code=once(code,'extra_evidence,parent_evidence,warning_audit,camera_evidence','extra_evidence,parent_evidence,warning_audit,camera_evidence,payload_evidence')
 code=once(code,"functional_evidence_scope='Full video/A retained, checked camera configuration plus proved local RAM payload specialization; current-map tests and physical gates required'",
 "functional_evidence_scope='Full video/A150MHz preserved; bounded issuer deadline and 26-bit count, hardware frame admission and atomic progress page changed. Current-map regression and physical gates required; complete frame and board picture remain unverified'")
 compile(code,'cnn-budget-full-package','exec');return source,code

def main():
 if sys.argv[1:]==['--check-only']:
  package_code();print('CNN_BUDGET_ALL_RELEASE_GENERATORS_COMPILE');return
 if sys.argv[1:]==['--proofs-only']:
  evidence={}
  def bind(p):evidence[str(p)]=sha(p);return json.loads(p.read_text())
  def verify(rows):
   for f,h in rows.items():assert sha(f)==h,f;evidence[f]=h
  parent_evidence(bind(PARENT/'manifest.json'),verify);camera_evidence(bind,verify);warning_audit(BOARD)
  out=ROOT/'tmp/board-test/cnn-budget-release-preflight1.json';assert not out.exists()
  out.write_text(json.dumps(dict(status='CURRENT_FUNCTIONAL_RELEASE_PREFLIGHT_PASS_PHYSICAL_PENDING',evidence=evidence,hardware_access=False),indent=2));print('CNN_BUDGET_FUNCTIONAL_PREFLIGHT_PASS');return
 mode=sys.argv.pop(1)
 if mode=='pll':
  import audit_pll_shift_routed as check
  check.audit_delta=audit_delta;check.main();return
 if mode=='bundle':source,code=package_code();key='--output'
 else:
  assert mode=='finish'
  from finish_local_driver_release import finish_code
  source,code=finish_code();code=code.replace('finish_local_driver_release.py','finish_cnn_run_budget_release.py').replace('a.output.resolve()],240)','a.output.resolve()],360)');key='--root'
 args=sys.argv[1:];out=Path(args[args.index(key)+1]).resolve();audit_delta(BOARD);assert not out.exists()
 driver=out.parent/(out.name+'-cnn-release.py');assert not driver.exists()
 driver.write_text(code,encoding='utf8');driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),adapter_sha256=sha(__file__)),indent=2))
 sys.argv=[str(driver),*args];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
