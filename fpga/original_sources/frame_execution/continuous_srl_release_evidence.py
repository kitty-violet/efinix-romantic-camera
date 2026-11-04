"""Reuse complete physical/CDC/bitstream gates and bind the finite-run delta."""
import inspect,json,re,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_legacy_cache_recovery import MANIFEST,prove
from control_command_repair import once
from continuous_srl_register_board import BOARD,audit_delta,TRANSFORM
from cnn_run_budget_native import PARENT as BASE
from finish_split_logic_outputs_release import archived_placement_evidence
from finish_local_driver_release import wrappers
import finish_camera_payload_release as prior
PARENT=ROOT/'output/deployment/board-full-150-camera-compact-20261014'
CAMERA_PARENT=prior.CAMERA_PARENT
SHORTS=ROOT/'tmp/frame-execution/continuous-srl-register-evidence1';RTL=SHORTS/'rtl'
SERIAL=SHORTS.parent/(SHORTS.name+'-serial/new');MAP_AUDIT=SHORTS/'map-audit';ADAPTER=SHORTS.parent/(SHORTS.name+'-adapter')

def parent_evidence(parent,verify):
 assert Path(parent['board'])==BASE
 verify(parent['required_evidence']);audit_delta(BOARD)
 return dict(status='FROZEN_FULL_VIDEO_A_PARENT_PLUS_BOUNDED_RUN_DELTA',manifest=str(PARENT/'manifest.json'),sha256=sha(PARENT/'manifest.json'),
  scope='Parent numerical/camera/DDR proofs retain original map identity. Current run-control/interface/physical evidence is separate; no inherited board PASS.')

def payload_evidence(bind,verify):
 import finish_cnn_run_budget_release as parent
 contract=parent.payload_evidence(bind,verify)
 from continuous_control_board import transform_check,PARENT as RUN_PARENT
 proof=transform_check();bind(TRANSFORM/'result.json');verify(proof['inputs'])
 assert contract==dict(deadline_cycles=33554432,counter_bits=26,hardware_frame_limit=16,progress_schema='D-run-progress-111')
 mapped=bind(BOARD/'result.json');verify(mapped['evidence']);audit_delta(BOARD)
 for n,status in [('cnn-observer-trim-controls2', 'MAPPED_WRITER_ERROR_COFACTOR_OFFICIAL_LUT_SHORT_PASS')]:
  root=ROOT/'tmp/frame-execution'/n;r=bind(root/'result.json');i=bind(root/'inputs.json')
  assert r['status']==status and r['inputs_sha256']==sha(root/'inputs.json')
  if 'map_sha256' in i:assert i['map_sha256']==mapped['map_sha256']
  verify(i.get('evidence',{}));verify({str(root/f):h for f,h in i['files'].items()})
  verify({str(root/'simulation.log'):r.get('log_sha256',r.get('simulation_sha256'))})
  if 'base_runner_sha256' in i:verify({str(HERE/'validate_native_local_drivers.py'):i['base_runner_sha256']})
 return contract

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
 retired=dangling(b)-dangling(a)
 assert dangling(a)<=dangling(b)
 proof=json.loads((TRANSFORM/'writer_cofactor_proof.json').read_text())
 allowed={r['old'][1] for r in proof['outputs']}
 assert retired<=allowed and len(retired)==6,retired-allowed
 from native_local_drivers import graph
 from camera_joint_native import OUT
 actual=graph((board/'outflow/ti60f225_oob.map.v').read_text())
 for name in retired:
  term=('net',name,None);drivers=[];loads=[]
  for n,(kind,pins) in actual.items():
   for pin,terms in pins.items():
    if term in terms:(drivers if pin in OUT.get(kind,[]) else loads).append((n,pin))
  assert len(drivers)==1 and not loads,(name,drivers,loads)
 m=json.loads((PARENT/'manifest.json').read_text());r=dict(m['native_warning_audit'])
 r.update(status='SAME_PROTECTED_UNUSED_BITS_AND_PROVED_RETIRED_WRITER_OUTPUTS_PASS',retired_writer_ce_outputs=sorted(retired),new_functional_unconnected_inputs=False,
  evidence={**r['evidence'],str(old):sha(old),str(new):sha(new),str(BOARD/'public_connectivity_audit.json'):sha(BOARD/'public_connectivity_audit.json'),str(Path(__file__)):sha(__file__),str(TRANSFORM/'writer_cofactor_proof.json'):sha(TRANSFORM/'writer_cofactor_proof.json')})
 return r

def payload_prepare_code():
 from continuous_srl_register_route import prepare_code
 return prepare_code()[1]

code=prior.extra_code().replace('-ratio-place-wrapper','-place-wrapper').replace('-prepare-payload.py','-prepare-cnn-budget.py')
code=once(code,"wrappers(physical,[('-place-wrapper',HERE/'camera_payload_flow.py')]","wrappers(physical,[('-place-wrapper',HERE/'continuous_srl_register_place.py')]")
code=once(code,"wrappers(batch,[('-local-driver-route',HERE/'camera_payload_flow.py')]","wrappers(batch,[('-local-driver-route',HERE/'continuous_srl_register_route.py')]")
code=once(code,"str(HERE/'camera_payload_flow.py'):pm['adapter_sha256']","str(HERE/'continuous_srl_register_route.py'):pm['adapter_sha256']")
exec(compile(code,__file__+':all-inherited-physical-gates','exec'),globals())
base_extra=extra_evidence
def extra_evidence(board,batch,physical,serial,bind,verify):
 assert bind(physical/'route_review.json')['route_profile_cli_override']==dict(before='CONGESTION_2',after='CONGESTION_2')
 base_extra(board,batch,physical,serial,bind,verify)
 from fixed_cnn_ram_locations import verify_placed as check_ram
 assert check_ram(physical)['count']==113
 assert not (physical/'arena_island_placement.json').exists()
 assert not (physical/'arena_island_actual_locations.json').exists()
 assert bind(physical/'route_review.json')['budget_seconds']==2700
 bind(physical/'cnn_ram_actual_locations.json');r=bind(physical/'cnn_ram_placement.json')
 if 'evidence' in r:verify(r['evidence'])
 for n in ('cnn_small_native.py','continuous_control_board.py','continuous_srl_register_place.py','continuous_control_flow.py','continuous_control_implementation.py','continuous_srl_register_route.py','validate_cnn_timing_drivers.py','validate_cnn_timing_controls.py','validate_cnn_small_controls.py','cnn_small_release_factory.py','cnn_small_board_factory.py','finish_cnn_run_budget_release.py','read_cnn_run_progress.py','read_cnn_run_snapshot.py'):
  verify({str(HERE/n):sha(HERE/n)})

def package_code():
 source,code=prior.package_code()
 code=code.replace('finish_camera_payload_release','finish_continuous_control_release').replace('camera_payload_board','continuous_control_board').replace('camera_payload_implementation','continuous_control_implementation')
 code=once(code,"PARENT=ROOT/'output/deployment/board-full-150-local-drivers-20261012'","PARENT=ROOT/'output/deployment/board-full-150-camera-compact-20261014'")
 code=once(code,"mapped['auditor_sha256']","mapped['evidence'][str(HERE/'continuous_control_board.py')]")
 code=code.replace('from camera_checked_cdc_v2 import inventory','from camera_joint_cdc import inventory')
 code=code.replace('CAMERA_CHECKED_RELEASE_EARLY_RATIO.md','CNN_RUN_BUDGET_RELEASE.md')
 code=once(code,'from read_camera_checked_snapshot_v2 import main','from read_cnn_run_snapshot import main')
 code=once(code,"manifest=dict(status=","manifest=dict(cnn_run_contract=payload_evidence(bind,verify),status=")
 code=once(code,'extra_evidence,parent_evidence,warning_audit,camera_evidence','extra_evidence,parent_evidence,warning_audit,camera_evidence,payload_evidence')
 code=once(code,"functional_evidence_scope='Full video/A retained, checked camera configuration plus proved local RAM payload specialization; current-map tests and physical gates required'",
 "functional_evidence_scope='Full video/A150MHz preserved; bounded issuer deadline and 26-bit count, hardware frame admission and atomic progress page changed. Exact late-writer-error factorization and targeted equivalent data-driver replication added without latency. Current-map regression and physical gates required; complete frame and board picture remain unverified'")
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
  source,code=finish_code();code=code.replace('finish_local_driver_release.py','finish_continuous_control_release.py').replace('a.output.resolve()],240)','a.output.resolve()],360)');key='--root'
 args=sys.argv[1:];out=Path(args[args.index(key)+1]).resolve();audit_delta(BOARD);assert not out.exists()
 driver=out.parent/(out.name+'-cnn-release.py');assert not driver.exists()
 driver.write_text(code,encoding='utf8');driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),adapter_sha256=sha(__file__)),indent=2))
 sys.argv=[str(driver),*args];runpy.run_path(str(driver),run_name='__main__')

_complete_extra_evidence=extra_evidence
def extra_evidence(board,batch,physical,serial,bind,verify):
    _complete_extra_evidence(board,batch,physical,serial,bind,verify)
    recovery=bind(MANIFEST)
    assert recovery==prove()
    verify(recovery['evidence'])

_original_package_code=package_code
def package_code():
    source,body=_original_package_code()
    detail=dict(status=prove()['status'],manifest=str(MANIFEST),sha256=sha(MANIFEST),historical_vdb_bytes_restored=False)
    body=once(body,'manifest=dict(','manifest=dict(legacy_cache_revalidation='+repr(detail)+',')
    return source,body




def payload_evidence(bind,verify):
    import finish_cnn_sparse_control_release as parent
    contract=parent.payload_evidence(bind,verify)
    from continuous_control_board_v2 import transform_check,audit_delta as current_audit
    proof=transform_check();bind(TRANSFORM/'result.json');verify(proof['inputs'])
    verify({str(TRANSFORM/'native_source.v'):proof['source_sha256'],str(TRANSFORM/'boundary.json'):proof['boundary_sha256']})
    assert proof['unchanged_primitives']==59191 and proof['protected_byte_identical']
    cut=bind(TRANSFORM/'expanded_cut_audit2.json');verify(cut['inputs'])
    assert cut['status']=='EXPANDED_NATIVE_CUT_PASS' and cut['crossing_loads']==407 and not cut['missing']
    mapped=bind(BOARD/'result.json');verify(mapped['evidence']);current_audit(BOARD)
    assert mapped['removed_dead_old_valid_luts']==[f'LUT__{i}' for i in range(159749,159754)]
    assert mapped['postmap_auditor_sha256']==sha(HERE/'continuous_control_board_v2.py')
    for name,status in (
        ('continuous-control-short2','CONTINUOUS_PREVIEW_CONTROL_SHORT_PASS'),
        ('preview-boundary-short7','ORIGINAL_MAPPED_CONTROLLER_EXPORT_MEANINGS_PASS'),
        ('preview-owner-short1','CONTINUOUS_OWNER_ADMISSION_DRAIN_PASS'),
        ('preview-mapped-control3','CONTINUOUS_ACTUAL_MAP_LOAD_VERIFY_REARM_PASS'),
        ('preview-mapped-admission1','ACTUAL_MAP_ADMISSION_ALL_BUDGETS_AND_SATURATION_PASS')):
        root=ROOT/'tmp/frame-execution'/name;r=bind(root/'result.json');i=bind(root/'inputs.json')
        assert r['status']==status and r['inputs_sha256']==sha(root/'inputs.json')
        verify(i.get('files',i))
        if 'map_sha256' in r:assert r['map_sha256']==mapped['map_sha256']
        if 'log_sha256' in r:
            log=root/'simulate.log'
            verify({str(log):r['log_sha256']})
        for stage in r.get('stages',[]):
            if 'markers' in stage:
                log=root/(stage['stage']+'.log');bind_log=log.read_text()
                assert all(marker in bind_log for marker in stage['markers'])
                verify({str(log):sha(log)})
        if (root/'simulate.log').exists():
            verify({str(root/'simulate.log'):sha(root/'simulate.log')})
    entry=bind(ROOT/'tmp/board-test/continuous-package-entry1.json');verify(entry['evidence'])
    assert entry['status']=='CONTINUOUS_PACKAGE_READER_IMPORT_HELP_AND_GATE_PASS'
    decoder=bind(ROOT/'tmp/board-test/continuous-progress-decoder1.json');verify(decoder['evidence'])
    assert decoder['status']=='CURRENT_MAPPED_PROGRESS_POLARITY_AND_SATURATION_PASS'
    assert decoder['map_sha256']==mapped['map_sha256']
    host=bind(ROOT/'tmp/board-test/continuous-run-host-tests1.json');verify(host['evidence'])
    assert host['status']=='BOUNDED_CONTINUOUS_HOST_POLICY_PASS' and host['tests']==13
    recovery=bind(ROOT/'tmp/frame-execution/continuous-srl-register-evidence1/recovery.json');verify(recovery['evidence'])
    return dict(contract,hardware_frame_limit=512,frame_budget_bits=10,boot_run_watchdog_cycles=9000000000,
        explicit_success_rearm=True,full_reload_on_rearm=True,diagnostic_admitted='saturated31_not_full_count',
        progress_schema='D-continuous-progress-v1',single_frame_cnn_deadline_cycles=33554432,
        actual_continuous_fps_validated=False,screen_frame_validated=False)

def warning_audit(board):
    import finish_cnn_sparse_control_release as parent
    from cnn_sparse_control_board import BOARD as old_board
    assert board==BOARD;audit_delta(board)
    result=parent.warning_audit(old_board)
    a=(old_board/'outflow/ti60f225_oob.warn.log').read_text()
    b=(board/'outflow/ti60f225_oob.warn.log').read_text()
    assert '[EFX-0654' not in b
    for marker in ('[EFX-0200 WARNING]','[EFX-0201 WARNING]'):assert a.count(marker)==b.count(marker)
    from probe_native_open_ports import protected_pins
    from audit_native_roundtrip import tokens
    from native_local_drivers import graph
    undriven=lambda s:re.findall(r"\[EFX-0201 WARNING\] Re-wiring to GND non-driven net/port '([^']+)'",s)
    for root,log in ((old_board,a),(board,b)):
        text=(root/'native_source.v').read_text();names=undriven(log)
        samples=tokens(protected_pins(text)['u_ddr3_top/samples'])
        assert len(set(names))==len(names)==10 and len(samples)==40
        bits=sorted(39-samples.index(n) for n in names)
        assert bits==[12,13,14,15,32,33,36,37,38,39]
        assert all(len(re.findall(r'\b'+re.escape(n)+r'\b',text))==1 for n in names)
        mapped=tokens(protected_pins((root/'outflow/ti60f225_oob.map.v').read_text())['u_ddr3_top/samples'])
        assert len(mapped)==40 and all(re.fullmatch(r'Open_\d+',mapped[39-bit]) for bit in bits)
    dangling=lambda s:set(re.findall(r'Net (.*?) is dangling',s))
    assert dangling(a)==dangling(b)
    result.update(status='SPARSE_CONTROL_IDENTICAL_WARNING_CLASSES_AND_PROTECTED_OPEN_BITS_PASS',new_functional_unconnected_inputs=False)
    result['evidence'].update({str(f):sha(f) for f in (old_board/'outflow/ti60f225_oob.warn.log',board/'outflow/ti60f225_oob.warn.log',board/'public_connectivity_audit.json',Path(__file__))})
    return result

_observer_extra_parent=extra_evidence
def extra_evidence(board,batch,physical,serial,bind,verify):
    _observer_extra_parent(board,batch,physical,serial,bind,verify)
    proof=bind(TRANSFORM/'result.json');verify(proof['inputs'])
    for name in ('continuous_control_native.py','continuous_control_board.py','continuous_control_flow.py','continuous_srl_register_place.py','continuous_control_implementation.py','continuous_srl_register_route.py','validate_preview_mapped_control.py','validate_preview_admission_mapped.py','audit_continuous_vectors.py','validate_continuous_adapter.py','read_continuous_progress.py','continuous_control_board_v2.py'):
        verify({str(HERE/name):sha(HERE/name)})
    canonical=batch/'canonical-seed3';cm=bind(canonical/'inputs.json')
    verify({str(HERE/'canonical_snapshot_sta_isolated.py'):cm['runner_sha256'],str(HERE/'canonical_snapshot_sta.py'):cm['canonical_source_sha256']})
    assert cm['process_isolation']=='timing_then_checks_both_required'
    stages=bind(canonical/'isolated_stages.json')
    assert [s['name'] for s in stages]==['sta','checks'] and all(s['returncode']==0 for s in stages)
    for stage in stages:verify({str(canonical/(stage['name']+'.log')):stage['log_sha256']})
    for n in ('cnn_observer_trim_native.py','cnn_observer_trim_board.py','cnn_observer_trim_place.py','cnn_observer_trim_flow.py','cnn_observer_trim_implementation.py','cnn_observer_trim_free_route.py','audit_observer_lut_folding.py','read_cnn_observer_trim.py','read_continuous_snapshot_v2.py'):
        verify({str(HERE/n):sha(HERE/n)})

_observer_package_parent=package_code
def package_code():
    source,body=_observer_package_parent()
    body=once(body,'from read_cnn_run_snapshot import main','from read_continuous_snapshot_v2 import main')
    body=once(body,'manifest=dict(',"manifest=dict(observer_diagnostic_profile='cnn-continuous-progress-v1',")
    return source,body

if __name__=='__main__':main()

