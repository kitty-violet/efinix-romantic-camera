"""All existing release gates plus joint decoder/replica actual-map proofs."""
import inspect,json,re,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once
from camera_joint_board import BOARD,TRANSFORM,audit_delta
import finish_camera_prefetch_release as prior
from finish_split_logic_outputs_release import archived_placement_evidence
from finish_local_driver_release import wrappers
BASE=prior.BASE;PARENT=prior.PARENT;CAMERA_PARENT=prior.CAMERA_PARENT
SHORTS=ROOT/'tmp/frame-execution/camera-joint-evidence2'
RTL=SHORTS/'rtl';SERIAL=SHORTS.parent/(SHORTS.name+'-serial/new')
MAP_AUDIT=SHORTS/'map-audit';ADAPTER=SHORTS.parent/(SHORTS.name+'-adapter')

def parent_evidence(parent,verify):
 r=prior.parent_evidence(parent,verify);audit_delta(BOARD)
 r.update(status='FULL_VIDEO_A_CAPTURE_AND_JOINT_NATIVE_LINEAGE_BOUND',
  scope='Selected recovery parent, verified prefetch capture delta, and exact decoder plus identical local drivers. Historical proof map identities remain explicit; no board PASS inherited.')
 return r

def payload_evidence(bind,verify):
 r=prior.payload_evidence(bind,verify)
 m=bind(BOARD/'result.json');assert m['status']=='CAMERA_JOINT_FULL_BOARD_MAP_PASS';verify(m['evidence'])
 t=bind(TRANSFORM/'result.json');verify(t['inputs']);verify({str(TRANSFORM/'native_source.v'):t['source_sha256']})
 assert t['divider_exhaustive_patterns']==8192 and t['old_divider_depth']==12 and t['new_divider_depth']==8 and t['cycles_added']==0
 r.update(joint_map_sha256=m['map_sha256'],known_decoder_patterns=8192,cycles_added=0)
 return r

def camera_evidence(bind,verify):
 r=prior.camera_evidence(bind,verify);payload_evidence(bind,verify);mapped=sha(BOARD/'outflow/ti60f225_oob.map.v')
 root=ROOT/'tmp/frame-execution/camera-joint-primitives1';v=bind(root/'result.json');i=bind(root/'inputs.json')
 assert v['status']=='CAMERA_JOINT_ACTUAL_MAP_PRIMITIVES_PASS' and v['decoder_checks']==98304 and v['replica_checks']==1728512
 assert i['map_sha256']==mapped and i['transform_sha256']==sha(TRANSFORM/'result.json')
 assert v['inputs_sha256']==sha(root/'inputs.json') and i['map_result_sha256']==sha(BOARD/'result.json')
 verify({str(root/n):h for n,h in i['files'].items()});verify({str(root/'simulation.log'):v['log_sha256'],str(HERE/'validate_camera_joint.py'):i['runner_sha256']})
 root=ROOT/'tmp/frame-execution/camera-joint-capture3';v=bind(root/'result.json')
 assert v['status']=='PASS' and v['map_sha256']==mapped and v['cells']==1205 and v['burst']==16
 verify(v['inputs']);verify({str(root/'proof.json'):sha(root/'proof.json'),str(root/'sim.log'):sha(root/'sim.log')})
 assert 'checks=30000 pixels=7501 transactions=90' in (root/'sim.log').read_text()
 for f in ('test-camera-joint-capture.py','test-capture-native-boundary.py','check-camera-joint-progress.py','check-camera-prefetch-progress.py'):
  verify({str(ROOT/'tmp/board-test'/f):sha(ROOT/'tmp/board-test'/f)})
 v=bind(ROOT/'tmp/board-test/camera-joint-progress1.json');assert v['status']=='CAPTURE_PROGRESS_SIX_FLAGS_AND_PAGE_MUX_PASS' and v['map_sha256']==mapped and v['page_truth_checks']==48
 verify({str(ROOT/'tmp/board-test/check-camera-joint-progress.py'):v['runner_sha256']})
 c=bind(ROOT/'tmp/board-test/camera-joint-cdc-structure1.json');assert c['map_sha256']==mapped and len(c['held'])==76 and c['truth_checks']==41196
 verify({str(HERE/'camera_joint_cdc.py'):c['auditor_sha256']})
 reuse=bind(SHORTS/'reused_evidence.json');assert reuse['status']=='SUCCESSFUL_SAME_MAP_PREFIX_REUSED' and reuse['map_sha256']==mapped
 verify(reuse['evidence']);verify({str(HERE/'resume_camera_joint_shorts.py'):reuse['runner_sha256']})
 for f in ('camera_joint_semantics.py','camera_joint_gray.py','camera_joint_flow.py','camera_joint_implementation.py'):
  verify({str(HERE/f):sha(HERE/f)})
 r.update(joint_current_map_sha256=mapped,joint_actual_primitive_checks=98304+1728512,joint_actual_capture_checks=30000,current_cdc_truth_checks=41196,historical_map_identity_preserved=True)
 return r

def warning_audit(board):
 assert board==BOARD;r=prior.warning_audit(prior.BOARD)
 p=ROOT/'tmp/board-test/camera-joint-warnings1.json';v=json.loads(p.read_text())
 assert v['status']=='JOINT_UNUSED_CARRY_SUMS_AND_PROTECTED_OPEN_BITS_PROVED' and v['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
 for f,h in v['evidence'].items():assert sha(f)==h,f
 r.update(status=v['status'],joint_dangling_sums=v['new_dangling']);r['evidence'].update(v['evidence']);r['evidence'][str(p)]=sha(p)
 return r

def payload_prepare_code():
 from camera_early_ratio_route import prepare_code
 return once(prepare_code()[1],'from camera_checked_board import audit_delta','from camera_joint_board import audit_delta')

code=prior.code.replace('camera_prefetch_flow','camera_joint_flow')
exec(compile(code,__file__+':all-inherited-physical-gates','exec'),globals())

def package_code():
 source,code=prior.package_code()
 code=code.replace('camera_capture_prefetch_board','camera_joint_board').replace('camera_prefetch_implementation','camera_joint_implementation').replace('finish_camera_prefetch_release','finish_camera_joint_release')
 code=once(code,"mi['burst_builder_sha256']","mi['joint_builder_sha256']")
 code=code.replace('from camera_recovery_cdc import inventory','from camera_joint_cdc import inventory')
 code=code.replace('CAMERA_BURST_RELEASE.md','CAMERA_JOINT_RELEASE.md')
 old='Full video/A retained; capture writer uses reserved sixteen-beat bursts and synchronous prefetch buffering without a wide forwarding register; six progress flags added. Parent unaffected proofs plus current mapped/interface/physical gates; no inherited board PASS'
 new='Full video/A150MHz retained; reserved16-beat capture with synchronous prefetch, exact12-to8-level loader decode and identical local data/enable drivers. Current-map numerical/interface/CDC/physical evidence required; no inherited board PASS'
 code=once(code,old,new);compile(code,'joint-complete-package','exec');return source,code

def main():
 if sys.argv[1:]==['--check-only']:package_code();print('JOINT_ALL_RELEASE_GENERATORS_COMPILE');return
 mode=sys.argv.pop(1)
 if mode=='pll':
  import audit_pll_shift_routed as check
  check.audit_delta=audit_delta;check.main();return
 if mode=='bundle':source,code=package_code();key='--output'
 else:
  assert mode=='finish'
  from finish_local_driver_release import finish_code
  source,code=finish_code();code=code.replace('finish_local_driver_release.py','finish_camera_joint_release.py').replace('a.output.resolve()],240)','a.output.resolve()],360)');key='--root'
 args=sys.argv[1:];out=Path(args[args.index(key)+1]).resolve();audit_delta(BOARD);assert not out.exists()
 driver=out.parent/(out.name+'-camera-release.py');attempt=1
 while driver.exists():attempt+=1;driver=out.parent/(out.name+f'-camera-release-attempt{attempt}.py')
 compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),adapter_sha256=sha(__file__)),indent=2))
 sys.argv=[str(driver),*args];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
