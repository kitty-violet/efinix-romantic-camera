"""Complete existing release gates plus current mapped capture-burst proofs."""
import inspect,json,re,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once
from camera_capture_prefetch_board import BOARD,PARENT as BASE,audit_delta
from camera_checked_v2 import BOARD as CAMERA_PARENT
from finish_split_logic_outputs_release import archived_placement_evidence
from finish_local_driver_release import wrappers
import finish_camera_recovery_release as prior
PARENT=ROOT/'output/deployment/board-full-150-camera-recovery-20261013'
SHORTS=ROOT/'tmp/frame-execution/camera-prefetch-evidence1'
RTL=SHORTS/'rtl';SERIAL=SHORTS.parent/(SHORTS.name+'-serial/new')
MAP_AUDIT=SHORTS/'map-audit';ADAPTER=SHORTS.parent/(SHORTS.name+'-adapter')
def parent_evidence(parent,verify):
 assert Path(parent['board'])==BASE;verify(parent['required_evidence']);audit_delta(BOARD)
 return dict(status='FULL_VIDEO_A_CAPTURE_BURST_DELTA_BOUND',manifest=str(PARENT/'manifest.json'),sha256=sha(PARENT/'manifest.json'),scope='Parent full video/A with exact capture/AWLEN replacement and six progress bits; unaffected graph and protected boundary preserved. No inherited board PASS.')
def warning_audit(board):
 assert board==BOARD
 p=ROOT/'tmp/board-test/camera-prefetch-warnings1.json';r=json.loads(p.read_text())
 assert r['status']=='UNCHANGED_PROTECTED_OPEN_BITS_WITH_RENAMED_UNUSED_NETS_PASS'
 assert r['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
 script=ROOT/'tmp/board-test/check-camera-prefetch-warning.py';assert r['auditor_sha256']==sha(script)
 for f,h in r['evidence'].items():assert sha(f)==h
 r['evidence'].update({str(p):sha(p),str(script):sha(script)});return r
def payload_evidence(bind,verify):
 # Preserve the parent's local RAM mux/ready proofs through the exact new
 # graph audit. Their old map identities remain explicit.
 r=prior.payload_evidence(bind,verify);m=bind(BOARD/'result.json')
 assert m['status']=='CAMERA_BURST_FULL_BOARD_MAP_PASS';verify(m['evidence'])
 r.update(capture_burst_map_sha256=m['map_sha256']);return r
def camera_evidence(bind,verify):
 r=prior.camera_evidence(bind,verify);payload_evidence(bind,verify)
 mapped=sha(BOARD/'outflow/ti60f225_oob.map.v')
 for folder in ('camera-capture-prefetch-unit1','camera-capture-prefetch-adjacent1'):
  root=ROOT/'tmp/frame-execution'/folder;v=bind(root/'result.json');assert v['status']=='PASS'
  verify({str(root/n):h for n,h in v['inputs'].items()})
  assert v['inputs']['rgb720_stream_writer.sv']==bind(BOARD/'inputs.json')['burst_variant_sha256']
  logs=list(root.glob('*sim*.log'));assert logs
  for f in logs:verify({str(f):sha(f)})
 root=ROOT/'tmp/frame-execution/camera-prefetch-integrated1';v=bind(root/'result.json')
 assert v['status']=='PASS' and v['burst']==16 and v['map_sha256']==mapped and v['cells']==1198
 verify(v['inputs']);verify({str(root/'sim.log'):sha(root/'sim.log')})
 assert 'PASS NATIVE_CAPTURE_BOUNDARY' in (root/'sim.log').read_text()
 for name,status,script in [('camera-prefetch-progress1.json','CAPTURE_PROGRESS_SIX_FLAGS_AND_PAGE_MUX_PASS','check-camera-prefetch-progress.py'),('camera-burst-decoder1.json','CAPTURE_BURST_DECODER_PASS','test-camera-burst-decoder.py')]:
  v=bind(ROOT/'tmp/board-test'/name);assert v['status']==status
  if 'map_sha256' in v:assert v['map_sha256']==mapped
  verify(v.get('evidence',{}));verify({str(ROOT/'tmp/board-test'/script):v.get('runner_sha256',sha(ROOT/'tmp/board-test'/script))})
 c=bind(ROOT/'tmp/board-test/camera-prefetch-cdc-structure1.json')
 assert c['map_sha256']==mapped and len(c['held'])==76
 verify({str(HERE/'camera_recovery_cdc.py'):c['auditor_sha256']})
 for f in ('read_camera_burst.py','read_camera_burst_snapshot.py','camera_prefetch_flow.py','camera_prefetch_implementation.py'):
  verify({str(HERE/f):sha(HERE/f)})
 r.update(capture_burst_beats=16,capture_buffer_beats=32,capture_progress_bits=6,capture_actual_map_checks=30000,capture_decoder_checks=169,stream_held_bits=76,parent_camera_proofs_map_identity_preserved=True)
 return r
def payload_prepare_code():
 from camera_early_ratio_route import prepare_code
 return once(prepare_code()[1],'from camera_checked_board import audit_delta','from camera_capture_prefetch_board import audit_delta')
# Reuse the complete routed canonical STA, Gray/held data, PLL, reset, and
# wrapper lineage gates. The callbacks above bind the changed capture scope.
code=prior.code.replace('camera_recovery_flow','camera_prefetch_flow')
exec(compile(code,__file__+':full-hardware-gates','exec'),globals())
def package_code():
 source,code=prior.package_code()
 code=code.replace('camera_recovery_board','camera_capture_prefetch_board').replace('camera_recovery_implementation','camera_prefetch_implementation').replace('finish_camera_recovery_release','finish_camera_prefetch_release')
 code=once(code,'output/deployment/board-full-150-camera-ready-20261013','output/deployment/board-full-150-camera-recovery-20261013')
 code=once(code,"verify({str(HERE/'camera_capture_prefetch_board.py'):mapped['auditor_sha256']})","verify({str(HERE/'camera_capture_prefetch_board.py'):mi['burst_builder_sha256']})")
 code=code.replace('from read_camera_recovery_snapshot import main','from read_camera_burst_snapshot import main').replace('CAMERA_RECOVERY_RELEASE.md','CAMERA_BURST_RELEASE.md')
 code=once(code,"camera_page_schema='C7-config/E-first-fault'","camera_page_schema='C7-config/E-capture-progress'")
 code=once(code,'Full video/A retained; camera boundary cycle alignment, re-learning and RAW-reset alignment repaired; first-fault E page added. Parent unaffected proofs plus current mapped/interface/physical gates; no inherited board PASS','Full video/A retained; capture writer uses reserved sixteen-beat bursts and synchronous prefetch buffering without a wide forwarding register; six progress flags added. Parent unaffected proofs plus current mapped/interface/physical gates; no inherited board PASS')
 compile(code,'capture-burst-package','exec');return source,code
def main():
 if sys.argv[1:]==['--check-only']:
  package_code();print('CAPTURE_BURST_COMPLETE_RELEASE_GENERATOR_COMPILES');return
 mode=sys.argv.pop(1)
 if mode=='pll':
  import audit_pll_shift_routed as check
  check.audit_delta=audit_delta;check.main();return
 if mode=='bundle':source,code=package_code();key='--output'
 else:
  assert mode=='finish'
  from finish_local_driver_release import finish_code
  source,code=finish_code();code=code.replace('finish_local_driver_release.py','finish_camera_prefetch_release.py').replace('a.output.resolve()],240)','a.output.resolve()],360)');key='--root'
 args=sys.argv[1:];out=Path(args[args.index(key)+1]).resolve();audit_delta(BOARD);assert not out.exists()
 driver=out.parent/(out.name+'-camera-release.py');attempt=1
 while driver.exists():attempt+=1;driver=out.parent/(out.name+f'-camera-release-attempt{attempt}.py')
 compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),adapter_sha256=sha(__file__)),indent=2))
 sys.argv=[str(driver),*args];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
