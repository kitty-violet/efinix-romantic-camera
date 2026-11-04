"""OFP5 observation delta bound to its own tests, route and read-only hosts."""
import inspect,json
from pathlib import Path
from run_units import ROOT,HERE,sha
from startup_cause_board import BASE,audit_delta
SHORTS=ROOT/'tmp/frame-execution/startup-cause-evidence1'
RTL=ROOT/'tmp/frame-execution/startup-cause-control-rtl2'
SERIAL=ROOT/'tmp/frame-execution/startup-cause-evidence1-serial-questa2/new'
PARENT=ROOT/'output/deployment/board-full-150-fault-probe-20261011'
MAP_AUDIT=ROOT/'tmp/frame-execution/startup-cause-map-audit1'
ADAPTER=ROOT/'tmp/frame-execution/startup-cause-adapter-idle2'

def parent_evidence(parent,verify):
 assert Path(parent['board'])==BASE
 verify(parent['required_evidence'])
 return dict(status='EXACT_OFP4_PARENT_EVIDENCE_UNCHANGED',
  manifest=str(PARENT/'manifest.json'),sha256=sha(PARENT/'manifest.json'),
  scope='Functional video/A/control/DDR/pins/clocks unchanged; current observation implementation separately verified.')

def extra_evidence(board,batch,physical,serial,bind,verify):
 assert audit_delta(board)==bind(board/'delta_audit.json')
 mapped=bind(board/'result.json');ma=bind(MAP_AUDIT/'result.json')
 assert ma['map_sha256']==mapped['map_sha256'] and ma['board_inputs_sha256']==sha(board/'inputs.json')
 assert ma['status']=='FAULT_PROBE_ACTUAL_CONTROL_BOOLEAN_AND_CDC_MAP_PASS' and ma['checks']==17504
 verify(ma['evidence'])
 for name,checker in (('new_diagnostic_cells.json','audit_startup_cause_map.py'),
                       ('held_data_structure.json','startup_cause_clock_cdc.py')):
  r=bind(MAP_AUDIT/name);assert r['map_sha256']==mapped['map_sha256']
  verify({str(HERE/checker):r['auditor_sha256']})
 from startup_cause_flow import proof
 unit=ROOT/'tmp/frame-execution/startup-cause-shorts4';proof(board,unit)
 ur=bind(unit/'result.json');ui=bind(unit/'inputs.json')
 verify({str(unit/n):h for n,h in ui['files'].items()})
 verify({str(unit/x['log']):x['sha256'] for x in ur['results']})
 for folder,status in ((SHORTS/'observe','MAPPED_DDR_STATUS_AND_EVENTS_CONTROL_RESET_PASS'),
                        (ADAPTER,'ACTUAL_MAPPED_CONTROL_ADAPTER_IDLE_THREE_RESETS_PASS')):
  r=bind(folder/'result.json');i=bind(folder/'inputs.json')
  assert r['status']==status and r['inputs_sha256']==sha(folder/'inputs.json')
  assert i.get('source_map_sha256',i.get('original_map_sha256'))==mapped['map_sha256']
  verify({str(folder/n):h for n,h in i['files'].items()})
  verify({str(folder/'simulation.log'):r.get('log_sha256',r.get('simulation_sha256'))})
 import finish_local_decode_release as old
 code=inspect.getsource(old.extra_evidence)
 code='def inherited_physical(board,batch,physical,serial,bind,verify):\n'+code[code.index("    si=bind(serial/'inputs.json')"):]
 code=code.replace('from inline_official_state_luts import inline','from inline_fault_probe_luts import inline')
 code=code.replace("HERE/'inline_official_state_luts.py'","HERE/'inline_fault_probe_luts.py'")
 code=code.replace("len(details['instances'])==23","len(details['instances'])==proof['primitive_instances_removed']")
 scope=dict(old.__dict__);exec(compile(code,old.__file__+':OFP5-physical','exec'),scope)
 scope['inherited_physical'](board,batch,physical,serial,bind,verify)
 driver=batch.parent/(batch.name+'-driver.py');dm=bind(driver.with_suffix('.json'))
 verify({str(driver):dm['driver_sha256'],dm['source']:dm['source_sha256'],str(HERE/'startup_cause_route.py'):dm['wrapper_sha256']})
 inv=bind(batch/'gray-seed3/clock_crossing_inventory.json')
 assert inv['cell_probe_bundled_settling']['max_delay_ns']<13.332
 assert inv['cell_probe_structure']['map_sha256']==mapped['map_sha256']
 if (batch/'recovery.json').exists():
  recovered=bind(batch/'recovery.json');verify({recovered['path']:recovered['sha256']})
  record=bind(Path(recovered['path']))
  assert record['status']=='INTACT_ROUTE_STA_CDC_REPORTS_REVALIDATED' and not record['tool_rerun']
  verify(record['required_evidence'])
  verify({r['preserved']:r['sha256'] for r in record['backups']})
  verify({str(ROOT/'tmp/board-test/recover-startup-cause-reports.py'):record['script_sha256']})
  verify({str(HERE/'windows_bounded_job.py'):sha(HERE/'windows_bounded_job.py')})
 for name in ('startup_cause_board.py','startup_cause_flow.py','startup_cause_route.py','startup_cause_clock_cdc.py',
  'audit_startup_cause_gray.py','audit_startup_cause_map.py','validate_startup_cause.py','read_startup_cause.py',
  'read_startup_cause_snapshot.py','read_startup_cause_status.py','test_startup_cause_host.py','finish_startup_cause_release.py'):
  verify({str(HERE/name):sha(HERE/name)})

def main():
 import finish_fault_probe_release as parent
 import finish_local_decode_release as original
 for name in ('BASE','SHORTS','RTL','SERIAL','PARENT','MAP_AUDIT','ADAPTER','audit_delta'):
  setattr(parent,name,globals()[name])
 execute=original.execute
 def execute_ofp5(source,code,driver,args):
  if source.name=='finish_command_progress_release.py':
   # OFP4 binding already took 87.407 s of its 90 s limit. OFP5 adds
   # held-data CDC/map checks; budget hashing separately from hardware work.
   before="'--output',a.output.resolve()],90)"
   assert code.count(before)==1
   code=code.replace(before,"'--output',a.output.resolve()],180)")
  replacements={
   'finish_fault_probe_release':'finish_startup_cause_release',
   'from fault_probe_board import BASE,audit_delta':'from startup_cause_board import BASE,audit_delta',
   "verify({str(HERE/'fault_probe_board.py'):mi['fault_probe_transform_sha256']})":
    "verify({str(HERE/'startup_cause_board.py'):mi['startup_cause_transform_sha256']})",
   'output/deployment/board-full-150-complete-transition2-20261011':'output/deployment/board-full-150-fault-probe-20261011',
   'from fault_probe_clock_cdc import inventory':'from startup_cause_clock_cdc import inventory',
   "'test_fault_probe_host','-v'":"'test_fault_probe_host','test_startup_cause_host','-v'",
   "'Ran 31 tests'":"'Ran 40 tests'",
   "('read_fault_probe','read_fault_probe_status')":"('read_startup_cause_snapshot','read_startup_cause_status')",
   "name=='read_fault_probe_status'":"name=='read_startup_cause_status'",
   'read_fault_probe.py只读':'read_startup_cause_snapshot.py只读',
   'read_fault_probe_status.py读':'read_startup_cause_status.py读',
   'OFP4':'OFP5','770位':'1026位',
   '完整视频/A系统的适配器首错和DDR时钟观察':'完整视频/A系统的启动首因及原语有限对照',
   '只新增只读观察：适配器首次错误前一周期的状态/地址/握手，DDR四时钟相对频率及复位/CKE历史；保持原功能语句。该包用于定位余下板测故障，不宣称DDR校准已修复。':
    '只新增只读观察：首次非空闲及不相容跳转历史、两时钟域有限原语原始采样；保留DDR四时钟与复位/CKE观察，保持完整视频/A功能。该包用于定位尚未证实的启动根因，不宣称DDR校准已修复。'}
  for before,after in replacements.items():code=code.replace(before,after)
  return execute(source,code,driver,args)
 original.execute=execute_ofp5
 parent.main()
if __name__=='__main__':main()
