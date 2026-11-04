"""Bind the independent-output candidate to its own complete physical gates."""
import inspect,json
from pathlib import Path
from run_units import ROOT,HERE,sha
from split_logic_outputs_board import BASE,audit_delta
import finish_startup_cause_release as prior

SHORTS=ROOT/'tmp/frame-execution/split-outputs-evidence1'
RTL=SHORTS/'rtl'
SERIAL=ROOT/'tmp/frame-execution/split-outputs-evidence1-serial2/new'
MAP_AUDIT=SHORTS/'map-audit'
ADAPTER=ROOT/'tmp/frame-execution/split-outputs-evidence1-adapter'
PARENT=ROOT/'output/deployment/board-full-150-startup-cause-20261011'

def archived_placement_evidence(rows,physical,bind):
 # The reviewed pre-route result is copied verbatim by the route driver before
 # that driver publishes a new routed result at the original path. Bind both
 # phases explicitly; never rewrite the frozen review or accept a hash drift.
 rows=dict(rows);key=str(physical/'result.json')
 if key not in rows:return rows
 expected=rows.pop(key);archive=physical/'before_route/result.json'
 assert sha(archive)==expected
 previous=bind(archive);current=bind(physical/'result.json')
 assert previous['status']=='PLACEMENT_ONLY_FINISHED_NOT_RELEASE'
 assert previous['stages'] and all(r['exit_code']==0 for r in previous['stages'])
 assert current['status']=='PHYSICAL_CAPACITY_PROBE_FINISHED'
 assert current['stages'][:-1]==previous['stages']
 assert current['stages'][-1]['phase']=='route_after_review' and current['stages'][-1]['exit_code']==0
 record=dict(status='PRE_ROUTE_RESULT_ARCHIVE_HASH_AND_ROUTED_SUCCESS_BOUND',
  reviewed_path=key,reviewed_sha256=expected,archived_path=str(archive),
  current_sha256=sha(physical/'result.json'),frozen_review_unchanged=True)
 path=physical/'result_phase_binding.json'
 if path.exists():assert json.loads(path.read_text())==record
 else:path.write_text(json.dumps(record,indent=2))
 bind(path);rows[str(archive)]=expected
 return rows

def parent_evidence(parent,verify):
 assert Path(parent['board'])==BASE
 verify(parent['required_evidence'])
 return dict(status='EXACT_OFP5_PARENT_EVIDENCE_UNCHANGED',manifest=str(PARENT/'manifest.json'),
  sha256=sha(PARENT/'manifest.json'),scope='Functional video/A RTL and assets, protected IP, clocks, pins and constraints unchanged. Synthesis packing and equivalent finite probe changed, with separate current-map/physical evidence; no inherited board PASS.')

def extra_evidence(board,batch,physical,serial,bind,verify):
 # Reuse all previous physical/reset/CDC/asset requirements, rebound to this
 # map and its short gates. This does not run or modify any previous package.
 code=inspect.getsource(prior.extra_evidence)
 replacements={
  "('new_diagnostic_cells.json','audit_startup_cause_map.py')":"('new_diagnostic_cells.json','audit_split_logic_outputs_map.py')",
  "('held_data_structure.json','startup_cause_clock_cdc.py')":"('held_data_structure.json','split_logic_outputs_cdc.py')",
  'from startup_cause_flow import proof':
   'import startup_cause_flow as source_flow\n source_flow.audit_delta=audit_delta\n from split_logic_outputs_flow import proof',
  "tmp/frame-execution/startup-cause-shorts4":"tmp/frame-execution/split-outputs-shorts2",
  "str(HERE/'startup_cause_route.py'):dm['wrapper_sha256']":"str(HERE/'split_logic_outputs_route.py'):dm['wrapper_sha256']"}
 for before,after in replacements.items():
  assert code.count(before)==1,before;code=code.replace(before,after)
 if (physical/'route_inheritance.json').exists():
  # A route-only comparison retains the original placement; verify the actual
  # generated wrapper and its generator instead of pretending it is the
  # unmodified default C2 route wrapper.
  needle="str(HERE/'split_logic_outputs_route.py'):dm['wrapper_sha256']"
  assert code.count(needle)==1
  code=code.replace(needle,"str(batch.parent/(batch.name+'-wrapper.py')):dm['wrapper_sha256']")
  inherited=bind(physical/'route_inheritance.json')
  assert inherited['status']=='EXACT_PLACEMENT_REUSED_FOR_ROUTE_PROFILE_COMPARISON'
  assert all(inherited[k] is False for k in ('placement_executed','rtl_changed','constraints_changed','seed_changed','hardware_access'))
  assert inherited['route_profile_cli_override']==dict(before='CONGESTION_2',after='TIMING_1')
  verify(inherited['source_files']);verify(inherited['copied_files'])
  origin=Path(inherited['source'])
  verify({str(origin/'physical_inputs.json'):inherited['source_physical_inputs_sha256'],
   str(origin/'before_route/result.json'):inherited['source_placement_result_sha256'],
   inherited['source_route_result']:inherited['source_route_result_sha256'],
   inherited['generator']:inherited['generator_sha256']})
  wrapper=batch.parent/(batch.name+'-wrapper.py');wm=bind(wrapper.with_suffix('.json'))
  verify({str(wrapper):wm['driver_sha256'],wm['source']:wm['source_sha256'],wm['generator']:wm['generator_sha256']})
  assert 'optimization_level=TIMING_1' in (physical/'route_only.cmd').read_text()
  assert 'optimization_level=TIMING_1' in (batch/'route.log').read_text()
  bind(physical/'data_arc_analysis.json')
 # Adapt the nested physical evidence checks to the exact generated placement
 # wrapper, and to the explicitly archived pre-route phase result.
 anchor=" scope=dict(old.__dict__);exec(compile(code,old.__file__+':OFP5-physical','exec'),scope)"
 inject=""" code=code.replace("verify(route['frozen']);verify(route['required_evidence'])","verify(route['frozen']);verify(archived_placement_evidence(route['required_evidence'],physical,bind))")
 code=code.replace("verify(review['required_evidence'])","verify(archived_placement_evidence(review['required_evidence'],physical,bind))")
 needle="    verify({str(driver):driver_meta['driver_sha256'],driver_meta['source']:driver_meta['source_sha256'],str(wrapper):driver_meta['wrapper_sha256']})"
 assert code.count(needle)==1
 code=code.replace(needle,"    wrapper=physical.parent/(physical.name+'-eight-thread-wrapper.py')\\n"+needle)
 scope=dict(old.__dict__,archived_placement_evidence=archived_placement_evidence);exec(compile(code,old.__file__+':OFP5-physical','exec'),scope)"""
 assert code.count(anchor)==1
 code=code.replace(anchor,inject)
 scope=dict(prior.__dict__);scope.update(globals())
 exec(compile(code,prior.__file__+':split-output-evidence','exec'),scope)
 scope['extra_evidence'](board,batch,physical,serial,bind,verify)
 completed=bind(SHORTS/'result.json')
 assert completed['status']=='SPLIT_OUTPUTS_SAME_MAP_SHORT_INTEGRATION_PASS'
 assert completed['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
 verify(completed['evidence'])
 structure=bind(MAP_AUDIT/'new_diagnostic_cells.json')
 assert structure['whole_map_comb4']==0 and structure['truth_checks']==64
 alias=bind(SERIAL.parent.parent/(SERIAL.parent.name+'-reference-binding.json'))
 assert alias['map_sha256']==completed['map_sha256'] and not alias['hardware_source_changed']
 assert not alias['assertions_removed']
 verify({str(HERE/'split_logic_outputs_serial.py'):alias['wrapper_sha256']})
 for n in ('split_logic_outputs_board.py','validate_split_logic_outputs.py','split_logic_outputs_flow.py',
  'audit_split_logic_outputs_map.py','split_logic_outputs_cdc.py','audit_split_logic_outputs_gray.py',
  'split_logic_outputs_route.py','run_split_logic_outputs_evidence.py','collate_split_logic_outputs.py',
  'split_logic_outputs_serial.py','finish_split_logic_outputs_release.py','windows_bounded_job.py'):
  verify({str(HERE/n):sha(HERE/n)})

def main():
 for n in ('BASE','SHORTS','RTL','SERIAL','MAP_AUDIT','ADAPTER','PARENT','audit_delta'):setattr(prior,n,globals()[n])
 import finish_local_decode_release as original
 previous=original.execute
 def execute(source,code,driver,args):
  code=code.replace('finish_startup_cause_release','finish_split_logic_outputs_release')
  code=code.replace('from startup_cause_board import BASE,audit_delta','from split_logic_outputs_board import BASE,audit_delta')
  code=code.replace("verify({str(HERE/'startup_cause_board.py'):mi['startup_cause_transform_sha256']})",
   "verify({str(HERE/'split_logic_outputs_board.py'):mi['split_logic_transform_sha256']})")
  code=code.replace('output/deployment/board-full-150-fault-probe-20261011','output/deployment/board-full-150-startup-cause-20261011')
  code=code.replace('from startup_cause_clock_cdc import inventory','from split_logic_outputs_cdc import inventory')
  code=code.replace('observation_only=True','observation_only=False,independent_outputs_workaround=True')
  code=code.replace('完整视频/A系统的启动首因及原语有限对照','完整视频/A系统的独立逻辑输出候选')
  code=code.replace('只新增只读观察：首次非空闲及不相容跳转历史、两时钟域有限原语原始采样；保留DDR四时钟与复位/CKE观察，保持完整视频/A功能。该包用于定位尚未证实的启动根因，不宣称DDR校准已修复。',
   '官方综合选项关闭共享双输出合并，两域有限探针采用等价独立LUT；保持完整视频/A功能RTL、保护IP、频率、引脚与约束。当前网表零COMB4并独立验证功能/物理门禁，保留完整只读观察；实际DDR校准尚待监督板测。')
  return previous(source,code,driver,args)
 original.execute=execute
 prior.main()

if __name__=='__main__':main()
