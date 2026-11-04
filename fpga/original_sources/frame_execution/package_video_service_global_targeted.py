"""Combined global-control/area batch; preserve every original release gate."""
from pathlib import Path
source=Path(__file__).with_name('package_video_service_fault_ready.py');code=source.read_text(encoding='utf8')
for a,b in [
 ('video-service-native51-fault-ready','video-service-native56-global-targeted'),
 ('video-service-mapped-boot30-fault-ready','video-service-mapped-boot34-global-targeted'),
 ('video-service-owner-boundary18-fault-ready','video-service-owner-boundary23-global-targeted'),
 ('video-service-fault-ready','video-service-global-targeted'),
 ('mapped-control2','mapped-control1'),('1048576','2097152'),('==54430','==53847')]:
 assert a in code,a;code=code.replace(a,b)
needle=" bind(HERE/'validate_video_service_fault_ready.py');bind(HERE/'validate_video_service_fault_ready_v2.py')"
extra="""
 gb=bind(native/'global-batch-independent-audit.json')
 assert gb['status']=='GLOBAL_BATCH_INDEPENDENT_FULL_SOURCE_PASS_NOT_PHYSICAL'
 assert gb['source_sha256']==sha(native/'native_source.v') and gb['primitive_count']==53847
 assert gb['four_state_fusions']==612 and gb['select_replicas']==5 and gb['all_non_lut_primitives_exact'] and gb['latency_added']==0
 file_map(gb['inputs']);bind(native/'global-batch-edit.json')
 ob=bind(ROOT/'tmp/frame-execution/video-service-owner-boundary23-global-targeted/current-binding.json')
 assert ob['status']=='CURRENT_PROVED_NATIVE_PORT_BINDING' and ob['map_sha256']==sha(mapped) and ob['source_sha256']==sha(native/'native_source.v')
 bind(HERE/'validate_video_service_owner_boundary_v2.py',ob['checker_sha256']);bind(HERE/'validate_video_service_owner_boundary.py',ob['parent_checker_sha256'])
 sp=bind(board/'status-selection-proof.json')
 assert sp['front_bank_alias_proof']['reset_inductive_state_alias'] and sp['front_bank_alias_proof']['all_non_output_pins_identical'] and sp['front_bank_alias_proof']['all_parameters_identical']
 bind(HERE/'validate_video_service_narrow_status_selection_v2.py',sp['checker_sha256']);file_map(sp['inputs'])
 for label in ('video-service-global-targeted-source-cones1','video-service-global-targeted-mapped-cones1'):
  tr=ROOT/'tmp/frame-execution'/label;cr=bind(tr/'result.json')
  assert cr['status']=='ACTUAL_GLOBAL_CONTROL_CONES_PASS' and cr['source_sha256']==sha(native/'native_source.v')
  assert cr['groups']==5 and cr['vectors_per_group']==393282
  file_map(bind(tr/'inputs.json',cr['inputs_sha256']))
  for logfile in tr.glob('*.log'):bind(logfile)
  if cr['mapped']:assert cr['map_sha256']==sha(mapped)
 for script in ('validate_video_service_global_cones.py','filter_video_service_batch_fusions.py','validate_video_service_terminal_ready.py','package_video_service_fault_ready.py'):bind(HERE/script)
"""
assert needle in code;code=code.replace(needle,needle+extra)
exec(compile(code,str(source)+':global-combined-batch','exec'),globals())
