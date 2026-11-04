"""Selected data-reset candidate with qualified-data proofs and strict gates."""
from pathlib import Path
source=Path(__file__).with_name('package_video_service_rq.py');code=source.read_text()
changes=[
 ('video-service-native37-rq-local','video-service-native42-data-reset-only'),
 ('video-service-mapped-boot19-rq-local','video-service-mapped-boot21-data-reset'),
 ('video-service-owner-boundary6-rq-local','video-service-owner-boundary8-data-reset'),
 ('video-service-rq-local-mapped-transport1','video-service-data-reset-mapped-transport1'),
 ('video-service-rq-local-mapped-fixture1','video-service-data-reset-fixture1'),
 (" bind(native/'rq-local-edit.json')",''' assert ng['data_reset_bits']==512 and ng['data_reset_reversible_text']
 assert ng['data_reset_semantics']=='valid-qualified; invalid data intentionally unspecified'
 assert ng['data_reset_induction_cases']==504
 parent=Path(ng['data_reset_parent']);assert parent.name=='video-service-native37-rq-local'
 bind(parent/'rq-local-edit.json')
 receipt=bind(native/'data-reset-edit.json')
 assert receipt['data_bits']==512 and receipt['control_bits_changed']==receipt['latency_added']==0
 assert receipt['active_low_SR_inactive_value']==1
 for label,status in [('video-service-data-reset-short3','DATA_ONLY_RESET_PROTOCOL_AND_NUMERIC_SHORT_PASS_NOT_NATIVE_PROOF'),
                      ('video-service-data-reset-mapped1','ACTUAL_MAPPED_DATA_FF_QUALIFIED_RESET_PASS')]:
  testroot=ROOT/'tmp/frame-execution'/label;test=bind(testroot/'result.json');assert test['status']==status
  file_map(bind(testroot/'inputs.json',test['inputs_sha256']))
  for logfile in testroot.glob('*.log'):bind(logfile)
  if label.endswith('mapped1'):assert test['map_sha256']==sha(mapped)
 bw=bind(ROOT/'tmp/frame-execution/video-service-mapped-boot21-data-reset/warning-port-audit.json')
 assert bw['status']=='EVERY_MISSING_PORT_UNUSED_OUTPUT_OR_BOOLEAN_INDEPENDENT_INPUT_PASS';file_map(bw)
 for name in ('prepare_video_service_data_reset.py','validate_video_service_data_reset.py',
              'validate_video_service_data_reset_mapped.py','audit_video_service_mapped_boot_warnings.py',
              'audit_video_service_protected_warnings.py','video_service_ram_timing_pinmap.py',
              'run_video_service_checked_questa.py','map_video_service_safe.py','windows_process_tree_safe.py','package_video_service_rq.py'):
  bind(HERE/name)
'''),
]
for old,new in changes:
 assert old in code,old
 code=code.replace(old,new)
exec(compile(code,str(source)+':data-reset-proof-profile','exec'),globals())
