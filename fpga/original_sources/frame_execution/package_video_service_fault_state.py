"""Fault-state candidate profile: retain full physical and numerical gates."""
from pathlib import Path
source=Path(__file__).with_name('package_video_service_rq.py');code=source.read_text()
changes=[
 ('video-service-native37-rq-local','video-service-native43-fault-state'),
 ("assert ng['retained_primitives']==54577","assert ng['retained_primitives']==54579"),
 ('video-service-mapped-boot19-rq-local','video-service-mapped-boot22-fault-state'),
 ('video-service-owner-boundary6-rq-local','video-service-owner-boundary9-fault-state'),
 ('video-service-rq-local-mapped-transport1','video-service-fault-state-mapped-transport1'),
 ('video-service-rq-local-mapped-fixture1','video-service-fault-state-fixture1'),
 (" bind(native/'rq-local-edit.json')",''' assert ng['fault_state_copies']==2 and ng['fault_state_reversible_text']
 parent=Path(ng['fault_state_parent']);assert parent.name=='video-service-native37-rq-local'
 bind(parent/'rq-local-edit.json');bind(native/'fault-state-edit.json')
 fs=bind(board/'fault-state-independent-proof.json');assert fs['status']=='ACTUAL_MAPPED_FAULT_STATE_EQUIVALENT_COPIES_PASS'
 assert fs['map_sha256']==sha(mapped) and fs['latency_added']==0 and len(fs['copies'])==2
 file_map(fs)
 bw=bind(ROOT/'tmp/frame-execution/video-service-mapped-boot22-fault-state/warning-port-audit.json')
 assert bw['status']=='EVERY_MISSING_PORT_UNUSED_OUTPUT_OR_BOOLEAN_INDEPENDENT_INPUT_PASS';file_map(bw)
 for name in ('prepare_video_service_fault_state_local.py','audit_video_service_fault_state.py',
              'audit_video_service_mapped_boot_warnings.py','audit_video_service_protected_warnings.py',
              'video_service_ram_timing_pinmap.py','run_video_service_checked_questa.py',
              'map_video_service_safe.py','windows_process_tree_safe.py','package_video_service_rq.py'):
  bind(HERE/name)
''')]
for old,new in changes:
 assert old in code,old
 code=code.replace(old,new)
exec(compile(code,str(source)+':fault-state-profile','exec'),globals())
