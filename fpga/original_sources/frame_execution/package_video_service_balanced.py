"""Frozen balanced-ready implementation with current-map release evidence.

The existing numerical, physical, timing and supervised-board gates remain.
Only the selected candidate identity and its actual-map proofs change.
"""
from pathlib import Path

source = Path(__file__).with_name('package_video_service_rq.py')
code = source.read_text()
changes = [
 ('video-service-native37-rq-local', 'video-service-native39-ready-balanced'),
 ("ng['retained_primitives']==54577", "ng['retained_primitives']==54580"),
 ('video-service-mapped-boot19-rq-local', 'video-service-mapped-boot20-ready-balanced'),
 ('video-service-owner-boundary6-rq-local', 'video-service-owner-boundary7-ready-balanced'),
 ('video-service-rq-local-mapped-transport1', 'video-service-ready-balanced-mapped-transport1'),
 ('video-service-rq-local-mapped-fixture1', 'video-service-ready-balanced-mapped-fixture1'),
 (" bind(native/'rq-local-edit.json')", """ assert ng['ready_local_clones']==3 and ng['ready_local_four_state_comparisons']==576
 assert ng['ready_local_reversible_text'] and ng['ready_balanced_reversible_text']
 assert ng['ready_balanced_checks']==16384
 bind(Path(ng['ready_local_parent'])/'rq-local-edit.json')
 bind(Path(ng['ready_balanced_parent'])/'ready-local-edit.json')
 balanced=bind(native/'ready-balanced-edit.json')
 assert balanced['changed_state']==balanced['latency_added']==0
 for name in ('prepare_video_service_ready_local.py','prepare_video_service_ready_balanced.py',
              'video_service_ram_timing_pinmap.py','run_video_service_checked_questa.py',
              'map_video_service_safe.py','windows_process_tree_safe.py','package_video_service_rq.py'):
  bind(HERE/name)
"""),
]
for old,new in changes:
 assert old in code,old
 code=code.replace(old,new)
exec(compile(code,str(source)+':balanced-ready-release-profile','exec'),globals())
