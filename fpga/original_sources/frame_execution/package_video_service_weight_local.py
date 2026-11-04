"""Targeted weight-byte distribution, current-map proofs and original gates."""
from pathlib import Path
source = Path(__file__).with_name('package_video_service_data_reset.py')
code = source.read_text()
changes = [
 ('video-service-native42-data-reset-only','video-service-native44-weight-local'),
 ('video-service-mapped-boot21-data-reset','video-service-mapped-boot23-weight-local'),
 ('video-service-owner-boundary8-data-reset','video-service-owner-boundary10-weight-local'),
 ('video-service-data-reset-mapped-transport1','video-service-weight-local-mapped-transport1'),
 ('video-service-data-reset-fixture1','video-service-weight-local-fixture1'),
 ('video-service-data-reset-mapped1','video-service-weight-local-data-reset-mapped1'),
 (" receipt=bind(native/'data-reset-edit.json')", """ assert ng['weight_local_copies']==8 and ng['weight_local_reversible_text']
 weight_parent=Path(ng['weight_local_parent']);assert weight_parent.name=='video-service-native42-data-reset-only'
 bind(native/'weight-local-edit.json')
 wp=bind(board/'weight-local-independent-proof.json')
 assert wp['status']=='ACTUAL_MAPPED_WEIGHT_BYTE_SPLIT_EQUIVALENCE_PASS'
 assert wp['map_sha256']==sha(mapped) and wp['latency_added']==0 and len(wp['copies'])==8
 file_map(wp)
 receipt=bind(weight_parent/'data-reset-edit.json')"""),
 ("'validate_video_service_data_reset_mapped.py','audit_video_service_mapped_boot_warnings.py'",
  "'validate_video_service_data_reset_mapped.py','validate_video_service_weight_data_reset.py','prepare_video_service_weight_local.py','audit_video_service_weight_local.py','audit_video_service_mapped_boot_warnings.py'"),
]
for old, new in changes:
 assert old in code, old
 code = code.replace(old, new)
anchor = "for old,new in changes:"
assert code.count(anchor) == 1
code = code.replace(anchor, "changes.append((\"assert ng['retained_primitives']==54577\",\"assert ng['retained_primitives']==54585\"))\n"+anchor)
exec(compile(code, str(source)+':weight-local-profile', 'exec'), globals())
