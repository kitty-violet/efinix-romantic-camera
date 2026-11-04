"""RQ-specialized implementation: same physical gates, current-map proofs.

This profile only accepts the reviewed local-control candidate and its fixed
A payload. It does not extend the arithmetic policy to arbitrary small MAC
scales, and it never grants permission to access hardware.
"""
from pathlib import Path

source = Path(__file__).with_name('package_video_service_narrow.py')
code = source.read_text()
needle = "exec(compile(code,str(source)+':schema5-release-profile','exec'),globals())"
assert code.count(needle) == 1
changes = [
 ('from place_video_service_epoch import audit_delta', 'from place_video_service_rq import audit_delta'),
 ('video-service-mapped-boot15', 'video-service-mapped-boot19-rq-local'),
 ('video-service-owner-boundary5', 'video-service-owner-boundary6-rq-local'),
 ('video-service-narrow-mapped-transport1', 'video-service-rq-local-mapped-transport1'),
 ('video-service-narrow-mapped-fixture1', 'video-service-rq-local-mapped-fixture1'),
 (" 'video-service-host9':'PASS',", " 'video-service-host9':'PASS',\n 'video-service-rq-fast-short1':'GUARDED_FAST_RQ_INT32_NUMERIC_QUEUE_CANCEL_PASS',"),
]
extra = """
 assert native.name=='video-service-native37-rq-local'
 assert ng['retained_primitives']==54577
 assert ng['scope']=='Guarded RQ policy specialization; NOT unrestricted old low-scale equivalence'
 for name in ('audit_video_service_rq_map.py','audit_video_service_rq_warnings.py',
              'audit_video_service_rq_fast.py','audit_video_service_rq_local.py',
              'prepare_video_service_rq_fast.py','prepare_video_service_rq_local.py',
              'place_video_service_rq.py','validate_video_service_rq_fast.py',
              'package_video_service_narrow.py'):
  bind(HERE/name)
 # The local proof binds the complete parent graph, exact clone/load edits,
 # and all native SV files; file_map checks every referenced input hash.
 assert ng['local_reversible_text_proof'] and ng['local_clones']==16 and ng['local_loads']==240
 assert ng['local_four_state_comparisons']==64 and ng['latency_added']==0
 bind(native/'rq-local-edit.json')
 accepted=ROOT/'model-training-workspace/accepted/B1-A-20261008/coefficients-padded.bin'
 bind(accepted,'ceebe7b3187fd47fbf0924883d19a9a335923d9de378ae09d30f362d9b9d0523')
"""
patch = ''
for old, new in changes:
 patch += 'assert '+repr(old)+' in code\ncode=code.replace('+repr(old)+','+repr(new)+')\n'
anchor = " assert graft['display_extension']['schema_version']==graft['serializer_extension']['schema_version']==5"
patch += 'assert code.count('+repr(anchor)+')==1\ncode=code.replace('+repr(anchor)+','+repr(extra+anchor)+')\n'
code = code.replace(needle, patch+needle)
exec(compile(code, str(source)+':rq-current-proof-profile', 'exec'), globals())
