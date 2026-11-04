"""Cache-address simplification with exact current evidence and full gates."""
from pathlib import Path
source=Path(__file__).with_name('package_video_service_data_reset.py');code=source.read_text()
changes=[
 ('video-service-native42-data-reset-only','video-service-native45-cache-lut'),
 ('video-service-mapped-boot21-data-reset','video-service-mapped-boot24-cache-lut'),
 ('video-service-owner-boundary8-data-reset','video-service-owner-boundary11-cache-lut'),
 ('video-service-data-reset-mapped-transport1','video-service-cache-lut-mapped-transport1'),
 ('video-service-data-reset-fixture1','video-service-cache-lut-fixture1'),
 ('video-service-data-reset-mapped1','video-service-cache-lut-data-reset-mapped1'),
 (" receipt=bind(native/'data-reset-edit.json')", """ assert ng['cache_lut_four_state_cases']==64 and ng['cache_lut_reversible_text']
 cache_parent=Path(ng['cache_lut_parent']);assert cache_parent.name=='video-service-native42-data-reset-only'
 edit=bind(native/'cache-lut-edit.json');assert edit['lut_count_delta']==edit['ff_count_delta']==edit['latency_added']==0
 ct=ROOT/'tmp/frame-execution/video-service-cache-lut-short1'
 ctest=bind(ct/'result.json');assert ctest['status']=='EXACT_CACHE_LUT_VENDOR_FOUR_STATE_PASS' and ctest['cases']==256
 file_map(bind(ct/'inputs.json',ctest['inputs_sha256']))
 for logfile in ct.glob('*.log'):bind(logfile)
 receipt=bind(cache_parent/'data-reset-edit.json')"""),
 ("'validate_video_service_data_reset_mapped.py','audit_video_service_mapped_boot_warnings.py'",
  "'validate_video_service_data_reset_mapped.py','validate_video_service_cache_data_reset.py','prepare_video_service_cache_lut.py','validate_video_service_cache_lut.py','audit_video_service_mapped_boot_warnings.py'"),
]
for old,new in changes:
 assert old in code,old
 code=code.replace(old,new)
exec(compile(code,str(source)+':cache-lut-profile','exec'),globals())
