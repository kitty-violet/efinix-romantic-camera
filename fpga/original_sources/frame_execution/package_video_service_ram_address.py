"""Release profile for 45 address compositions; all original gates retained."""
from pathlib import Path
source=Path(__file__).with_name('package_video_service_cache_lut.py');code=source.read_text()
changes=[
 ('video-service-native45-cache-lut','video-service-native46-ram-address'),
 ('video-service-mapped-boot24-cache-lut','video-service-mapped-boot25-ram-address'),
 ('video-service-owner-boundary11-cache-lut','video-service-owner-boundary13-ram-address'),
 ('video-service-cache-lut-mapped-transport1','video-service-ram-address-mapped-transport1'),
 ('video-service-cache-lut-fixture1','video-service-ram-address-fixture1'),
 ('video-service-cache-lut-data-reset-mapped1','video-service-ram-address-data-reset-mapped1'),
 (" edit=bind(native/'cache-lut-edit.json')", """ assert ng['ram_address_bits']==45 and ng['ram_address_reversible_text']
 address=bind(native/'ram-address-edit.json')
 assert address['changed_address_bits']==45 and address['lut_count_delta']==address['ff_count_delta']==address['latency_added']==0
 bind(address['survey'],address['survey_sha256'])
 mapped_address=bind(board/'ram-address-map-proof.json')
 assert mapped_address['status']=='ACTUAL_45_RAM_ADDRESS_STAGES_AND_FANOUT_CHECKED_PASS' and mapped_address['address_bits']==45 and mapped_address['map_sha256']==sha(mapped)
 file_map(mapped_address['evidence'])
 ow=bind(ROOT/'tmp/frame-execution/video-service-owner-boundary13-ram-address/warning-port-audit.json')
 assert ow['status']=='EVERY_MISSING_PORT_UNUSED_OUTPUT_OR_BOOLEAN_INDEPENDENT_INPUT_PASS';file_map(ow['evidence'])
 rt=ROOT/'tmp/frame-execution/video-service-ram-address-short1'
 ar=bind(rt/'result.json');assert ar['status']=='EXACT_RAM_ADDRESS_VENDOR_FOUR_STATE_PASS' and ar['checks']==46080 and ar['address_bits']==45
 file_map(bind(rt/'inputs.json',ar['inputs_sha256']))
 for logfile in rt.glob('*.log'):bind(logfile)
 for name in ('prepare_video_service_ram_address.py','validate_video_service_ram_address.py','audit_video_service_ram_address_map.py','rerun_video_service_owner.py','audit_video_service_owner_warnings.py','package_video_service_cache_lut.py','package_video_service_data_reset.py'):
  bind(HERE/name)
 edit=bind(native/'cache-lut-edit.json')"""),
]
for old,new in changes:
 assert old in code,old
 code=code.replace(old,new)
exec(compile(code,str(source)+':ram-address-profile','exec'),globals())
