"""Schema5 current-source release profile; every physical gate is retained."""
from pathlib import Path
source=Path(__file__).with_name('package_video_service.py')
code=source.read_text()
for old,new in [
 ('video-service-host7','video-service-host9'),
 ('video-service-mapped-boot13','video-service-mapped-boot15'),
 ('video-service-owner-boundary4','video-service-owner-boundary5'),
 ('video-service-compact-map-ram-test1','video-service-narrow-mapped-transport1'),
 (" 'video-service-host9':'PASS',",
  " 'video-service-host9':'PASS',\n 'video-service-narrow-mailbox2':'WORD_MAILBOX_ALL_PARTS_STOPPED_CLOCK_LATE_ACK_PASS',"),
 ('diagnostic_schema=4,service_diagnostic_positions=list(range(192))',
  'diagnostic_schema=5,service_diagnostic_positions=list(range(96))'),
 ("('pages-proof.json','nested-epoch-proof.json')",
  "('pages-proof.json','nested-epoch-proof.json','status-selection-proof.json','sweep-counter-proof.json')"),
]:
 assert old in code,old
 code=code.replace(old,new)
needle=" graft=bind(native/'graft.json');bind(native/'native_source.v',graph['source_sha256'])"
assert code.count(needle)==1
code=code.replace(needle,needle+"""
 assert graft['display_extension']['schema_version']==graft['serializer_extension']['schema_version']==5
 assert graft['serializer_extension']['source_word_bits']==96
 assert graft['serializer_extension']['service_chunk_bits']==32
 assert graft['serializer_extension']['legacy_camera_selector']=='service_legacy_epoch[0]'
 bind(HERE/'package_video_service.py')
 for name in ('audit_video_service_narrow_warnings.py','audit_video_service_synthesis.py',
              'video_service_narrow_gray.py','video_service_gray.py',
              'audit_video_service_narrow_jtag.py','audit_video_service_compact_jtag.py',
              'audit_video_service_narrow_split_cdc.py','audit_video_service_compact_routed_cdc.py',
              'audit_video_service_narrow_routed_cdc.py','audit_video_service_routed_cdc.py',
              'video_service_narrow_sample_ram.py','video_service_sample_ram.py'):
  bind(HERE/name)
""")
needle=" assert sha(ram_unit)==sha(board/'video_dualclock_sample.sv')"
assert code.count(needle)==1
code=code.replace(needle,needle+"""
 mailbox=ROOT/'tmp/frame-execution/video-service-narrow-mailbox2'
 for name in ('video_debug_mailbox.sv','video_dualclock_sample.sv'):
  assert sha(mailbox/name)==sha(board/name)
 serial=bind(ROOT/'tmp/frame-execution/video-service-narrow-mapped-fixture1/inputs.json')
 assert serial['files'][str(mapped)]==sha(mapped)
 assert serial['selected']==323 and serial['held_word_input_bits']==list(range(96))
""")
exec(compile(code,str(source)+':schema5-release-profile','exec'),globals())
