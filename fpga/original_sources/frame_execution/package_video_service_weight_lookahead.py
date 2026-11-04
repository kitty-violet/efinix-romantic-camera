"""Same full release gates with an explicit exact-address-retiming profile."""
from pathlib import Path
source=Path(__file__).with_name('package_video_service_data_reset.py');code=source.read_text()
changes=[
 ('video-service-native42-data-reset-only','video-service-native47-weight-lookahead'),
 ('video-service-mapped-boot21-data-reset','video-service-mapped-boot26-weight-lookahead'),
 ('video-service-owner-boundary8-data-reset','video-service-owner-boundary14-weight-lookahead'),
 ('video-service-data-reset-mapped-transport1','video-service-weight-lookahead-mapped-transport1'),
 ('video-service-data-reset-fixture1','video-service-weight-lookahead-fixture1'),
 ('video-service-data-reset-mapped1','video-service-weight-lookahead-data-reset-mapped1'),
 (" receipt=bind(native/'data-reset-edit.json')", """ assert ng['weight_lookahead_registers']==24 and ng['weight_lookahead_added']==147 and ng['weight_lookahead_removed']==89
 weight_parent=Path(ng['weight_lookahead_parent']);assert weight_parent.name=='video-service-native42-data-reset-only'
 edit=bind(native/'weight-lookahead-edit.json');assert edit['latency_added']==0 and edit['all_original_ff_and_srl_state_preserved']
 assert Path(edit['parent'])==weight_parent and edit['source_sha256']==sha(weight_parent/'native_source.v')
 independent=bind(native/'weight-lookahead-independent-audit.json')
 assert independent['status']=='WEIGHT_LOOKAHEAD_EXACT_SOURCE_GRAPH_PASS_NOT_PHYSICAL'
 assert independent['retained']==54488 and independent['removed']==89 and independent['added']==147
 assert independent['changed_srl_addresses']==512 and independent['next_state_boolean_cases']==464
 assert independent['source_sha256']==sha(native/'native_source.v') and independent['known_synchronous_controls_required']
 file_map(independent['inputs'])
 for label,status in [('video-service-weight-lookahead-short2','WEIGHT_ADDRESS_NEXT_STATE_VENDOR_SHORT_PASS_NOT_INTEGRATED'),
                      ('video-service-weight-lookahead-mapped1','ACTUAL_MAPPED_WEIGHT_LOOKAHEAD_CACHE_PASS')]:
  wr=ROOT/'tmp/frame-execution'/label;wt=bind(wr/'result.json');assert wt['status']==status
  file_map(bind(wr/'inputs.json',wt['inputs_sha256']))
  for logfile in wr.glob('*.log'):bind(logfile)
  if label.endswith('mapped1'):
   assert wt['map_sha256']==sha(mapped) and wt['storage']==512 and wt['checks']==8388608
 ow=bind(ROOT/'tmp/frame-execution/video-service-owner-boundary14-weight-lookahead/warning-port-audit.json')
 assert ow['status']=='EVERY_MISSING_PORT_UNUSED_OUTPUT_OR_BOOLEAN_INDEPENDENT_INPUT_PASS';file_map(ow['evidence'])
 for script in ('validate_video_service_weight_lookahead.py','validate_video_service_weight_data_reset.py','package_video_service_data_reset.py'):
  bind(HERE/script)
 receipt=bind(weight_parent/'data-reset-edit.json')"""),
]
for old,new in changes:
 assert old in code,old
 code=code.replace(old,new)
# The original RQ identity gate still applies, but this new graph has 58
# more primitives. Change only its exact candidate count, never the gate.
needle='for old,new in changes:'
assert code.count(needle)==1
code=code.replace(needle,'changes.append(("assert ng[\'retained_primitives\']==54577", "assert ng[\'retained_primitives\']==54635"))\n'+needle)
exec(compile(code,str(source)+':weight-lookahead-profile','exec'),globals())
