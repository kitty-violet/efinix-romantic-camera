"""Guarded A-zero source, numerical/cache checks and unchanged full release gates."""
from pathlib import Path
source=Path(__file__).with_name('package_video_service_data_reset.py');code=source.read_text()
# Push the scope into the final README and manifest before their hashes and
# the programmer binding are made. Parent release scripts remain unchanged.
scope_changes=[
 ('- A系数、compute、DMA为已验收集成包；包内*.bin用于硬件装载，不用.pt文件。',
  '- A系数、compute、DMA为已验收集成包；包内*.bin用于硬件装载，不用.pt文件。\n- 本实现按当前A系数的恒零位精简缓存；不改变A系数或B1算术。装入不满足零位约束的系数将触发既有werror并停止，不能据此宣称任意其他权重兼容。'),
 ('native=str(native),board=str(board),',
  "coefficient_policy=dict(id='guarded-A-zero-v1',package_sha256=inventory['package_sha256'],guarded_zero_masks=inventory['guard_masks'],unsupported_nonzero='sticky cache error, compute inhibited'),native=str(native),board=str(board),"),
]
anchor="source=Path(__file__).with_name('package_video_service_rq.py');code=source.read_text()"
assert code.count(anchor)==1
insertion='changes = [\n'+''.join(' '+repr(pair)+',\n' for pair in scope_changes)
code=code.replace(anchor,anchor+'\nassert code.count("changes = [")==1\ncode=code.replace("changes = [",'+repr(insertion)+')')
changes=[
 ('video-service-native42-data-reset-only','video-service-native49-weight-zero'),
 ('video-service-mapped-boot21-data-reset','video-service-mapped-boot28-weight-zero'),
 ('video-service-owner-boundary8-data-reset','video-service-owner-boundary16-weight-zero'),
 ('video-service-data-reset-mapped-transport1','video-service-weight-zero-mapped-transport1'),
 ('video-service-data-reset-fixture1','video-service-weight-zero-fixture1'),
 ('video-service-data-reset-mapped1','video-service-weight-zero-data-reset-mapped1'),
 (" receipt=bind(native/'data-reset-edit.json')", """ assert ng['weight_zero_removed']==246 and ng['weight_zero_guard_luts']==93
 weight_parent=Path(ng['weight_zero_parent']);assert weight_parent.name=='video-service-native42-data-reset-only'
 edit=bind(native/'weight-zero-edit.json');assert edit['latency_added']==0
 assert Path(edit['parent'])==weight_parent and edit['source_sha256']==sha(weight_parent/'native_source.v')
 independent=bind(native/'weight-zero-independent-audit.json')
 assert independent['status']=='GUARDED_A_ZERO_EXACT_SOURCE_GRAPH_PASS_NOT_PHYSICAL'
 assert independent['retained']==54331 and independent['removed']==246 and independent['added']==93
 assert independent['guard_boolean_cases']==11644 and independent['only_error_d_changed'] and independent['other_controls_and_all_arithmetic_unchanged']
 assert independent['source_sha256']==sha(native/'native_source.v');file_map(independent['inputs'])
 inventory=bind(native/'a-zero-inventory.json');assert inventory['status']=='ACTUAL_A_PACKED_LOADER_ZERO_BITS_VERIFIED'
 assert inventory['byte_order_checks']==11232 and inventory['removed_storage']==246
 for label in ('video-service-weight-zero-source1','video-service-weight-zero-mapped1'):
  wr=ROOT/'tmp/frame-execution'/label;wt=bind(wr/'result.json');assert wt['status']=='ACTUAL_A_ZERO_CACHE_GUARD_PASS'
  assert wt['source_sha256']==sha(native/'native_source.v') and wt['valid_words']==11232
  file_map(bind(wr/'inputs.json',wt['inputs_sha256']))
  for logfile in wr.glob('*.log'):bind(logfile)
  if label.endswith('mapped1'):assert wt['mapped'] and wt['map_sha256']==sha(mapped)
 ow=bind(ROOT/'tmp/frame-execution/video-service-owner-boundary16-weight-zero/warning-port-audit.json')
 assert ow['status']=='EVERY_MISSING_PORT_UNUSED_OUTPUT_OR_BOOLEAN_INDEPENDENT_INPUT_PASS';file_map(ow['evidence'])
 for script in ('validate_video_service_weight_zero.py','validate_video_service_weight_zero_data_reset.py','validate_video_service_weight_data_reset.py','audit_video_service_weight_zero_map.py','audit_video_service_weight_zero_map_v2.py','package_video_service_data_reset.py'):
  bind(HERE/script)
 receipt=bind(weight_parent/'data-reset-edit.json')"""),
]
for old,new in changes:
 assert old in code,old
 code=code.replace(old,new)
needle='for old,new in changes:';assert code.count(needle)==1
code=code.replace(needle,'changes.append(("assert ng[\'retained_primitives\']==54577", "assert ng[\'retained_primitives\']==54424"))\n'+needle)
exec(compile(code,str(source)+':guarded-a-zero-profile','exec'),globals())
