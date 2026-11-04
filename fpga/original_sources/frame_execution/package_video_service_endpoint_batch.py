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
 ('video-service-native42-data-reset-only','video-service-native58-endpoint-batch'),
 ('video-service-mapped-boot21-data-reset','video-service-mapped-boot35-endpoint-batch'),
 ('video-service-owner-boundary8-data-reset','video-service-owner-boundary24-endpoint-batch'),
 ('video-service-data-reset-mapped-transport1','video-service-endpoint-batch-mapped-transport1'),
 ('video-service-data-reset-fixture1','video-service-endpoint-batch-fixture1'),
 ('video-service-data-reset-mapped1','video-service-endpoint-batch-data-reset-mapped1'),
 (" receipt=bind(native/'data-reset-edit.json')", """ assert ng['weight_zero_removed']==246 and ng['weight_zero_guard_luts']==91
 weight_parent=Path(ng['weight_zero_parent']);assert weight_parent.name=='video-service-native42-data-reset-only'
 edit=bind(native/'weight-zero-edit.json');assert edit['latency_added']==0
 assert Path(edit['parent'])==weight_parent and edit['source_sha256']==sha(weight_parent/'native_source.v')
 independent=bind(native/'weight-zero-independent-audit.json')
 assert independent['status']=='GUARDED_A_ZERO_EXACT_SOURCE_GRAPH_PASS_NOT_PHYSICAL'
 assert independent['retained']==54331 and independent['removed']==246 and independent['added']==91
 assert independent['guard_boolean_cases']==17788 and independent['ce_cone_truth_checks']==1024 and independent['selector_truth_checks']==32768 and independent['data_and_sets_exact'] and not independent['only_error_d_changed'] and not independent['other_controls_and_all_arithmetic_unchanged'] and independent['weight_cache_unchanged_from_parent']
 assert independent['source_sha256']==sha(native/'native_source.v');file_map(independent['inputs'])
 ep=bind(native/'endpoint-batch-independent-audit-v2.json')
 assert ep['status']=='ENDPOINT_BATCH_FULL_SOURCE_INDUCTIVE_PASS_NOT_PHYSICAL' and ep['source_sha256']==sha(native/'native_source.v')
 assert ep['primitive_count']==53868 and ep['induction_rows']==512 and ep['base_case_zero'] and ep['latency_added']==0 and ep['all_original_state_primitives_exact']
 file_map(ep['inputs']);bind(native/'endpoint-batch-edit.json')
 parent56=ROOT/'tmp/frame-execution/video-service-native56-global-targeted'
 gp=bind(parent56/'global-batch-independent-audit.json');assert gp['status']=='GLOBAL_BATCH_INDEPENDENT_FULL_SOURCE_PASS_NOT_PHYSICAL' and gp['source_sha256']==ep['parent_sha256']
 file_map(gp['inputs'])
 er=bind(native/'endpoint-ready-independent-audit.json');assert er['status']=='ENDPOINT_EXPANDED_READY_BDD_PASS' and er['current_bdd_proof']['Boolean_complete'] and er['source_sha256']==sha(native/'native_source.v');file_map(er['inputs'])
 for suffix,status in [('source-control2','ACTUAL_ENDPOINT_EXPANDED_READY_PASS'),('mapped-control1','ACTUAL_ENDPOINT_EXPANDED_READY_PASS'),('source-cones1','ACTUAL_ENDPOINT_CONTROL_CONES_PASS'),('mapped-cones2','ACTUAL_ENDPOINT_CONTROL_CONES_PASS'),('projection-source','ACTUAL_FETCH_PROJECTION_SAME_EDGE_PASS'),('projection-mapped2','ACTUAL_FETCH_PROJECTION_SAME_EDGE_PASS')]:
  tr=ROOT/'tmp/frame-execution'/('video-service-endpoint-batch-'+suffix);rr=bind(tr/'result.json');assert rr['status']==status and rr['source_sha256']==sha(native/'native_source.v')
  file_map(bind(tr/'inputs.json',rr['inputs_sha256']))
  for logfile in tr.glob('*.log'):bind(logfile)
  if rr['mapped']:assert rr['map_sha256']==sha(mapped)
 ob=bind(ROOT/'tmp/frame-execution/video-service-owner-boundary24-endpoint-batch/current-binding.json')
 assert ob['status']=='CURRENT_PROVED_NATIVE_PORT_BINDING' and ob['map_sha256']==sha(mapped) and ob['source_sha256']==sha(native/'native_source.v')
 bind(HERE/'validate_video_service_owner_boundary_v2.py',ob['checker_sha256']);bind(HERE/'validate_video_service_owner_boundary.py',ob['parent_checker_sha256'])
 sp=bind(board/'status-selection-proof.json')
 assert sp['front_bank_alias_proof']['reset_inductive_state_alias'] and sp['front_bank_alias_proof']['all_non_output_pins_identical'] and sp['front_bank_alias_proof']['all_parameters_identical']
 bind(HERE/'validate_video_service_narrow_status_selection_v2.py',sp['checker_sha256']);file_map(sp['inputs'])
 for script in ('validate_video_service_endpoint_ready_v2.py','validate_video_service_endpoint_cones_v2.py','validate_video_service_fetch_projection_v2.py','bind_video_service_mapped_cut.py','validate_video_service_endpoint_ready.py','validate_video_service_endpoint_cones.py','validate_video_service_fetch_projection.py','validate_video_service_fault_ready.py','validate_video_service_fault_ready_v2.py','validate_video_service_global_cones.py','package_video_service_fault_ready.py'):bind(HERE/script)

 inventory=bind(native/'a-zero-inventory.json');assert inventory['status']=='ACTUAL_A_PACKED_LOADER_ZERO_BITS_VERIFIED'
 assert inventory['byte_order_checks']==11232 and inventory['removed_storage']==246
 for label in ('video-service-endpoint-batch-source1','video-service-endpoint-batch-mapped1'):
  wr=ROOT/'tmp/frame-execution'/label;wt=bind(wr/'result.json');assert wt['status']=='ACTUAL_A_ZERO_CACHE_GUARD_PASS'
  assert wt['source_sha256']==sha(native/'native_source.v') and wt['valid_words']==11232
  file_map(bind(wr/'inputs.json',wt['inputs_sha256']))
  for logfile in wr.glob('*.log'):bind(logfile)
  if label.endswith('mapped1'):assert wt['mapped'] and wt['map_sha256']==sha(mapped)
 ow=bind(ROOT/'tmp/frame-execution/video-service-owner-boundary24-endpoint-batch/warning-port-audit.json')
 assert ow['status']=='EVERY_MISSING_PORT_UNUSED_OUTPUT_OR_BOOLEAN_INDEPENDENT_INPUT_PASS';file_map(ow['evidence'])
 for script in ('validate_video_service_weight_zero_factored.py','validate_video_service_weight_zero_data_reset.py','validate_video_service_weight_data_reset.py','audit_video_service_weight_zero_map.py','audit_video_service_weight_zero_map_v2.py','package_video_service_data_reset.py'):
  bind(HERE/script)
 receipt=bind(weight_parent/'data-reset-edit.json')"""),
]
for old,new in changes:
 assert old in code,old
 code=code.replace(old,new)
needle='for old,new in changes:';assert code.count(needle)==1
code=code.replace(needle,'changes.append(("assert ng[\'retained_primitives\']==54577", "assert ng[\'retained_primitives\']==53868"))\n'+needle)
exec(compile(code,str(source)+':guarded-a-zero-profile','exec'),globals())
