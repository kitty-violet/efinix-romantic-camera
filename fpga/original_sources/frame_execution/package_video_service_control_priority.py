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
code=code.replace(anchor,anchor+".replace('from place_video_service_rq import audit_delta','from place_video_service_control_priority import audit_delta')"+'\nassert code.count("changes = [")==1\ncode=code.replace("changes = [",'+repr(insertion)+')')
changes=[
 ('video-service-native42-data-reset-only','video-service-native61-control-priority'),
 ('video-service-mapped-boot21-data-reset','video-service-mapped-boot36-control-priority'),
 ('video-service-owner-boundary8-data-reset','video-service-owner-boundary25-control-priority'),
 ('video-service-data-reset-mapped-transport1','video-service-control-priority-mapped-transport1'),
 ('video-service-data-reset-fixture1','video-service-control-priority-fixture1'),
 ('video-service-data-reset-mapped1','video-service-control-priority-data-reset-mapped1'),
 (" receipt=bind(native/'data-reset-edit.json')", """ assert ng['weight_zero_removed']==246 and ng['weight_zero_guard_luts']==91
 weight_parent=Path(ng['weight_zero_parent']);assert weight_parent.name=='video-service-native42-data-reset-only'
 edit=bind(native/'weight-zero-edit.json');assert edit['latency_added']==0
 assert Path(edit['parent'])==weight_parent and edit['source_sha256']==sha(weight_parent/'native_source.v')
 independent=bind(native/'weight-zero-independent-audit.json')
 assert independent['status']=='GUARDED_A_ZERO_EXACT_SOURCE_GRAPH_PASS_NOT_PHYSICAL'
 assert independent['retained']==54331 and independent['removed']==246 and independent['added']==91
 assert independent['guard_boolean_cases']==17788 and independent['ce_cone_truth_checks']==1024 and independent['selector_truth_checks']==32768 and independent['data_and_sets_exact'] and not independent['only_error_d_changed'] and not independent['other_controls_and_all_arithmetic_unchanged'] and independent['control_remap_proof']==str(native/'control-remap-independent-audit.json')
 assert independent['source_sha256']==sha(native/'native_source.v');file_map(independent['inputs'])
 rp=bind(native/'control-remap-independent-audit.json')
 assert rp['status']=='CONTROL_REMAP_FULL_SOURCE_CEC_PASS_NOT_PHYSICAL' and rp['source_sha256']==sha(native/'native_source.v')
 assert rp['cec_all_boolean_inputs'] and rp['tampered_output_rejected'] and rp['all_outside_cut_and_sequential_state_exact'] and rp['all_external_consumers_retained'] and rp['latency_added']==0
 file_map(rp['inputs']);bind(native/'control-remap-edit.json')
 prior=ROOT/'tmp/frame-execution/video-service-native58-endpoint-batch'
 assert rp['parent_sha256']==sha(prior/'native_source.v')
 historical=bind(ROOT/'tmp/board-test/video-service-review1/current-functional-endpoint-batch-1.json')
 assert historical['status']=='CURRENT_SERVICE_FUNCTIONAL_RELEASE_INPUTS_CHECKED_NOT_PHYSICAL_RELEASE'
 file_map(historical['evidence'])
 for suffix in ('source6','mapped8'):
  tr=ROOT/'tmp/frame-execution'/('video-service-control-priority-'+suffix);rr=bind(tr/'result.json')
  assert rr['status']=='ACTUAL_VENDOR_CONTROL_REMAP_PASS' and rr['source_sha256']==sha(native/'native_source.v')
  assert rr['outputs']==rp['boundary_outputs'] and rr['vectors']==2+2*rp['boundary_inputs']+256 and rr['walking_all_inputs'] and rr['primitive_four_state_cases']==256 and rr['binary_CEC_complete']
  file_map(bind(tr/'inputs.json',rr['inputs_sha256']))
  for logfile in tr.glob('*.log'):bind(logfile)
  if rr['mapped']:assert rr['map_sha256']==sha(mapped)
 for name in ('audit_video_service_control_remap.py','validate_video_service_control_remap_v6.py','validate_video_service_control_remap_v8.py','audit_video_service_control_priority_map.py','place_video_service_control_priority.py','bind_video_service_control_cut.py','audit_video_service_control_priority_warnings.py','check_video_service_control_priority_map.py','control_remap_cut.py','run_installed_abc.py'):
  bind(HERE/name)
 ob=bind(ROOT/'tmp/frame-execution/video-service-owner-boundary25-control-priority/current-binding.json')
 assert ob['status']=='CURRENT_PROVED_NATIVE_PORT_BINDING' and ob['map_sha256']==sha(mapped) and ob['source_sha256']==sha(native/'native_source.v')
 bind(HERE/'validate_video_service_owner_boundary_v2.py',ob['checker_sha256']);bind(HERE/'validate_video_service_owner_boundary.py',ob['parent_checker_sha256'])
 sp=bind(board/'status-selection-proof.json')
 assert sp['front_bank_alias_proof']['reset_inductive_state_alias'] and sp['front_bank_alias_proof']['all_non_output_pins_identical'] and sp['front_bank_alias_proof']['all_parameters_identical']
 bind(HERE/'validate_video_service_narrow_status_selection_v2.py',sp['checker_sha256']);file_map(sp['inputs'])
 for script in ('validate_video_service_endpoint_ready_v2.py','validate_video_service_endpoint_cones_v2.py','validate_video_service_fetch_projection_v2.py','bind_video_service_mapped_cut.py','validate_video_service_endpoint_ready.py','validate_video_service_endpoint_cones.py','validate_video_service_fetch_projection.py','validate_video_service_fault_ready.py','validate_video_service_fault_ready_v2.py','validate_video_service_global_cones.py','package_video_service_fault_ready.py'):bind(HERE/script)

 inventory=bind(native/'a-zero-inventory.json');assert inventory['status']=='ACTUAL_A_PACKED_LOADER_ZERO_BITS_VERIFIED'
 assert inventory['byte_order_checks']==11232 and inventory['removed_storage']==246
 # No fabricated current A-cache test: preserve actual parent58 numerical
 # evidence and use full step-equivalence plus current vendor-LUT checks.
 for label in ('video-service-endpoint-batch-source1','video-service-endpoint-batch-mapped1'):
  wr=ROOT/'tmp/frame-execution'/label;wt=bind(wr/'result.json');assert wt['status']=='ACTUAL_A_ZERO_CACHE_GUARD_PASS'
  assert wt['source_sha256']==rp['parent_sha256'] and wt['valid_words']==11232
  file_map(bind(wr/'inputs.json',wt['inputs_sha256']))
  for logfile in wr.glob('*.log'):bind(logfile)
  if wt['mapped']:assert wt['map_sha256']==sha(ROOT/'tmp/frame-execution/connected-board-video-service-probe49-endpoint-batch/outflow/ti60f225_oob.map.v')
 ow=bind(ROOT/'tmp/frame-execution/video-service-owner-boundary25-control-priority/warning-port-audit.json')
 assert ow['status']=='EVERY_MISSING_PORT_UNUSED_OUTPUT_OR_BOOLEAN_INDEPENDENT_INPUT_PASS';file_map(ow['evidence'])
 for script in ('validate_video_service_weight_zero_factored.py','validate_video_service_weight_zero_data_reset.py','validate_video_service_weight_data_reset.py','audit_video_service_weight_zero_map.py','audit_video_service_weight_zero_map_v2.py','package_video_service_data_reset.py'):
  bind(HERE/script)
 receipt=bind(weight_parent/'data-reset-edit.json')"""),
]
for old,new in changes:
 assert old in code,old
 code=code.replace(old,new)
needle='for old,new in changes:';assert code.count(needle)==1
code=code.replace(needle,'changes.append(("assert ng[\'retained_primitives\']==54577", "assert ng[\'retained_primitives\']==54068"))\n'+needle)
exec(compile(code,str(source)+':guarded-a-zero-profile','exec'),globals())
