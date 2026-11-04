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
code=code.replace(anchor,anchor+".replace('from place_video_service_rq import audit_delta','from place_video_service_shared_frontier import audit_delta')"+'\nassert code.count("changes = [")==1\ncode=code.replace("changes = [",'+repr(insertion)+')')
changes=[
 ('video-service-native42-data-reset-only','video-service-native63c-shared-frontier'),
 ('video-service-mapped-boot21-data-reset','video-service-mapped-boot38-shared-frontier'),
 ('video-service-owner-boundary8-data-reset','video-service-owner-boundary27-shared-frontier'),
 ('video-service-data-reset-mapped-transport1','video-service-shared-frontier-mapped-transport1'),
 ('video-service-data-reset-fixture1','video-service-shared-frontier-fixture1'),
 ('video-service-data-reset-mapped1','video-service-shared-frontier-data-reset-mapped1'),
 (" receipt=bind(native/'data-reset-edit.json')", """ assert ng['weight_zero_removed']==246 and ng['weight_zero_guard_luts']==91
 weight_parent=Path(ng['weight_zero_parent']);assert weight_parent.name=='video-service-native42-data-reset-only'
 edit=bind(native/'weight-zero-edit.json');assert edit['latency_added']==0
 assert Path(edit['parent'])==weight_parent and edit['source_sha256']==sha(weight_parent/'native_source.v')
 independent=bind(native/'weight-zero-independent-audit.json')
 assert independent['status']=='GUARDED_A_ZERO_EXACT_SOURCE_GRAPH_PASS_NOT_PHYSICAL'
 assert independent['retained']==54331 and independent['removed']==246 and independent['added']==91
 assert independent['shared_frontier_proof']==str(native/'shared-frontier-independent-audit.json')
 assert independent['source_sha256']==sha(native/'native_source.v');file_map(independent['inputs'])
 current=bind(native/'shared-frontier-independent-audit.json')
 assert current['status']=='SHARED_FRONTIER_COMPOSITIONAL_CEC_PASS_NOT_PHYSICAL' and current['source_sha256']==sha(native/'native_source.v')
 assert current['all_original_sequential_state_exact'] and current['ports_protected_assigns_exact'] and current['extra_cycles']==0
 assert current['all_boolean_inputs'] and current['tampered_output_rejected']
 assert current['all_boundary_inputs']==1263 and current['all_exposed_outputs']==124
 file_map(current['inputs']);edit63=bind(native/'shared-frontier-edit.json');file_map(edit63['inputs'])
 synt=bind(ROOT/'tmp/frame-execution/video-service63c-syntax1/result.json');assert synt['status']=='ACTUAL_OPEN_VERILOG_SYNTAX_PASS_NOT_IP_ELABORATION';file_map(synt['inputs'])
 carry_edges=bind(native/'dedicated-carry-audit.json');assert carry_edges['status']=='ALL_DEDICATED_CARRY_OUTPUTS_ONLY_SINGLE_ADD_CI_OR_UNUSED' and carry_edges['source_sha256']==sha(native/'native_source.v') and carry_edges['edges']==4275
 parent62=ROOT/'tmp/frame-execution/video-service-native62-same-edge-enable'
 assert current['parent_sha256']==sha(parent62/'native_source.v') and Path(edit63['parent'])==parent62
 historical=bind(ROOT/'tmp/board-test/video-service-review1/current-functional-same-edge-1.json')
 assert historical['status']=='CURRENT_SERVICE_FUNCTIONAL_RELEASE_INPUTS_CHECKED_NOT_PHYSICAL_RELEASE';file_map(historical['evidence'])
 # The full Boolean cone proof composes with the parent62 legal-state proof.
 parentproof=bind(parent62/'same-edge-independent-audit.json');file_map(parentproof['inputs'])
 assert parentproof['legal_state_induction'] and parentproof['external_cycles_added']==0
 prior=ROOT/'tmp/frame-execution/video-service-native61-control-priority'
 rp=bind(prior/'control-remap-independent-audit.json');file_map(rp['inputs'])
 ar=ROOT/'tmp/frame-execution/video-service63c-source-delta1';rr=bind(ar/'result.json')
 assert rr['status']=='ACTUAL_VENDOR_SHARED_FRONTIER_CARRY_PASS' and rr['source_sha256']==sha(native/'native_source.v')
 assert rr['comparisons']==rr['vectors']*current['all_exposed_outputs'] and rr['vectors']==2+2*current['all_boundary_inputs']+4096
 assert rr['four_state_carry_exact']+rr['four_state_carry_X_refinements']==18432
 assert rr['four_state_carry_cases']==18432 and not rr['arbitrary_X_for_complete_cut']
 file_map(bind(ar/'inputs.json',rr['inputs_sha256']))
 for logfile in ar.glob('*.log'):bind(logfile)
 mr=ROOT/'tmp/frame-execution/video-service63-mapped-delta1';mm=bind(mr/'result.json')
 assert mm['status']=='ACTUAL_VENDOR_SHARED_FRONTIER_ALL_CHANGED_LUTS_MAP_PASS' and mm['source_sha256']==sha(native/'native_source.v') and mm['map_sha256']==sha(mapped)
 assert mm['luts']>0 and mm['four_state_cases_per_lut']==256 and mm['pruned_luts_four_state_proof']['all_aliases_four_state_proved']
 file_map(bind(mr/'inputs.json',mm['inputs_sha256']))
 for logfile in mr.glob('*.log'):bind(logfile)
 for name in ('audit_video_service_shared_frontier_carry.py','validate_video_service_shared_frontier_carry.py','audit_video_service_shared_frontier.py','validate_video_service_shared_frontier.py','audit_video_service_shared_frontier_warnings.py','check_video_service_shared_frontier_map.py','audit_video_service_shared_frontier_map.py','shared_frontier_map_aliases.py','place_video_service_shared_frontier.py','validate_video_service_shared_frontier_map.py','validate_video_service_shared_frontier_pages.py','shared_frontier_observation_binding.py','validate_video_service_same_edge_pages.py','validate_video_service_narrow_mapped_pages.py','validate_video_service_compact_mapped_pages.py'):bind(HERE/name)
 pg=bind(board/'pages-proof.json');ab=pg['apb_observation_binding']
 assert pg['checker_sha256']==sha(HERE/'validate_video_service_shared_frontier_pages.py')
 assert ab['legal_reset_same_edge_equivalence'] and ab['cycles_added']==0 and ab['current_cell']=='SERVICE62_APB_ENABLE' and ab['token']==['net','apb_enable',None]
 file_map(ab['evidence'])
 ob=bind(ROOT/'tmp/frame-execution/video-service-owner-boundary27-shared-frontier/current-binding.json')
 assert ob['status']=='CURRENT_PROVED_NATIVE_PORT_BINDING' and ob['map_sha256']==sha(mapped) and ob['source_sha256']==sha(native/'native_source.v')
 bind(HERE/'validate_video_service_owner_boundary_v2.py',ob['checker_sha256']);bind(HERE/'validate_video_service_owner_boundary.py',ob['parent_checker_sha256'])
 sp=bind(board/'status-selection-proof.json')
 assert sp['front_bank_alias_proof']['reset_inductive_state_alias'] and sp['front_bank_alias_proof']['all_non_output_pins_identical'] and sp['front_bank_alias_proof']['all_parameters_identical']
 bind(HERE/'validate_video_service_shared_frontier_status.py',sp['checker_sha256']);file_map(sp['inputs'])
 sab=sp['apb_observation_binding'];assert sab['old_observation_cone_boolean_cases']==2048 and sab['legal_onehot_cases']==20
 assert sab['old_state_boundary_identical'] and sab['legal_reset_same_edge_equivalence'] and sab['cycles_added']==0
 file_map(sab['evidence']);bind(HERE/'validate_video_service_narrow_status_selection_v2.py');bind(HERE/'shared_frontier_status_binding.py')
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
 ow=bind(ROOT/'tmp/frame-execution/video-service-owner-boundary27-shared-frontier/warning-port-audit.json')
 assert ow['status']=='EVERY_MISSING_PORT_UNUSED_OUTPUT_OR_BOOLEAN_INDEPENDENT_INPUT_PASS';file_map(ow['evidence'])
 for script in ('validate_video_service_weight_zero_factored.py','validate_video_service_weight_zero_data_reset.py','validate_video_service_weight_data_reset.py','audit_video_service_weight_zero_map.py','audit_video_service_weight_zero_map_v2.py','package_video_service_data_reset.py'):
  bind(HERE/script)
 receipt=bind(weight_parent/'data-reset-edit.json')"""),
]
for old,new in changes:
 assert old in code,old
 code=code.replace(old,new)
needle='for old,new in changes:';assert code.count(needle)==1
code=code.replace(needle,'changes.append(("assert ng[\'retained_primitives\']==54577", "assert ng[\'retained_primitives\']==54097"))\n'+needle)
exec(compile(code,str(source)+':guarded-a-zero-profile','exec'),globals())
