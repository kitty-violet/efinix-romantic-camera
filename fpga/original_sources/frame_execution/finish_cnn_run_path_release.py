"""All current run-control gates plus strict temporary-target provenance."""
from pathlib import Path
from control_command_repair import once
SOURCE=Path(__file__).with_name('finish_cnn_run_budget_release.py')
code=SOURCE.read_text(encoding='utf8')
code=code.replace('finish_cnn_run_budget_release','finish_cnn_run_path_release').replace('cnn_run_budget_c1_flow','cnn_run_budget_path_flow')
code=once(code,"dict(before='CONGESTION_2',after='CONGESTION_1')","dict(before='CONGESTION_2',after='CONGESTION_2')")
anchor=" base_extra(board,batch,physical,serial,bind,verify)"
extra="""
 from cnn_run_budget_path_flow import EXTRA,PROBE,verified_targets,route_code
 targets=verified_targets()
 restoration=bind(ROOT/'tmp/board-test/cnn-budget-path-restore1/result.json')
 assert restoration['status']=='ACTUAL_ROUTE_SDC_FINALLY_RESTORATION_THREE_CASES_PASS'
 assert [c['mode'] for c in restoration['cases']]==['success','tool_failure','host_timeout']
 assert all(c['exact_bytes_restored'] for c in restoration['cases'])
 verify(restoration['evidence'])
 receipt=bind(batch/'route_optimization_inputs.json')
 assert receipt['status']=='ORIGINAL_SDC_RESTORED_BEFORE_ACCEPTANCE' and receipt['target_check']==targets
 assert receipt['full_original_sta_required'] and not receipt['hardware_clock_changed'] and not receipt['placement_changed']
 assert receipt['extra']==EXTRA
 acceptance=batch/'route_acceptance.sdc';optimization=batch/'route_optimization.sdc'
 verify({str(acceptance):receipt['canonical_sha256'],str(optimization):receipt['optimization_sha256'],
  receipt['probe']:receipt['probe_sha256'],str(HERE/'cnn_run_budget_path_flow.py'):receipt['adapter_sha256']})
 assert acceptance.read_bytes()==(physical/'ti60f225_oob.sdc').read_bytes()
 assert optimization.read_text()==acceptance.read_text()+EXTRA
 assert (batch/'canonical-seed3/ti60f225_oob.sdc').read_bytes()==acceptance.read_bytes()
 expected=route_code()[1]
 assert (batch.parent/(batch.name+'-local-driver-route.py')).read_text()==expected
 assert (batch/'runner.py').read_text()==expected
"""
code=once(code,anchor,anchor+extra)
code=once(code,"manifest=dict(cnn_run_contract=payload_evidence(bind,verify),status=", "manifest=dict(route_optimization=bind(batch/'route_optimization_inputs.json'),cnn_run_contract=payload_evidence(bind,verify),status=")
exec(compile(code,str(SOURCE)+':run-budget-targeted-release','exec'),globals())
