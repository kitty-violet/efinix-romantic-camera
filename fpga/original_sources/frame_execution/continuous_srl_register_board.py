"""Map the four-cell substitution with complete native connectivity checks."""
from pathlib import Path
source=Path(__file__).with_name('cnn_timing_closure_board.py')
MAPPER_TEMPLATE=source
code=source.read_text(encoding='utf8')
changes=[
 ('from cnn_run_budget_board import BOARD as PARENT,audit_delta as parent_audit','from continuous_control_board_v2 import BOARD as PARENT,audit_delta as parent_audit'),
 ('from cnn_timing_closure_native import OUT as TRANSFORM','from continuous_srl_register_native import OUT as TRANSFORM'),
 ('connected-board-cnn-timing-closure1','connected-board-continuous-srl-register1'),
 ('CNN_TIMING_CLOSURE_FULL_BOARD_MAP_PASS','CONTINUOUS_SRL_REGISTER_FULL_BOARD_MAP_PASS'),
 ('CNN_TIMING_CLOSURE_FULL_VIDEO_A_MAP_BOUND','CONTINUOUS_SRL_REGISTER_FULL_VIDEO_A_MAP_BOUND'),
 ('timing_parent_sha256','srl_parent_sha256'),('timing_transform_sha256','srl_transform_sha256'),('timing_builder_sha256','srl_builder_sha256'),
 ('replications.json','replacement_contracts.json'),("proof['total_added_kinds']","proof['resource_delta']")]
for old,new in changes:
 assert old in code,old;code=code.replace(old,new)
begin=code.index('def transform_check():');end=code.index('\ndef audit_delta(',begin)
code=code[:begin]+'''def transform_check():
    r=json.loads((TRANSFORM/'result.json').read_text())
    assert r['status']=='FOUR_SLOW_SRL_EXACT_GRAPH_CANDIDATE_PENDING_ACTUAL_SLICE_TEST'
    assert r['latency_cycles_added']==0 and len(r['removed_cells'])==4
    assert r['source_sha256']==sha(TRANSFORM/'native_source.v')
    for f,h in r['inputs'].items():assert sha(f)==h,f
    short=ROOT/'tmp/frame-execution/continuous-srl-register-short1'
    sr=json.loads((short/'result.json').read_text())
    assert sr['status']=='ACTUAL_FOUR_NATIVE_SRL_EXPANSIONS_OFFICIAL_MODEL_SHORT_PASS'
    assert sr['inputs_sha256']==sha(short/'inputs.json') and sr['simulation_sha256']==sha(short/'simulate.log')
    wa=json.loads((short/'warning-audit.json').read_text())
    assert wa['status']=='ONLY_FOUR_ORIGINAL_UNUSED_Q7_PORT_WARNINGS' and wa['simulation_sha256']==sr['simulation_sha256'] and wa['other_model_warnings']==0
    for f,h in json.loads((short/'inputs.json').read_text())['files'].items():assert sha(f)==h,f
    return r
''' +code[end:]
# Keep the generic snapshot and all public/lexical/protected/count audits.
needle="TRANSFORM/'result.json',b/'public_connectivity_audit.json'"
assert code.count(needle)==1
code=code.replace(needle,"TRANSFORM/'result.json',ROOT/'tmp/frame-execution/continuous-srl-register-short1/result.json',ROOT/'tmp/frame-execution/continuous-srl-register-short1/warning-audit.json',MAPPER_TEMPLATE,b/'public_connectivity_audit.json'")
compile(code,str(source)+':four-srl-expansion','exec')
exec(compile(code,str(source)+':four-srl-expansion','exec'),globals())
