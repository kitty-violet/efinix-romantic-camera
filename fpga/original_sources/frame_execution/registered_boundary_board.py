"""Exact two-file boundary delta, bound to current local and accepted-A tests."""
import json
from pathlib import Path
from run_units import ROOT,sha
from split_logic_outputs_board import audit_delta as audit_parent

BASE=ROOT/'tmp/frame-execution/connected-board-split-outputs1'
BOARD=ROOT/'tmp/frame-execution/connected-board-split-registered2'

def audit_delta(board):
    assert board==BOARD;audit_parent(BASE)
    assert sha(board/'inputs.json')=='aed4adc0f7777c26256ef625ea1f2ee3957111179c3fdb5fd772f811214804c8'
    inputs=json.loads((board/'inputs.json').read_text())
    parent=json.loads((BASE/'inputs.json').read_text())
    expected={Path(r['snapshot']).relative_to(BASE).as_posix() for r in parent['inputs']}
    actual=set();changed=[]
    for row in inputs['inputs']:
        p=Path(row['snapshot']);assert p.is_relative_to(board) and sha(p)==row['sha256']
        rel=p.relative_to(board).as_posix();actual.add(rel)
        if sha(p)!=sha(BASE/rel):changed.append(rel)
    assert actual==expected and set(changed)=={'assembly/fixed_source_columns6.sv','assembly/fixed_rq32_mixed.sv'}
    bind=json.loads((board/'boundary_binding.json').read_text())
    for field in ('short_gates','rate_gate'):
        entry=bind[field];assert sha(Path(entry['path'])/'result.json')==entry['sha256']
    accepted=ROOT/'tmp/frame-execution/split-registered-accepted-cluster1'
    proof=json.loads((accepted/'result.json').read_text())
    assert proof['status']=='REGISTERED_BOUNDARY_ACCEPTED_A_CLUSTER_PASS'
    assert proof['inputs_sha256']==sha(accepted/'inputs.json') and proof['log_sha256']==sha(accepted/'simulation.log')
    assert json.loads((accepted/'inputs.json').read_text())['board_inputs_sha256']==sha(board/'inputs.json')
    return dict(status='REGISTERED_BOUNDARIES_EXACT_DELTA_ACCEPTED_A_SHORT_PASS',changed=changed,
        board_inputs_sha256=sha(board/'inputs.json'),accepted_a_proof_sha256=sha(accepted/'result.json'),
        pins_clocks_sdc_ip_unchanged=True,hardware_access=False,board_release=False)
