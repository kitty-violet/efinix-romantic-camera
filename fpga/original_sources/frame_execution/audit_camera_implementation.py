"""Full existing reset/pin audit, bound to the replacement camera wait counter."""
import inspect, json, sys
from pathlib import Path
from run_units import ROOT, HERE, sha
from control_command_repair import once
from camera_checked_board import audit_delta
import audit_control_state_implementation as prior

code=inspect.getsource(prior.audit)
code=once(code,"n.startswith('i2c_rst_cnt[')","n.startswith('camera_checked/wait_count[')")
code=once(code,'assert len(counters)==13','assert len(counters)==15')
scope=dict(prior.__dict__,audit_delta=audit_delta)
exec(compile(code,__file__+':replacement_counter','exec'),scope)

def audit(board,physical=None):
    result=scope['audit'](board,physical)
    result['camera_reset_counter']=dict(prefix='camera_checked/wait_count',bits=15,
        parent_counter_removed_by_bound_dead_cone_proof=True,
        source_auditor_sha256=sha(prior.__file__),adapter_sha256=sha(__file__))
    return result

if __name__=='__main__':
    from camera_checked_v2 import BOARD
    out=ROOT/'tmp/frame-execution/camera-checked-route2/camera_implementation_reaudit.json'
    assert not out.exists()
    result=audit(BOARD,ROOT/'tmp/frame-execution/camera-checked-bound2')
    out.write_text(json.dumps(result,indent=2));print(result['status'])
