"""Same full route/STA/Gray gates on the joint-logic candidate, no RCF."""
import json,runpy,sys
from pathlib import Path
from run_units import sha
from control_command_repair import once
from cnn_run_budget_convergence_flow import prepare_code as old_prepare,route_code as old_route

def prepare_code():
    source,code=old_prepare();code=once(code,'from cnn_run_budget_board import audit_delta','from cnn_timing_closure_board import audit_delta')
    return source,code

def route_code():
    source,code=old_route();code=once(code,'from cnn_run_budget_board import audit_delta','from cnn_timing_closure_board import audit_delta')
    code=once(code,'import cnn_run_budget_implementation as impl','import cnn_timing_closure_implementation as impl')
    return source,code

def main():
    action=sys.argv.pop(1)
    if action=='--check-only':
        for k,fn in [('prepare',prepare_code),('route',route_code)]:compile(fn()[1],k,'exec')
        print('CLOSURE_ROUTE_ORIGINAL_GATES_COMPILE');return
    if action=='prepare':
        source,code=prepare_code();review=Path(sys.argv[sys.argv.index('--review')+1]).resolve();root=Path(json.loads(review.read_text())['physical']);suffix='-prepare-cnn-budget'
    else:
        assert action=='route';source,code=route_code();root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();suffix='-local-driver-route'
    driver=root.parent/(root.name+suffix+'.py');assert not driver.exists();compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),driver_sha256=sha(driver),adapter_sha256=sha(__file__),wrapper_sha256=sha(__file__),hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
