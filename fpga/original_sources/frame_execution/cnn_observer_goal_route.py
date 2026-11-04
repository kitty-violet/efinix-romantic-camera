"""Single C2 route comparison retaining a proven stricter process setup goal."""
import json,runpy,sys
from pathlib import Path
from run_units import ROOT,sha
from control_command_repair import once
import cnn_observer_trim_margin_route as parent

def prepare_code():return parent.prepare_code()

def route_code():
    source,code=parent.route_code()
    needle="        row=run('route',['cmd.exe','/d','/c',cmd],physical,a.budget+45)"
    code=once(code,needle,"        from cnn_routing_goal import optimization_scope,verify_restored\n        with optimization_scope(physical):\n    "+needle+"\n        verify_restored(physical)")
    return source,code

def main():
    action=sys.argv.pop(1)
    if action=='--check-only':
        for fn in (prepare_code,route_code):compile(fn()[1],__file__,'exec')
        print('PROVEN_PROCESS_GOAL_ROUTE_RESTORE_GATES_COMPILE');return
    if action=='prepare':
        source,code=prepare_code();r=json.loads(Path(sys.argv[sys.argv.index('--review')+1]).read_text())
        assert r['route_profile_cli_override']==dict(before='CONGESTION_2',after='CONGESTION_2')
        root=Path(r['physical']);suffix='-prepare-cnn-budget'
    else:
        assert action=='route';source,code=route_code();root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();suffix='-local-driver-route'
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    driver=root.parent/(root.name+suffix+'.py');assert not driver.exists();driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),driver_sha256=sha(driver),adapter_sha256=sha(__file__),wrapper_sha256=sha(__file__),hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
