"""Real process tests: silent or chatty children cannot evade wall deadlines."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from host_runtime_budget import HostRuntimeBudget
from owned_simulator_process import spawn_owned, stop_owned_tree, close_owned
from run_units import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    root.mkdir()
    cases = []
    sentinel = spawn_owned([sys.executable,'-c','import time;time.sleep(60)'])
    try:
        for name, code in [
            ('silent_parent_and_child', "import subprocess,sys,time;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);print(p.pid,flush=True);time.sleep(60)"),
            ('continuous_output', "import time\nwhile True:\n print('progress',flush=True)\n time.sleep(.01)")]:
            case = root/name
            case.mkdir()
            begin = time.monotonic()
            proc = spawn_owned([sys.executable,'-u','-c',code],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            budget = HostRuntimeBudget(proc,1,case,begin).start()
            try:
                output,_ = proc.communicate(timeout=8)
                result = budget.finish()
                assert result and result['kind']=='HOST_TIME_BUDGET'
                assert result['status']=='HOST_BUDGET_STOP_REQUESTED'
                assert proc.returncode!=0 and sentinel.poll() is None
                assert time.monotonic()-begin<8
                assert json.loads((case/'host_timeout.json').read_text())['termination']
                cases.append(dict(name=name,status='PASS',seconds=time.monotonic()-begin))
            finally:
                if proc.poll() is None:stop_owned_tree(proc)
                budget.finish()
                close_owned(proc)
        case=root/'normal_completion';case.mkdir()
        proc=spawn_owned([sys.executable,'-c','print("done")'],stdout=subprocess.PIPE,text=True)
        budget=HostRuntimeBudget(proc,5,case).start()
        try:
            output,_=proc.communicate(timeout=5)
            assert budget.finish() is None and proc.returncode==0
            assert output.strip()=='done' and not (case/'host_timeout.json').exists()
            cases.append(dict(name='normal_completion',status='PASS'))
        finally:
            if proc.poll() is None:stop_owned_tree(proc)
            budget.finish();close_owned(proc)
        for seconds in [0,-1,14401,float('inf'),float('nan')]:
            try:HostRuntimeBudget(sentinel,seconds,root)
            except ValueError:pass
            else:raise AssertionError('Invalid budget accepted')
        cases.append(dict(name='invalid_limits_rejected',status='PASS',count=5))
        assert sentinel.poll() is None
    finally:
        stop_owned_tree(sentinel);sentinel.wait(timeout=8);close_owned(sentinel)
    result=dict(status='HOST_RUNTIME_BUDGET_PROCESS_TESTS_PASS',cases=cases,
        unrelated_process_survived=True,eda_simulation_launched=False)
    (root/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
