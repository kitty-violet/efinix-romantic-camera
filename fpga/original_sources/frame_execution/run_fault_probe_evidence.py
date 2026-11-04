"""Reuse the passing superset startup test, then serialize one-license gates."""
import json,shutil,subprocess,sys,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from windows_process_tree import stop_launched_tree
from finish_fault_probe_release import SHORTS,SERIAL,ADAPTER

def main():
    board=ROOT/'tmp/frame-execution/connected-board-fault-probe1'
    root=SHORTS;assert not root.exists();root.mkdir()
    sr=json.loads((ADAPTER/'result.json').read_text());si=json.loads((ADAPTER/'inputs.json').read_text())
    assert sr['status']=='ACTUAL_MAPPED_CONTROL_ADAPTER_IDLE_THREE_RESETS_PASS'
    assert sr['inputs_sha256']==sha(ADAPTER/'inputs.json') and sr['simulation_sha256']==sha(ADAPTER/'simulation.log')
    assert si['original_map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
    body=(ADAPTER/'tb.sv').read_text()
    for check in ('adapter idle','startup boot','unexpected wait-loop state','polling made no progress','MAPPED_STARTUP_RECOVERY_PASS'):
        assert check in body
    startup=root/'startup';startup.mkdir()
    for n,h in si['files'].items():
        assert sha(ADAPTER/n)==h;shutil.copy2(ADAPTER/n,startup/n)
    shutil.copy2(ADAPTER/'simulation.log',startup/'simulation.log')
    meta=dict(si,source_replay=str(ADAPTER),source_inputs_sha256=sha(ADAPTER/'inputs.json'),
              source_result_sha256=sha(ADAPTER/'result.json'),collator_sha256=sha(__file__),
              scope=si['scope']+'; superset includes actual adapter idle/reset and merged read RAM, protected reset explicitly modeled')
    (startup/'inputs.json').write_text(json.dumps(meta,indent=2))
    result=dict(sr,status='COMMAND_PROGRESS_MAPPED_STARTUP_PASS_PENDING_SERIAL_FAULT',
                inputs_sha256=sha(startup/'inputs.json'),source_result_sha256=sha(ADAPTER/'result.json'),
                startup_checks_reused_from_passing_superset=True)
    (startup/'result.json').write_text(json.dumps(result,indent=2))
    stages=[]
    try:
        for name,args,budget in (
            ('observe',['observe','--source',startup,'--root',root/'observe'],150),
            ('serial',['serial','--board',board,'--startup',startup,'--root',SERIAL.parent],300)):
            start=time.monotonic()
            with (root/(name+'.log')).open('x') as f:
                child=subprocess.Popen([sys.executable,str(HERE/'fault_probe_flow.py'),*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
                row=dict(stage=name,pid=child.pid,status='running',budget_s=budget);stages.append(row)
                (root/'stages.json').write_text(json.dumps(stages,indent=2))
                try:rc=child.wait(timeout=budget)
                except subprocess.TimeoutExpired:
                    (root/(name+'-timeout.json')).write_text(json.dumps(stop_launched_tree(child),indent=2));raise
            row.update(status='finished',exit_code=rc,seconds=time.monotonic()-start)
            (root/'stages.json').write_text(json.dumps(stages,indent=2));assert rc==0,(name,rc)
        for n in ('progress_counter_checks.json','paired_reply_result.json'):
            shutil.copy2(SERIAL.parent/n,SERIAL/n)
        (root/'result.json').write_text(json.dumps(dict(status='FAULT_PROBE_SAME_MAP_STARTUP_OBSERVE_SERIAL_PASS',
            map_sha256=sha(board/'outflow/ti60f225_oob.map.v'),runner_sha256=sha(__file__),hardware_access=False),indent=2))
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
