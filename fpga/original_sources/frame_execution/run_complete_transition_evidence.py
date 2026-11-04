"""Run same-map control gates alongside placement; stop before any route."""
import concurrent.futures,json,subprocess,sys,threading,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from complete_control_transition import audit_delta
from windows_process_tree import stop_launched_tree

def main():
    board=ROOT/'tmp/frame-execution/connected-board-complete-transition2'
    root=ROOT/'tmp/frame-execution/complete-transition-evidence2'
    mapped=ROOT/'tmp/frame-execution/complete-transition-mapped2'
    physical=ROOT/'tmp/frame-execution/complete-transition-place2'
    audit_delta(board)
    r=json.loads((board/'result.json').read_text())
    assert r['status']=='CONNECTED_BOARD_MAP_MEASURED_NOT_RELEASE' and r['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
    assert r['inputs_sha256']==sha(board/'inputs.json')
    audit=json.loads((board/'complete_transition_map_audit.json').read_text())
    assert audit['status']=='D_CONE_SHARED_CELL_REMAINS' and audit['map_sha256']==r['map_sha256']
    boundary=ROOT/'tmp/frame-execution/complete-transition-boundary2'
    reviewed=json.loads((boundary/'result.json').read_text())
    assert reviewed['status']=='COMPLETE_TRANSITION_LOCAL_STRUCTURE_AND_UPSTREAM_BOUNDARY_PASS'
    assert reviewed['map_sha256']==r['map_sha256'] and reviewed['inputs_sha256']==sha(boundary/'inputs.json')
    assert reviewed['boolean_result_sha256']==sha(boundary/'boolean_result.json')
    assert json.loads((boundary/'inputs.json').read_text())['strict_audit_sha256']==sha(board/'complete_transition_map_audit.json')
    eq=ROOT/'tmp/frame-execution/complete-transition-equivalence2'
    x=json.loads((eq/'result.json').read_text());assert x['status']=='COMPLETE_TRANSITION_CYCLE_EQUIVALENCE_PASS'
    assert x['inputs_sha256']==sha(eq/'inputs.json')
    assert json.loads((eq/'inputs.json').read_text())['board_inputs_sha256']==sha(board/'inputs.json')
    assert not root.exists() and not mapped.exists() and not physical.exists()
    root.mkdir();mapped.mkdir();stages=[];lock=threading.Lock()
    def save():
        with lock:(root/'stages.json').write_text(json.dumps(stages,indent=2))
    def run(name,script,args,budget):
        row=dict(name=name,status='running',budget_s=budget);stages.append(row);started=time.monotonic()
        with (root/(name+'.log')).open('x') as log:
            proc=subprocess.Popen([sys.executable,'-B',str(HERE/script),*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            row['pid']=proc.pid;save()
            try:code=proc.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                row.update(status='timeout',cleanup=stop_launched_tree(proc));save();raise
        row.update(status='pass' if code==0 else 'failed',exit_code=code,seconds=time.monotonic()-started);save()
        assert code==0,(name,code)
    def short():
        flow='complete_transition_flow.py'
        run('startup',flow,['startup','--board',board,'--root',mapped/'startup'],240)
        run('observe',flow,['observe','--source',mapped/'startup','--root',mapped/'observe'],180)
        run('serial',flow,['serial','--board',board,'--root',mapped.parent/(mapped.name+'-serial'),'--startup',mapped/'startup'],480)
    def place():
        run('place','complete_transition_physical_flow.py',['place','--base',board,'--root',physical,
            '--logic-limit',60800,'--pnr-budget',1800,'--optimization','CONGESTION_2','--seed',3,
            '--fixed-dsp-layout','--cnn-ram-reference',ROOT/'tmp/frame-execution/first-fault-reference-physical1/canonical-seed1',
            '--clock-profile-proof',ROOT/'tmp/frame-execution/axi-valid-payload-physical1/clock_crossing_inventory.json'],1950)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(short),pool.submit(place)];errors=[]
            for f in futures:
                try:f.result()
                except Exception as exc:errors.append(repr(exc))
        assert not errors,errors
        audit_delta(board)
        (root/'result.json').write_text(json.dumps(dict(status='COMPLETE_TRANSITION_SHORTS_PLACED_AWAIT_REVIEW',
            board_inputs_sha256=sha(board/'inputs.json'),map_sha256=r['map_sha256'],stages=stages,
            runner_sha256=sha(__file__),boundary_review_sha256=sha(boundary/'result.json'),hardware_access=False,board_release=False),indent=2))
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
