"""Two independent offline jobs under a single enforced host resource budget."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
from run_units import ROOT,HERE,sha


def main():
    root=ROOT/'tmp/frame-execution/native-split-stage1'
    assert not root.exists();root.mkdir()
    board=ROOT/'tmp/frame-execution/connected-board-native-split1'
    from native_split_board import audit_delta
    audit_delta(board)
    for key in ('SALT_LICENSE_SERVER','MGLS_LICENSE_FILE','LM_LICENSE_FILE'):
        os.environ[key]='E:/quatra/LR-195771_License.dat'
    jobs={
        'place':['place_native_split_board.py','--root',ROOT/'tmp/frame-execution/native-split-place1','--base',board,
                 '--cnn-ram-reference',ROOT/'tmp/frame-execution/startup-cause-route1/canonical-seed3',
                 '--clock-profile-proof',ROOT/'tmp/frame-execution/axi-valid-payload-physical1/clock_crossing_inventory.json',
                 '--pnr-budget','1400','--logic-limit','60800','--optimization','CONGESTION_2','--fixed-dsp-layout','--seed','3'],
        'shorts':['run_native_split_evidence.py','--board',board,'--root',ROOT/'tmp/frame-execution/native-split-evidence1']}
    record=dict(runner_sha256=sha(__file__),board_inputs_sha256=sha(board/'inputs.json'),
                commands={k:list(map(str,v)) for k,v in jobs.items()},hardware_access=False)
    (root/'inputs.json').write_text(json.dumps(record,indent=2))
    def run(name,args):
        start=time.monotonic();cmd=[sys.executable,'-B',str(HERE/args[0]),*map(str,args[1:])]
        with (root/(name+'.log')).open('x') as log:
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            (root/(name+'-process.json')).write_text(json.dumps(dict(pid=p.pid,command=cmd)))
            rc=p.wait()
        value=dict(name=name,exit_code=rc,seconds=time.monotonic()-start,log_sha256=sha(root/(name+'.log')))
        (root/(name+'-result.json')).write_text(json.dumps(value,indent=2));return value
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda item:run(*item),jobs.items()))
    result=dict(status='NATIVE_PLACE_AND_SHORTS_PASS' if all(r['exit_code']==0 for r in results) else 'NATIVE_STAGE_HAS_FAILURE',results=results)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(result['status'])
    return 0 if all(r['exit_code']==0 for r in results) else 1


if __name__=='__main__':raise SystemExit(main())
