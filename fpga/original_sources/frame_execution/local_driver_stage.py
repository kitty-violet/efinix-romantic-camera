"""One bounded job: complete-board placement and same-map short evidence."""
import concurrent.futures,json,os,subprocess,sys,time
from run_units import ROOT,HERE,sha
from local_driver_board import audit_delta

def main():
    root=ROOT/'tmp/frame-execution/native-local-driver-stage1'
    board=ROOT/'tmp/frame-execution/connected-board-local-drivers1'
    audit_delta(board);assert not root.exists();root.mkdir()
    for key in ('SALT_LICENSE_SERVER','MGLS_LICENSE_FILE','LM_LICENSE_FILE'):os.environ[key]='E:/quatra/LR-195771_License.dat'
    jobs={
        'place':['place','--root',ROOT/'tmp/frame-execution/native-local-driver-place1','--base',board,
            '--cnn-ram-reference',ROOT/'tmp/frame-execution/startup-cause-route1/canonical-seed3',
            '--clock-profile-proof',ROOT/'tmp/frame-execution/axi-valid-payload-physical1/clock_crossing_inventory.json',
            '--pnr-budget','1400','--logic-limit','60800','--optimization','CONGESTION_2','--fixed-dsp-layout','--seed','3'],
        'shorts':['shorts','--board',board,'--root',ROOT/'tmp/frame-execution/native-local-driver-evidence1']}
    commands={k:[sys.executable,'-B',str(HERE/'local_driver_flow.py'),*map(str,v)] for k,v in jobs.items()}
    (root/'inputs.json').write_text(json.dumps(dict(commands=commands,board_inputs_sha256=sha(board/'inputs.json'),
        runner_sha256=sha(__file__),hardware_access=False),indent=2))
    def work(item):
        name,cmd=item;start=time.monotonic()
        with (root/(name+'.log')).open('x') as f:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            (root/(name+'-process.json')).write_text(json.dumps(dict(pid=child.pid,command=cmd)))
            rc=child.wait()
        row=dict(name=name,exit_code=rc,seconds=time.monotonic()-start,log_sha256=sha(root/(name+'.log')))
        (root/(name+'-result.json')).write_text(json.dumps(row,indent=2));return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(work,commands.items()))
    passed=all(r['exit_code']==0 for r in results)
    (root/'result.json').write_text(json.dumps(dict(status='LOCAL_DRIVER_PLACE_AND_SHORTS_PASS' if passed else 'LOCAL_DRIVER_STAGE_HAS_FAILURE',results=results),indent=2))
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
