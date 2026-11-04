"""Independent map and adjacent tensor regression, under one bounded Job."""
import concurrent.futures,json,subprocess,sys,time
from run_units import ROOT,HERE,sha
from validate_registered_board_cluster import BOARD,audit

RUN=ROOT/'tmp/frame-execution/split-registered-map-cluster1'

def main():
    audit();assert not RUN.exists();RUN.mkdir()
    commands={
      'map':[sys.executable,'-B',str(HERE/'map_prepared_board.py'),'--root',str(BOARD)],
      'cluster':[sys.executable,'-B',str(HERE/'validate_registered_board_cluster.py')]}
    (RUN/'inputs.json').write_text(json.dumps(dict(commands=commands,board_inputs_sha256=sha(BOARD/'inputs.json'),
      runner_sha256=sha(__file__),workers={n:sha(c[2]) for n,c in commands.items()},hardware_access=False),indent=2))
    def worker(name):
        start=time.monotonic()
        with (RUN/(name+'.log')).open('w',encoding='utf8') as log:
            p=subprocess.run(commands[name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1500)
        result=dict(exit_code=p.returncode,seconds=time.monotonic()-start,log_sha256=sha(RUN/(name+'.log')))
        (RUN/(name+'-result.json')).write_text(json.dumps(result,indent=2));return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures={k:pool.submit(worker,k) for k in commands}
        results={}
        for key,future in futures.items():
            try:results[key]=future.result()
            except Exception as error:results[key]=dict(error=repr(error))
    result=dict(status='MAP_AND_ADJACENT_REGRESSION_PASS' if all(v.get('exit_code')==0 for v in results.values()) else 'INCOMPLETE',
      results=results,physical_pass=False,board_release=False,hardware_access=False)
    (RUN/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return 0 if result['status']=='MAP_AND_ADJACENT_REGRESSION_PASS' else 1

if __name__=='__main__':raise SystemExit(main())
