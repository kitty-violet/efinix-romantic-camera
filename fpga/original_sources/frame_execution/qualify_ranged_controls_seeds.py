"""Two independent tool processes at a time; first routed seed must already pass."""
import argparse,concurrent.futures,json,subprocess,sys,threading,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_canonical_multiseed import audit
from prepare_ranged_control_board import audit_delta,check_proofs
from windows_process_tree import stop_launched_tree

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    board=ROOT/'tmp/frame-execution/connected-board-ranged-controls1'
    first=ROOT/'tmp/frame-execution/ranged-controls-canonical1'
    good=audit(first);assert good['current_timing_pass'] and good['seed']==1
    audit_delta(board);check_proofs()
    root.mkdir();(root/'runner.py').write_bytes(Path(__file__).read_bytes())
    (root/'inputs.json').write_text(json.dumps(dict(first_result_sha256=sha(first/'result.json'),board_inputs_sha256=sha(board/'inputs.json'),runner_sha256=sha(__file__),seeds=[1,3,5,6],max_independent_processes=2,hardware_access=False),indent=2))
    stages=[];lock=threading.Lock();sta_lock=threading.Lock()
    def run(seed,name,script,args,budget):
        start=time.monotonic();logfile=root/f'{seed}_{name}.log'
        with logfile.open('x') as log:
            child=subprocess.Popen([sys.executable,str(HERE/script),*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            with lock:
                stages.append(dict(seed=seed,stage=name,pid=child.pid,status='running',budget_s=budget))
                (root/'stages.json').write_text(json.dumps(stages,indent=2))
            try:code=child.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                (root/f'{seed}_{name}_cleanup.json').write_text(json.dumps(stop_launched_tree(child),indent=2));raise
        with lock:
            next(s for s in stages if s['seed']==seed and s['stage']==name).update(status='finished',exit_code=code,seconds=time.monotonic()-start,log_sha256=sha(logfile))
            (root/'stages.json').write_text(json.dumps(stages,indent=2))
        assert code==0,(seed,name,logfile)
    def one(seed):
        physical=root/f'place-seed{seed}';canonical=root/f'canonical-seed{seed}'
        run(seed,'place','probe_global_place.py',['--base',board,'--root',physical,'--pnr-budget',600,'--logic-limit',60800,
            '--optimization','CONGESTION_1','--fixed-dsp-layout','--clock-profile-proof',ROOT/'tmp/frame-execution/axi-valid-payload-physical1/clock_crossing_inventory.json',
            '--seed',seed,'--setup-reserve-proof',ROOT/'tmp/frame-execution/setup-margin-proof2/result.json'],1020)
        screen=json.loads((physical/'placement_gate.json').read_text());assert screen['status']=='PLACEMENT_SCREEN_PASS_ROUTED_STA_REQUIRED',screen['reasons']
        run(seed,'route','route_screened_board.py',['--root',physical,'--startup-proof',ROOT/'tmp/frame-execution/ranged-controls-startup1',
            '--source-kind','range-local','--cluster-proof',ROOT/'tmp/frame-execution/ranged-controls-cluster-native1','--budget',900],990)
        with sta_lock:run(seed,'sta','canonical_snapshot_sta.py',['--base',physical,'--root',canonical],270)
        result=audit(canonical);assert result['current_timing_pass'],(seed,result['wns_ns'],result['whs_ns'])
        assert result['identity']==good['identity'],'Recipe differs beyond seed'
        return result
    try:
        # Check both results; do not hide a failure behind the other worker.
        pair=[];failures=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(one,s):s for s in (3,5)}
            for future in concurrent.futures.as_completed(futures):
                try:pair.append(future.result())
                except Exception as e:failures.append(dict(seed=futures[future],error=repr(e),traceback=traceback.format_exc()))
        if failures:
            (root/'failure.json').write_text(json.dumps(failures,indent=2));raise RuntimeError('Pair failed; seed6 not launched')
        rows=[good,*pair,one(6)]
        output=ROOT/'output/toolchain/ranged-controls-four-seed-20261010.json'
        run(0,'audit','audit_canonical_multiseed.py',['--roots',*[r['canonical'] for r in sorted(rows,key=lambda r:r['seed'])],'--output',output],90)
        result=dict(status='FOUR_SEED_FINISHED_NOT_BOARD_RELEASE',evidence=str(output),evidence_sha256=sha(output),hardware_access=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    except Exception as e:
        if not (root/'failure.json').exists():(root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2))
        raise
if __name__=='__main__':main()
