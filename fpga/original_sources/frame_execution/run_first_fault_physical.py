"""Selected-seed physical checks for observation-only full-board copy. No USB."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from windows_process_tree import stop_launched_tree
from first_fault_observer import BASE,audit_delta
from prepare_first_fault_board import check_proof
from audit_canonical_multiseed import audit

def main():
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,default=1);a=p.parse_args();board=a.board.resolve();root=a.root.resolve()
    audit_delta(board);proof=check_proof();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    (root/'runner.py').write_bytes(Path(__file__).read_bytes());stages=[]
    (root/'inputs.json').write_text(json.dumps(dict(board=str(board),board_inputs_sha256=sha(board/'inputs.json'),proof=proof,
        seed=a.seed,source_delta='first-fault read-only payload and OFS1 signature only',hardware_access=False,
        runner_sha256=sha(__file__)),indent=2))
    def run(name,script,args,budget):
        start=time.monotonic();logfile=root/(name+'.log')
        with logfile.open('x') as log:
            child=subprocess.Popen([sys.executable,str(HERE/script),*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            row=dict(stage=name,pid=child.pid,status='running',budget_s=budget);stages.append(row)
            (root/'stages.json').write_text(json.dumps(stages,indent=2))
            try:code=child.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                (root/(name+'_cleanup.json')).write_text(json.dumps(stop_launched_tree(child),indent=2));raise
        row.update(status='finished',exit_code=code,seconds=time.monotonic()-start,log_sha256=sha(logfile))
        (root/'stages.json').write_text(json.dumps(stages,indent=2));assert code==0,(name,logfile)
    try:
        run('startup','validate_first_fault_mapped.py',['--board',board,'--root',root/'startup'],180)
        physical=root/f'pnr-seed{a.seed}';canonical=root/f'canonical-seed{a.seed}';gray=root/f'gray-seed{a.seed}'
        run('physical','probe_fourpll_reset_physical.py',['--base',board,'--root',physical,'--pnr-budget',1500,
            '--logic-limit',60800,'--optimization','CONGESTION_2','--fixed-dsp-layout','--seed',a.seed,
            '--clock-profile-proof',ROOT/'tmp/frame-execution/axi-valid-payload-physical1/clock_crossing_inventory.json',
            '--setup-reserve-proof',ROOT/'tmp/frame-execution/setup-margin-proof2/result.json'],1950)
        pr=json.loads((physical/'result.json').read_text());assert all(r['exit_code']==0 for r in pr['stages'])
        # Reuse every mapped reset/periphery/pin check, substituting only the
        # exact, stricter observation-only source-delta audit for this version.
        import audit_control_state_implementation as impl
        impl.audit_delta=audit_delta
        implementation=impl.audit(board,physical)
        implementation['required_evidence'][str(HERE/'first_fault_observer.py')]=sha(HERE/'first_fault_observer.py')
        implementation['required_evidence'][str(Path(__file__))]=sha(__file__)
        (root/'implementation.json').write_text(json.dumps(implementation,indent=2))
        run('canonical','canonical_snapshot_sta.py',['--base',physical,'--root',canonical],270)
        selected=audit(canonical);(root/'selected.json').write_text(json.dumps(selected,indent=2))
        assert selected['current_timing_pass'],(selected['wns_ns'],selected['whs_ns'])
        run('gray','audit_snapshot_routed_gray.py',['--base',physical,'--root',gray],370)
        audit_delta(board)
        result=dict(status='FIRST_FAULT_SELECTED_ROUTE_PASS_PENDING_OFFLINE_BUNDLE',selected=selected,gray=str(gray),
                    functional_parent=str(BASE),hardware_access=False,board_download_allowed=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc(),hardware_access=False),indent=2));raise
if __name__=='__main__':main()
