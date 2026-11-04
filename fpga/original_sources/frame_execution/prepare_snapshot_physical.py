"""Two offline P&R workers; serialize STA; preserve every seed verdict."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from run_units import ROOT,HERE,sha
from prepare_snapshot_board import audit_delta
from windows_process_tree import stop_launched_tree


def main():
    p=argparse.ArgumentParser();p.add_argument('--board',required=True,type=Path)
    p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    board=a.board.resolve();root=a.root.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    delta=audit_delta(board);mapped=json.loads((board/'result.json').read_text())
    assert mapped['status']=='OBSERVATION_FULL_BOARD_MAP_NOT_RELEASE'
    assert sha(board/'inputs.json')==mapped['inputs_sha256']
    assert sha(board/'outflow/ti60f225_oob.map.v')==mapped['map_sha256']
    assert sha(board/'snapshot_sync_audit.json')==mapped['snapshot_sync_audit_sha256']
    assert json.loads((board/'snapshot_sync_audit.json').read_text())['status']=='SNAPSHOT_20_FF_SYNC_NO_SRL_PASS'
    root.mkdir();(root/'runner.py').write_bytes(Path(__file__).read_bytes())
    lock=threading.Lock();sta_lock=threading.Lock();stages=[];rows=[]
    def record(row):
        with lock:
            stages.append(row);(root/'stages.json').write_text(json.dumps(stages,indent=2))
    def run(name,args,budget):
        start=time.monotonic()
        with (root/(name+'.log')).open('w') as log:
            child=subprocess.Popen([sys.executable,str(HERE/args[0]),*map(str,args[1:])],
                                   cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            record(dict(stage=name,status='RUNNING',pid=child.pid,budget_s=budget))
            try:code=child.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                record(dict(stage=name,status='HOST_TIMEOUT',cleanup=stop_launched_tree(child)));raise
        row=dict(stage=name,status='EXIT_ZERO' if code==0 else 'FAIL',exit_code=code,
                 seconds=time.monotonic()-start,log_sha256=sha(root/(name+'.log')))
        record(row);assert code==0,row
    def seed_run(seed):
        physical=root/f'pnr-seed{seed}';canonical=root/f'canonical-seed{seed}'
        run(f'pnr{seed}',['probe_fourpll_reset_physical.py','--base',board,'--root',physical,
            '--logic-limit','60800','--optimization','CONGESTION_1','--seed',seed,'--fixed-dsp-layout',
            '--clock-profile-proof',ROOT/'tmp/frame-execution/axi-valid-payload-physical1/clock_crossing_inventory.json',
            '--setup-reserve-proof',ROOT/'tmp/frame-execution/setup-margin-proof2/result.json'],2220)
        result=json.loads((physical/'result.json').read_text())
        assert any(s['flow']=='pnr' and s['exit_code']==0 for s in result['stages'])
        with sta_lock:run(f'sta{seed}',['canonical_routed_sta.py','--base',physical,'--root',canonical],250)
        from audit_canonical_multiseed import audit
        checked=audit(canonical)
        assert checked['identity']['map']==mapped['map_sha256']
        row={k:checked[k] for k in ('seed','physical','canonical','current_timing_pass','wns_ns','whs_ns','resources')}
        with lock:
            rows.append(row);(root/'all_attempts.json').write_text(json.dumps(rows,indent=2))
        return row
    inputs=dict(board=str(board),delta=delta,map_sha256=mapped['map_sha256'],
                inputs_sha256=sha(board/'inputs.json'),runner_sha256=sha(__file__),
                seeds=[1,3,5,6],pnr_parallel=2,sta_parallel=1,hardware_access=False)
    (root/'inputs.json').write_text(json.dumps(inputs,indent=2))
    start=time.monotonic()
    try:
        for pair in ((1,3),(5,6)):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(seed_run,s) for s in pair]
                pair_rows=[f.result() for f in futures]
            assert all(r['current_timing_pass'] for r in pair_rows),('Stop after failed pair; diagnose before more seeds',pair_rows)
        passed=[r for r in rows if r['current_timing_pass']]
        assert len(passed)==4,('Insufficient passing seeds; preserve failures and diagnose',rows)
        receipt=ROOT/'output/toolchain/只读控制观察四seed原约束_2026-10-09.json'
        run('multiseed',['audit_canonical_multiseed.py','--roots',*[r['canonical'] for r in passed],
                        '--output',receipt],45)
        selected=max(passed,key=lambda r:(min(r['wns_ns'],r['whs_ns']),r['wns_ns']))
        result=dict(status='SNAPSHOT_FOUR_SEED_BASIC_TIMING_PASS_PENDING_CDC',seeds=rows,selected=selected,
                    multiseed_receipt=str(receipt),elapsed_s=time.monotonic()-start,
                    board_release=False,hardware_access=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(status='SNAPSHOT_PHYSICAL_BATCH_STOPPED',
            error=repr(exc),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-start,
            hardware_access=False),indent=2));raise


if __name__=='__main__':main()




