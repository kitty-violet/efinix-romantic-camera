"""One bounded additional seed pair; retain every failed routed receipt."""
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from run_units import ROOT, HERE, sha
from audit_canonical_multiseed import audit
from audit_measured_pinout import audit as pinout
from board_measured_pinmap import audit as source_audit
from windows_process_tree import stop_launched_tree


def main():
    prior = ROOT/'tmp/frame-execution/measured-pinmap-physical1'
    root = ROOT/'tmp/frame-execution/measured-pinmap-physical3'
    board = ROOT/'tmp/frame-execution/connected-board-measured-pinmap1'
    assert (prior/'failure.json').exists() and not root.exists()
    gate = source_audit(board)
    old = [audit(prior/f'canonical-seed{s}') for s in (1, 2, 3, 4)]
    assert [r['seed'] for r in old if r['current_timing_pass']] == [1, 3]
    assert all(r['identity'] == old[0]['identity'] for r in old)
    root.mkdir()
    (root/'runner.py').write_bytes(Path(__file__).read_bytes())
    records = []; lock = threading.Lock(); sta_lock = threading.Lock()
    def record(row):
        with lock:
            records.append(row)
            (root/'stages.json').write_text(json.dumps(records, indent=2))
    def run(name, script, args, budget):
        start = time.monotonic()
        with (root/(name+'.log')).open('w') as log:
            proc = subprocess.Popen([sys.executable, str(HERE/script), *map(str,args)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            record(dict(stage=name,status='RUNNING',pid=proc.pid,budget_s=budget))
            try: code=proc.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                record(dict(stage=name,status='HOST_TIMEOUT',cleanup=stop_launched_tree(proc)))
                raise
        record(dict(stage=name,status='EXIT_ZERO' if code==0 else 'FAIL',exit_code=code,
                    seconds=time.monotonic()-start,log_sha256=sha(root/(name+'.log'))))
        assert code==0, (name,code)
    def seed(s):
        physical=root/f'pnr-seed{s}'; canonical=root/f'canonical-seed{s}'
        run(f'pnr{s}', 'probe_measured_pinmap_physical.py',[
            '--base',board,'--root',physical,'--logic-limit',60800,'--optimization','CONGESTION_1',
            '--seed',s,'--fixed-dsp-layout','--clock-profile-proof',
            ROOT/'tmp/frame-execution/axi-valid-payload-physical1/clock_crossing_inventory.json',
            '--setup-reserve-proof',ROOT/'tmp/frame-execution/setup-margin-proof2/result.json'],2220)
        physical_result=json.loads((physical/'result.json').read_text())
        assert any(r['flow']=='pnr' and r['exit_code']==0 for r in physical_result['stages']), physical_result
        with sta_lock:
            run(f'sta{s}','canonical_routed_sta.py',['--base',physical,'--root',canonical],250)
        row=audit(canonical)
        assert row['identity']==old[0]['identity'], 'Configuration changed beyond seed'
        (root/f'pinout-seed{s}.json').write_text(json.dumps(pinout(physical),indent=2))
        return row
    (root/'inputs.json').write_text(json.dumps(dict(board=str(board),source_audit=gate,
        previous_batch=str(prior),previous_failure_sha256=sha(prior/'failure.json'),
        seeds=[5,6],parallel_processes=2,parallel_sta=1,hardware_access=False,
        runner_sha256=sha(__file__),tool_failure_preserved=str(ROOT/'tmp/frame-execution/measured-pinmap-physical2'),
        policy='One additional pair only; physical2 did not reach P&R because tool database access was denied'),indent=2))
    start=time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(seed,s) for s in (5,6)]
            extra=[f.result() for f in futures]
        attempts=old+extra
        (root/'all_attempts.json').write_text(json.dumps(attempts,indent=2))
        passing=[r for r in attempts if r['current_timing_pass']]
        assert len(passing)==4, 'Additional pair did not close four PASS seeds; diagnose structure, no blind rerun'
        receipt=ROOT/'output/toolchain/实测引脚修正版四seed原约束_2026-10-09.json'
        run('multiseed','audit_canonical_multiseed.py',[
            '--roots',*[r['canonical'] for r in passing],'--output',receipt],45)
        rows=[dict(seed=r['seed'],physical=r['physical'],canonical=r['canonical'],
                   wns=r['wns_ns'],whs=r['whs_ns']) for r in passing]
        selected=next(r for r in rows if r['seed']==1)
        gray=ROOT/'tmp/frame-execution/measured-pinmap-gray-seed1'
        bit=ROOT/'tmp/frame-execution/measured-pinmap-bit-seed1'
        assert json.loads((gray/'result.json').read_text())['status']=='NEW_ROUTE_GRAY_SKEW_PASS_NOT_FULL_CDC_RELEASE'
        assert json.loads((bit/'result.json').read_text())['status']=='OFFLINE_BITSTREAM_GENERATED_NOT_RELEASED'
        (root/'host-regression3.log').write_bytes((prior/'host-regression3.log').read_bytes())
        result=dict(status='OFFLINE_PHYSICAL_EVIDENCE_COMPLETE_AWAIT_REVIEW',seeds=rows,selected=selected,
            failed_seeds=[dict(seed=r['seed'],canonical=r['canonical'],wns=r['wns_ns'],whs=r['whs_ns'])
                          for r in attempts if not r['current_timing_pass']],
            all_attempts_sha256=sha(root/'all_attempts.json'),previous_batch=str(prior),
            multiseed_receipt=str(receipt),gray=str(gray),bitstream=str(bit),
            elapsed_s=time.monotonic()-start,board_release=False,hardware_access=False,
            next='Review all attempts including failures and prepare supervised diagnostic bundle')
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(status='SUPPLEMENT_STOPPED',error=repr(exc),
            traceback=traceback.format_exc(),elapsed_s=time.monotonic()-start,hardware_access=False),indent=2))
        raise


if __name__=='__main__':main()
